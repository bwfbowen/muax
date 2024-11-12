from typing import List, Optional

from absl import app
from absl import flags

import os 
import acme
from acme import wrappers
from acme.tf import savers
from acme import specs
from acme import types
from muax.frameworks.acme.tf import mcts
from muax.frameworks.acme.tf.mcts.models import simulator
from muax.frameworks.acme.tf.mcts import search
from acme.wrappers import open_spiel_wrapper
from open_spiel.python import rl_environment
from acme.environment_loops import open_spiel_environment_loop
import sonnet as snt
import tensorflow as tf
import numpy as np
import dm_env
import launchpad as lp


flags.DEFINE_string('game', 'go', 'Name of the game')
flags.DEFINE_bool('run_distributed', True, 'run_distributed')
flags.DEFINE_integer('num_actors', 5000, 'Number of self-play actors')
flags.DEFINE_integer('num_simulations', 800, 'Number of MCTS simulations per move')
flags.DEFINE_integer('training_steps', 700000, 'Number of training steps')
flags.DEFINE_integer('checkpoint_interval', 1000, 'Number of steps between checkpoints')
flags.DEFINE_integer('window_size', 1000000, 'Self-play window size')
flags.DEFINE_integer('batch_size', 4096, 'Batch size for training')
flags.DEFINE_float('weight_decay', 1e-4, 'L2 weight decay')
flags.DEFINE_float('momentum', 0.9, 'Momentum for optimizer')
flags.DEFINE_integer('replay_capacity', 500000, 'Replay buffer capacity')
flags.DEFINE_float('root_dirichlet_alpha', 0.03, 'Dirichlet noise alpha parameter')
flags.DEFINE_float('root_exploration_fraction', 0.25, 'Root prior exploration noise fraction')
flags.DEFINE_float('pb_c_base', 19652, 'UCB formula constant')
flags.DEFINE_float('pb_c_init', 1.25, 'UCB formula constant')
flags.DEFINE_string('search_policy', 'puct', 'Search policy to use: puct, ucb/pucb, or ltr/pltr/pnltr')
flags.DEFINE_integer('num_sampling_moves', 30, 'Number of moves for temperature sampling')
flags.DEFINE_integer('max_moves', 722, 'Maximum number of moves per game for Go')
flags.DEFINE_integer('board_size', 19, 'Board size of game Go')
flags.DEFINE_float('temperature', 10., 'Temperature for action selection')
flags.DEFINE_string('checkpoint_dir', '~/acme', 'checkpoint model save path')

FLAGS = flags.FLAGS

class AlphaZeroWrapper(open_spiel_wrapper.OpenSpielWrapper):
    def __init__(self, environment: rl_environment.Environment, history_size: int = 8):
        super().__init__(environment)
        self._history_size = history_size
        self._board_size = FLAGS.board_size
        self._state_history = []
        self._num_planes = self._history_size * 2 + 1  # 2 planes per history step + 1 for current player

    def _convert_obs(self, observations: List[open_spiel_wrapper.OLT]) -> List[open_spiel_wrapper.OLT]:
        # Extract the current board state (4 planes: black, white, empty, current player)
        # Here each observation in observations is identical(perfect information game), 
        # please refer to https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/games/go/go.cc#L109
        current_state = observations[0].observation.reshape(self._board_size, self._board_size, 4)
        
        # Update state history (we'll only use the first 2 planes for history)
        self._state_history.append(current_state)
        if len(self._state_history) > self._history_size:
            self._state_history.pop(0)
        
        # Construct the n-plane representation
        alphazero_observation = self._construct_alphazero_planes()
        
        # Update the observation in the OLT named tuple for both players
        new_observations = []
        for obs in observations:
            new_obs = open_spiel_wrapper.OLT(
                observation=alphazero_observation,
                legal_actions=obs.legal_actions,
                terminal=obs.terminal
            )
            new_observations.append(new_obs)
        
        return new_observations

    def _construct_alphazero_planes(self):
        observation = np.zeros((self._board_size, self._board_size, self._num_planes), dtype=np.float32)
        
        for i, state in enumerate(reversed(self._state_history)):
            if i >= self._history_size:
                break
            observation[:, :, i*2] = state[:, :, 0]  # Black stones
            observation[:, :, i*2+1] = state[:, :, 1]  # White stones
        
        # Set the current player plane
        current_player = self._state_history[-1][:, :, 3]  # Use the 4th plane from the most recent state
        observation[:, :, -1] = current_player
        
        return observation

    def _initialize_state_history(self):
        empty_state = np.zeros((self._board_size, self._board_size, self._num_planes), dtype=np.float32)
        self._state_history = [empty_state] * self._history_size

    def observation_spec(self):
        spec = super().observation_spec()
        new_shape = (self._board_size, self._board_size, self._num_planes)
        return open_spiel_wrapper.OLT(
            observation=dm_env.specs.BoundedArray(shape=new_shape, dtype=np.float32, name='observation', minimum=0, maximum=1),
            legal_actions=spec.legal_actions,
            terminal=spec.terminal
        )

    def reset(self) -> dm_env.TimeStep:
        timestep = super().reset()
        self._initialize_state_history()
        return self._update_timestep(timestep)

    def step(self, action: types.NestedArray) -> dm_env.TimeStep:
        timestep = super().step(action)
        return self._update_timestep(timestep)

    def _update_timestep(self, timestep: dm_env.TimeStep) -> dm_env.TimeStep:
        new_observation = self._convert_obs(timestep.observation)
        return dm_env.TimeStep(
            step_type=timestep.step_type,
            reward=timestep.reward,
            discount=timestep.discount,
            observation=new_observation
        )

def make_network(
        action_spec: specs.DiscreteArray,
        board_size=19,
        num_residual_blocks=19, 
        num_filters=256,
        gpus: Optional[list] = None,
    ):
    gpus = tf.config.list_physical_devices('GPU') if gpus is None else gpus 
    num_gpus = len(gpus)
    device_map = None
    if num_gpus > 0:
        device_map = {
            'initial': f'/gpu:0',
            'residual': [f'/gpu:{i % num_gpus}' for i in range(num_residual_blocks)],
            'policy': f'/gpu:{(num_gpus - 2) % num_gpus if num_gpus > 1 else 0}',
            'value': f'/gpu:{(num_gpus - 1) % num_gpus if num_gpus > 1 else 0}'
        }
    class AlphaZeroGoNetwork(snt.Module):
        def __init__(self):
            super().__init__()
            self.board_size = board_size
            self.num_residual_blocks = num_residual_blocks
            self.num_filters = num_filters

            # Initial convolutional layer
            if device_map:
                with tf.device(device_map['initial']):
                    self._create_initial_layers()
            else:
                self._create_initial_layers()

            # Residual blocks
            self.residual_blocks = []
            for i in range(num_residual_blocks):
                if device_map:
                    with tf.device(device_map['residual'][i]):
                        self.residual_blocks.append(self.ResidualBlock())
                else:
                    self.residual_blocks.append(self.ResidualBlock())

            # Policy head
            if device_map:
                with tf.device(device_map['policy']):
                    self._create_policy_head()
            else:
                self._create_policy_head()

            # Value head
            if device_map:
                with tf.device(device_map['value']):
                    self._create_value_head()
            else:
                self._create_value_head()

        def __call__(self, inputs, is_training=True):
            observation, legal_actions, terminal = inputs.observation, inputs.legal_actions, inputs.terminal

            # Initial conv
            if device_map:
                with tf.device(device_map['initial']):
                    x = self._initial_forward(observation, is_training)
            else:
                x = self._initial_forward(observation, is_training)

            # Residual blocks
            for i, block in enumerate(self.residual_blocks):
                if device_map:
                    with tf.device(device_map['residual'][i]):
                        x = block(x, is_training=is_training)
                else:
                    x = block(x, is_training=is_training)

            # Policy head
            if device_map:
                with tf.device(device_map['policy']):
                    policy = self._policy_forward(x, legal_actions, is_training)
            else:
                policy = self._policy_forward(x, legal_actions, is_training)

            # Value head
            if device_map:
                with tf.device(device_map['value']):
                    value = self._value_forward(x, terminal, is_training)
            else:
                value = self._value_forward(x, terminal, is_training)

            return policy, value
        
        def _create_initial_layers(self):
            self.conv_initial = snt.Conv2D(num_filters, 3, padding='SAME')
            self.batch_norm_initial = snt.BatchNorm(create_scale=True, create_offset=True)
        
        def _create_policy_head(self):
            self.policy_conv = snt.Conv2D(2, 1)
            self.policy_batch_norm = snt.BatchNorm(create_scale=True, create_offset=True)
            self.policy_fc = snt.Linear(action_spec.num_values)  # board * board +1 for pass move
        
        def _create_value_head(self):
            self.value_conv = snt.Conv2D(1, 1)
            self.value_batch_norm = snt.BatchNorm(create_scale=True, create_offset=True)
            self.value_fc1 = snt.Linear(256)
            self.value_fc2 = snt.Linear(1)
        
        def _initial_forward(self, observation, is_training):
            x = self.conv_initial(observation)
            x = self.batch_norm_initial(x, is_training=is_training)
            return tf.nn.relu(x)

        def _policy_forward(self, x, legal_actions, is_training):
            policy = self.policy_conv(x)
            policy = self.policy_batch_norm(policy, is_training=is_training)
            policy = tf.nn.relu(policy)
            policy = tf.reshape(policy, [-1, self.board_size * self.board_size * 2])
            policy = self.policy_fc(policy)
            policy = tf.where(legal_actions > 0, policy, tf.float32.min)
            return tf.nn.softmax(policy)

        def _value_forward(self, x, terminal, is_training):
            value = self.value_conv(x)
            value = self.value_batch_norm(value, is_training=is_training)
            value = tf.nn.relu(value)
            value = tf.reshape(value, [-1, self.board_size * self.board_size])
            value = self.value_fc1(value)
            value = tf.nn.relu(value)
            value = self.value_fc2(value)
            value = tf.nn.tanh(value)
            return tf.where(terminal > 0.5, 0.0, value)

        class ResidualBlock(snt.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = snt.Conv2D(num_filters, 3, padding='SAME')
                self.batch_norm1 = snt.BatchNorm(create_scale=True, create_offset=True)
                self.conv2 = snt.Conv2D(num_filters, 3, padding='SAME')
                self.batch_norm2 = snt.BatchNorm(create_scale=True, create_offset=True)

            def __call__(self, inputs, is_training=True):
                x = self.conv1(inputs)
                x = self.batch_norm1(x, is_training=is_training)
                x = tf.nn.relu(x)
                x = self.conv2(x)
                x = self.batch_norm2(x, is_training=is_training)
                return tf.nn.relu(x + inputs)

    return AlphaZeroGoNetwork()


def make_environment():
    env_configs = {
            'max_game_length': FLAGS.max_moves,
            'komi': 7.5,
            'board_size': FLAGS.board_size,
        }
    raw_environment = rl_environment.Environment(FLAGS.game, **env_configs)
    environment = AlphaZeroWrapper(raw_environment)
    environment = wrappers.SinglePrecisionWrapper(environment)
    return environment


def make_model(env):
    return simulator.OpenSpielSimulator(env)


def get_search_policy(policy_name):
    if policy_name == 'puct':
        return search.puct
    elif policy_name == 'pucb':
        return search.pucb
    elif policy_name == 'pltr':
        return search.pltr
    elif policy_name == 'pnltr':
        return search.pnltr
    elif policy_name == 'ltr':
        return search.ltr
    elif policy_name == 'ucb':
        return search.ucb
    

def main(_):

    environment = make_environment()
    environment_spec = acme.make_environment_spec(environment)
    
    if not FLAGS.run_distributed:

        search_policy = get_search_policy(FLAGS.search_policy)

        # Create the network
        network = make_network(environment_spec.actions, board_size=19, num_residual_blocks=19, num_filters=256)

        # Create a learning rate schedule
        lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            boundaries=[100000, 300000, 500000],
            values=[0.2, 0.02, 0.002, 0.0002]
        )

        checkpoint_dir = os.path.join(FLAGS.checkpoint_dir, FLAGS.search_policy)

        def make_agent():
            return mcts.MCTS(
                network=network,
                environment_spec=environment_spec,
                model=simulator.OpenSpielSimulator(environment),
                optimizer=snt.optimizers.Momentum(learning_rate=lr_schedule, momentum=FLAGS.momentum),
                search_policy=search_policy,
                value_transform=search.zero_sum_transform,
                n_step=1,  
                discount=1.0,  # AlphaZero uses no discounting
                replay_capacity=FLAGS.replay_capacity,
                num_simulations=FLAGS.num_simulations,
                batch_size=FLAGS.batch_size,
                mcts_policy=search.open_spiel_mcts,
                root_dirichlet_alpha=FLAGS.root_dirichlet_alpha,
                root_exploration_fraction=FLAGS.root_exploration_fraction,
                pb_c_base=FLAGS.pb_c_base,
                pb_c_init=FLAGS.pb_c_init,
                num_sampling_moves=FLAGS.num_sampling_moves,
                temperature=FLAGS.temperature,
                checkpoint=True,
                save_directory=checkpoint_dir,
                )

        agent1 = make_agent()
        agent2 = make_agent()
        
        agents = [agent1, agent2]

        loop = open_spiel_environment_loop.OpenSpielEnvironmentLoop(
            environment=environment,
            actors=agents,
            should_update=True,
        )
        loop.run(num_episodes=FLAGS.training_steps)
    else:
        program_builder = mcts.DistributedMCTS(
            environment_factory=make_environment,
            network_factory=make_network,
            model_factory=make_model,
            optimizer=snt.optimizers.Momentum(learning_rate=lr_schedule, momentum=FLAGS.momentum),
            n_step=1,  
            discount=1.0,  # AlphaZero uses no discounting
            replay_capacity=FLAGS.replay_capacity,
            num_simulations=FLAGS.num_simulations,
            batch_size=FLAGS.batch_size,
            search_policy=search_policy,
            root_dirichlet_alpha=FLAGS.root_dirichlet_alpha,
            root_exploration_fraction=FLAGS.root_exploration_fraction,
            pb_c_base=FLAGS.pb_c_base,
            pb_c_init=FLAGS.pb_c_init,
            num_sampling_moves=FLAGS.num_sampling_moves,
            temperature=FLAGS.temperature,
            checkpoint=True,
            save_directory=checkpoint_dir,
        )
        lp.launch(programs=program_builder.build())

if __name__ == '__main__':
    app.run(main)
