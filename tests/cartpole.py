from absl import app
from absl import flags
import jax
import jax.numpy as jnp
import gym
import muax
from muax import nn
from muax.frameworks.coax.model import MuZero
import optax

FLAGS = flags.FLAGS
flags.DEFINE_integer('num_episodes', 100, 'Number of episodes to train')
flags.DEFINE_integer('num_simulations', 5, 'Number of MCTS simulations')
flags.DEFINE_float('discount', 0.99, 'Discount factor')
flags.DEFINE_integer('support_size', 10, 'Support size for value representation')
flags.DEFINE_integer('embedding_size', 8, 'Embedding size for representation')
flags.DEFINE_integer('buffer_size', 500, 'Size of replay buffer')
flags.DEFINE_integer('batch_size', 32, 'Batch size for training')
flags.DEFINE_float('learning_rate', 0.02, 'Learning rate for optimizer')


def main(argv):
    del argv  # Unused

    # Environment setup
    env = gym.make('CartPole-v1')
    num_actions = env.action_space.n

    # Network setup
    full_support_size = int(FLAGS.support_size * 2 + 1)
    repr_fn = nn._init_representation_func(nn.Representation, FLAGS.embedding_size)
    pred_fn = nn._init_prediction_func(nn.Prediction, num_actions, full_support_size)
    dy_fn = nn._init_dynamic_func(nn.Dynamic, FLAGS.embedding_size, num_actions, full_support_size)

    # MuZero setup
    gradient_transform = optax.adam(FLAGS.learning_rate)
    model = MuZero(repr_fn, pred_fn, dy_fn, policy='muzero', discount=FLAGS.discount,
                   optimizer=gradient_transform, support_size=FLAGS.support_size)

    # Replay buffer and tracer setup
    buffer = muax.TrajectoryReplayBuffer(FLAGS.buffer_size)
    tracer = muax.PNStep(10, FLAGS.discount, 0.5)

    # Training loop
    key = jax.random.PRNGKey(0)
    for episode in range(FLAGS.num_episodes):
        obs, _ = env.reset()
        done = False
        trajectory = muax.Trajectory()
        episode_reward = 0

        while not done:
            key, subkey = jax.random.split(key)
            action = model.act(subkey, obs, num_simulations=FLAGS.num_simulations)
            next_obs, reward, done, _, _ = env.step(action)
            
            trajectory.append(obs, action, reward, done)
            episode_reward += reward
            obs = next_obs

        buffer.add(trajectory)
        tracer.trace(trajectory)

        if len(buffer) >= FLAGS.batch_size:
            batch = buffer.sample(FLAGS.batch_size)
            loss_info = model.update(batch)
            print(f"Episode {episode + 1}, Reward: {episode_reward}, Loss: {loss_info['loss']}")
        else:
            print(f"Episode {episode + 1}, Reward: {episode_reward}")

    env.close()

if __name__ == '__main__':
    app.run(main)
