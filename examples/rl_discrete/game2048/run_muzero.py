"""Example running MuZero on Game2048-v1."""

from typing import Sequence

from absl import app
from absl import flags
from acme import specs
from acme.jax import experiments
from acme.jax import types as acme_types
from acme.utils import lp_utils
import jax
import jumanji
import launchpad as lp
import muzero
import helpers

# Flags
SEED = flags.DEFINE_integer('seed', 42, 'Random seed.')
NUM_STEPS = flags.DEFINE_integer('num_steps', 1_000_000, 'Number of env steps to run.')
EVAL_EVERY = flags.DEFINE_integer('eval_every', 10_000, 'How often to run evaluation episodes.')
EVAL_EPISODES = flags.DEFINE_integer('evaluation_episodes', 10, 'Number of evaluation episodes to run.')
NUM_ACTORS = flags.DEFINE_integer('num_actors', 4, 'Number of actors.')
RUN_DISTRIBUTED = flags.DEFINE_bool('run_distributed', False, 'Should an agent be executed in a distributed way.')

def temperature_fn(max_training_steps, training_steps):
  r"""Determines the randomness for the action taken by the model"""
  if training_steps < 1e5:
      return 1.0
  elif training_steps < 2e5:
      return 0.5
  elif training_steps < 3e5:
      return 0.1
  else:
      return 0
  
class Game2048MZConfig(muzero.MZConfig):
    def __init__(self):
        super().__init__()
        self.num_simulations = 100
        self.batch_size = 1024
        self.num_bootstrapping = 10
        self.bootstrapping_lambda = 0.5
        self.max_replay_size = 125_000
        self.sequence_length = 200
        self.min_replay_size = 1000
        self.learning_rate = 3e-4
        self.discount = 0.999
        self.dirichlet_alpha = 0.25
        self.dirichlet_fraction = 0.1
        self.temperature_fn = temperature_fn
        self.full_support_size = 601
        self.vmin = 0.
        self.vmax = 600.

def build_experiment_config() -> experiments.ExperimentConfig:
    """Builds experiment config which can be executed in different ways."""
    muzero_config = Game2048MZConfig()
    
    def environment_factory(seed: int):
        return helpers.make_jumanji_environment(level='Game2048-v1', seed=seed)

    def network_factory(spec: specs.EnvironmentSpec) -> muzero.MzNetworks:
        return muzero.make_fully_connect_resnet_networks(
            spec, 
            embedding_dim=256,
            num_blocks=10,
            full_support_size=muzero_config.full_support_size,
            vmin=muzero_config.full_support_size,
            vmax=muzero_config.vmax,
            )

    env_spec = specs.make_environment_spec(environment_factory(SEED.value))
    extra_spec = {
        muzero.POLICY_PROBS_KEY: specs.Array(
            shape=(env_spec.actions.num_values,), dtype='float32'
        ),
        muzero.RAW_VALUES_KEY: specs.Array(shape=(), dtype='float32'),
    }
    muzero_builder = muzero.MzBuilder(
        muzero_config,
        extra_spec,
    )

    checkpointing_config = experiments.CheckpointingConfig(
      replay_checkpointing_time_delta_minutes=20,
      time_delta_minutes=1,
    )

    return experiments.ExperimentConfig(
        builder=muzero_builder,
        environment_factory=environment_factory,
        network_factory=network_factory,
        seed=SEED.value,
        max_num_actor_steps=NUM_STEPS.value,
        checkpointing=checkpointing_config,
    )

def main(_):
    experiment_config = build_experiment_config()

    if not RUN_DISTRIBUTED.value:
        experiments.run_experiment(
            experiment=experiment_config,
            eval_every=EVAL_EVERY.value,
            num_eval_episodes=EVAL_EPISODES.value)
    else:
        program = experiments.make_distributed_experiment(
            experiment=experiment_config,
            num_actors=NUM_ACTORS.value)
        lp.launch(program, xm_resources=lp_utils.make_xm_docker_resources(program))

if __name__ == '__main__':
    app.run(main)