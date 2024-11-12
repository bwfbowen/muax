# Copyright 2018 DeepMind Technologies Limited. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Example running Stochastic MuZero on discrete control tasks."""
from typing import Callable

import datetime
import math

from absl import flags
from acme import specs
from muax.frameworks.acme.jax import stochastic_muzero
import helpers
from absl import app
from acme.jax import experiments
from acme.jax import inference_server as inference_server_lib
from acme.utils import lp_utils
import dm_env
import launchpad as lp


ENV_NAME = flags.DEFINE_string('env_name', 'classic|CartPole', 'Environment type and name, e.g., "classic|CartPole" or "jumanji|Game2048-v1"')
SEED = flags.DEFINE_integer('seed', 42, 'Random seed.')
NUM_STEPS = flags.DEFINE_integer(
    'num_steps', 2_000_000, 'Number of env steps to run.'
)
EVAL_EVERY = flags.DEFINE_integer(
    'eval_every', 10_000,
    'How often (in actor environment steps) to run evaluation episodes.')
EVAL_EPISODES = flags.DEFINE_integer(
    'evaluation_episodes', 1,
    'Number of evaluation episodes to run periodically.')
NUM_LEARNERS = flags.DEFINE_integer('num_learners', 1, 'Number of learners.')
NUM_ACTORS = flags.DEFINE_integer('num_actors', 4, 'Number of actors.')
NUM_ACTORS_PER_NODE = flags.DEFINE_integer(
    'num_actors_per_node',
    2,
    'Number of colocated actors',
)
RUN_DISTRIBUTED = flags.DEFINE_bool(
    'run_distributed', False, 'Should an agent be executed in a distributed '
    'way. If False, will run single-threaded.',)


def build_experiment_config() -> experiments.ExperimentConfig:
  """Builds experiment config which can be executed in different ways."""
  env_type, env_name = ENV_NAME.value.split('|')
  muzero_config = stochastic_muzero.SMZConfig()
  
  def get_env_factory(env_type: str, env_name: str) -> Callable[[int], dm_env.Environment]:
    if env_type == 'classic':
      return lambda seed: helpers.make_classiccontrol_environment(level=env_name, seed=seed)
    elif env_type == 'jumanji':
      return lambda seed: helpers.make_jumanji_environment(level=env_name, seed=seed)
    elif env_type == 'atari':
      return lambda seed: helpers.make_atari_environment(
          level=env_name,
          sticky_actions=True,
          zero_discount_on_life_loss=True,
          num_stacked_frames=1,
          grayscaling=False,
          to_float=False,
          seed=seed
      )
    else:
      raise ValueError(f"Unknown environment type: {env_type}")

  env_factory = get_env_factory(env_type, env_name)

  def network_factory(
      spec: specs.EnvironmentSpec,
  ) -> stochastic_muzero.SmzNetworks:
    return stochastic_muzero.make_mlp_networks(
      spec, 
      full_support_size=muzero_config.full_support_size,
      vmin=muzero_config.vmin,
      vmax=muzero_config.vmax,
      representation_layer_sizes = (16, 64, 64, 64),
      prediction_layer_sizes = (16, 64, 64, 64),
      dynamic_layer_sizes = (16, 64, 64, 64),
      )

  # Construct the builder.
  env_spec = specs.make_environment_spec(env_factory(SEED.value))
  extra_spec = {
      stochastic_muzero.POLICY_PROBS_KEY: specs.Array(
          shape=(env_spec.actions.num_values,), dtype='float32'
      ),
      stochastic_muzero.RAW_VALUES_KEY: specs.Array(shape=(), dtype='float32'),
  }
  muzero_builder = stochastic_muzero.SmzBuilder(  # pytype: disable=wrong-arg-types  # jax-ndarray
      muzero_config,
      extra_spec,
  )

  checkpointing_config = experiments.CheckpointingConfig(
      replay_checkpointing_time_delta_minutes=20,
      time_delta_minutes=1,
  )
  return experiments.ExperimentConfig(
      builder=muzero_builder,
      environment_factory=env_factory,
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
        num_actors=NUM_ACTORS.value,
        num_learner_nodes=NUM_LEARNERS.value,
        num_actors_per_node=NUM_ACTORS_PER_NODE.value,)
    lp.launch(program, xm_resources=lp_utils.make_xm_docker_resources(program,),)


if __name__ == '__main__':
  app.run(main)
