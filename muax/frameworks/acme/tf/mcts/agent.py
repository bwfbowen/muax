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

"""A single-process MCTS agent."""
from typing import Callable, Optional

from acme import datasets
from acme import specs
from acme.adders import reverb as adders
from acme.agents import agent
from muax.frameworks.acme.tf.mcts import acting
from muax.frameworks.acme.tf.mcts import learning
from muax.frameworks.acme.tf.mcts import models
from muax.frameworks.acme.tf.mcts import search
from acme.tf import utils as tf2_utils

import numpy as np
import reverb
import sonnet as snt


class MCTS(agent.Agent):
  """A single-process MCTS agent."""

  def __init__(
      self,
      network: snt.Module,
      model: models.Model,
      optimizer: snt.Optimizer,
      n_step: int,
      discount: float,
      replay_capacity: int,
      num_simulations: int,
      environment_spec: specs.EnvironmentSpec,
      batch_size: int,
      *,
      mcts_policy: Callable = search.mcts,
      search_policy: Callable = search.puct,
      value_transform: Optional[Callable[[float, int, int], float]] = None,
      root_dirichlet_alpha: float = 0.03,
      root_exploration_fraction: float = 0.25,
      pb_c_base: float = 19652,
      pb_c_init: float = 1.25,
      num_sampling_moves: Optional[int] = None,
      temperature: float = 1.0,
      checkpoint: bool = True,
      save_directory: str = '~/acme',
  ):

    extra_spec = {
        'pi':
            specs.Array(
                shape=(environment_spec.actions.num_values,), dtype=np.float32)
    }
    # Create a replay server for storing transitions.
    replay_table = reverb.Table(
        name=adders.DEFAULT_PRIORITY_TABLE,
        sampler=reverb.selectors.Uniform(),
        remover=reverb.selectors.Fifo(),
        max_size=replay_capacity,
        rate_limiter=reverb.rate_limiters.MinSize(1),
        signature=adders.NStepTransitionAdder.signature(
            environment_spec, extra_spec))
    self._server = reverb.Server([replay_table], port=None)

    # The adder is used to insert observations into replay.
    address = f'localhost:{self._server.port}'
    adder = adders.NStepTransitionAdder(
        client=reverb.Client(address),
        n_step=n_step,
        discount=discount)

    # The dataset provides an interface to sample from replay.
    dataset = datasets.make_reverb_dataset(server_address=address)
    dataset = dataset.batch(batch_size, drop_remainder=True)

    tf2_utils.create_variables(network, [environment_spec.observations])

    # Now create the agent components: actor & learner.
    actor = acting.MCTSActor(
        environment_spec=environment_spec,
        model=model,
        network=network,
        discount=discount,
        adder=adder,
        num_simulations=num_simulations,
        mcts_policy=mcts_policy,
        search_policy=search_policy,
        value_transform=value_transform,
        root_dirichlet_alpha=root_dirichlet_alpha,
        root_exploration_fraction=root_exploration_fraction,
        pb_c_base=pb_c_base,
        pb_c_init=pb_c_init,
        num_sampling_moves=num_sampling_moves,
        temperature=temperature,
    )

    learner = learning.AZLearner(
        network=network,
        optimizer=optimizer,
        dataset=dataset,
        discount=discount,
        checkpoint=checkpoint,
        save_directory=save_directory,
    )

    # The parent class combines these together into one 'agent'.
    super().__init__(
        actor=actor,
        learner=learner,
        min_observations=10,
        observations_per_step=1,
    )


class ReprMCTS(agent.Agent):
  """A single-process MCTS agent with representation for raw observation."""

  def __init__(
      self,
      repr_network: snt.Module,
      eval_network: snt.Module,
      model: models.Model,
      optimizer: snt.Optimizer,
      n_step: int,
      discount: float,
      replay_capacity: int,
      num_simulations: int,
      environment_spec: specs.EnvironmentSpec,
      batch_size: int,
  ):

    extra_spec = {
        'pi':
            specs.Array(
                shape=(environment_spec.actions.num_values,), dtype=np.float32)
    }
    # Create a replay server for storing transitions.
    replay_table = reverb.Table(
        name=adders.DEFAULT_PRIORITY_TABLE,
        sampler=reverb.selectors.Uniform(),
        remover=reverb.selectors.Fifo(),
        max_size=replay_capacity,
        rate_limiter=reverb.rate_limiters.MinSize(1),
        signature=adders.NStepTransitionAdder.signature(
            environment_spec, extra_spec))
    self._server = reverb.Server([replay_table], port=None)

    # The adder is used to insert observations into replay.
    address = f'localhost:{self._server.port}'
    adder = adders.NStepTransitionAdder(
        client=reverb.Client(address),
        n_step=n_step,
        discount=discount)

    # The dataset provides an interface to sample from replay.
    dataset = datasets.make_reverb_dataset(server_address=address)
    dataset = dataset.batch(batch_size, drop_remainder=True)

    repr_output_spec = tf2_utils.create_variables(repr_network, [environment_spec.observations])
    eval_output_spec = tf2_utils.create_variables(eval_network, [repr_output_spec])

    # Now create the agent components: actor & learner.
    actor = acting.ReprMCTSActor(
        environment_spec=environment_spec,
        model=model,
        repr_network=repr_network,
        eval_network=eval_network,
        discount=discount,
        adder=adder,
        num_simulations=num_simulations,
    )

    learner = learning.MZLearner(
        repr_network=repr_network,
        eval_network=eval_network,
        optimizer=optimizer,
        dataset=dataset,
        discount=discount,
    )

    # The parent class combines these together into one 'agent'.
    super().__init__(
        actor=actor,
        learner=learner,
        min_observations=10,
        observations_per_step=1,
    )


class SampledMCTS(agent.Agent):
  """A single-process sampled MCTS agent."""

  def __init__(
      self,
      network: snt.Module,
      model: models.Model,
      optimizer: snt.Optimizer,
      n_step: int,
      discount: float,
      replay_capacity: int,
      num_simulations: int,
      environment_spec: specs.EnvironmentSpec,
      batch_size: int, 
      k_bins: int = 5,
      num_samples: int = 20,
  ):

    if isinstance(environment_spec.actions, specs.BoundedArray):
        _pi_shape = (np.prod(environment_spec.actions.shape), k_bins)
        num_samples = np.prod(_pi_shape) if num_samples is None else num_samples
    elif isinstance(environment_spec.actions, specs.DiscreteArray):
        _pi_shape = (1, environment_spec.actions.num_values) 
        num_samples = environment_spec.actions.num_values if num_samples is None else num_samples

    extra_spec = {
        'pi':
            specs.Array(
                shape=_pi_shape, dtype=np.float32)
    }
    # Create a replay server for storing transitions.
    replay_table = reverb.Table(
        name=adders.DEFAULT_PRIORITY_TABLE,
        sampler=reverb.selectors.Uniform(),
        remover=reverb.selectors.Fifo(),
        max_size=replay_capacity,
        rate_limiter=reverb.rate_limiters.MinSize(1),
        signature=adders.NStepTransitionAdder.signature(
            environment_spec, extra_spec))
    self._server = reverb.Server([replay_table], port=None)

    # The adder is used to insert observations into replay.
    address = f'localhost:{self._server.port}'
    adder = adders.NStepTransitionAdder(
        client=reverb.Client(address),
        n_step=n_step,
        discount=discount)

    # The dataset provides an interface to sample from replay.
    dataset = datasets.make_reverb_dataset(server_address=address)
    dataset = dataset.batch(batch_size, drop_remainder=True)

    tf2_utils.create_variables(network, [environment_spec.observations])

    # Now create the agent components: actor & learner.
    actor = acting.SampledMCTSActor(
        environment_spec=environment_spec,
        model=model,
        network=network,
        discount=discount,
        adder=adder,
        num_simulations=num_simulations,
        pi_shape=_pi_shape,
        num_samples=num_samples,
    )

    learner = learning.AZLearner(
        network=network,
        optimizer=optimizer,
        dataset=dataset,
        discount=discount,
    )

    # The parent class combines these together into one 'agent'.
    super().__init__(
        actor=actor,
        learner=learner,
        min_observations=10,
        observations_per_step=1,
    )
