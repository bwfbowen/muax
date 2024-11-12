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

"""Defines the distributed MCTS agent topology via Launchpad."""

from typing import Callable, Optional, Sequence

import acme
from acme import datasets
from acme import specs
from acme.adders import reverb as adders
from muax.frameworks.acme.tf.mcts import acting
from muax.frameworks.acme.tf.mcts import learning
from muax.frameworks.acme.tf.mcts import models
from muax.frameworks.acme.tf.mcts import search
from acme.tf import utils as tf2_utils
from acme.tf import variable_utils as tf2_variable_utils
from acme.utils import counting
from acme.utils import loggers
import dm_env
import launchpad as lp
import reverb
import sonnet as snt
import tensorflow as tf


class DistributedMCTS:
  """Distributed MCTS agent."""

  def __init__(
      self,
      environment_factory: Callable[[], dm_env.Environment],
      network_factory: Callable[[specs.DiscreteArray], snt.Module],
      model_factory: Callable[[dm_env.Environment], models.Model],
      num_actors: int,
      num_simulations: int = 50,
      batch_size: int = 256,
      prefetch_size: int = 4,
      target_update_period: int = 100,
      samples_per_insert: float = 32.0,
      min_replay_size: int = 1000,
      max_replay_size: int = 1000000,
      importance_sampling_exponent: float = 0.2,
      priority_exponent: float = 0.6,
      n_step: int = 5,
      learning_rate: float = 1e-3,
      discount: float = 0.99,
      environment_spec: Optional[specs.EnvironmentSpec] = None,
      save_logs: bool = False,
      variable_update_period: int = 1000,
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

    if environment_spec is None:
      environment_spec = specs.make_environment_spec(environment_factory())

    # These 'factories' create the relevant components on the workers.
    self._environment_factory = environment_factory
    self._network_factory = network_factory
    self._model_factory = model_factory

    # Internalize hyperparameters.
    self._num_actors = num_actors
    self._num_simulations = num_simulations
    self._env_spec = environment_spec
    self._batch_size = batch_size
    self._prefetch_size = prefetch_size
    self._target_update_period = target_update_period
    self._samples_per_insert = samples_per_insert
    self._min_replay_size = min_replay_size
    self._max_replay_size = max_replay_size
    self._importance_sampling_exponent = importance_sampling_exponent
    self._priority_exponent = priority_exponent
    self._n_step = n_step
    self._learning_rate = learning_rate
    self._discount = discount
    self._save_logs = save_logs
    self._variable_update_period = variable_update_period

    # Store MCTS configuration
    self._mcts_policy = mcts_policy
    self._search_policy = search_policy
    self._value_transform = value_transform
    self._root_dirichlet_alpha = root_dirichlet_alpha
    self._root_exploration_fraction = root_exploration_fraction
    self._pb_c_base = pb_c_base
    self._pb_c_init = pb_c_init
    self._num_sampling_moves = num_sampling_moves
    self._temperature = temperature
    self._checkpoint = checkpoint
    self._save_directory = save_directory

  def replay(self):
    """The replay storage worker."""
    limiter = reverb.rate_limiters.SampleToInsertRatio(
        min_size_to_sample=self._min_replay_size,
        samples_per_insert=self._samples_per_insert,
        error_buffer=self._batch_size)
    extra_spec = {
        'pi':
            specs.Array(
                shape=(self._env_spec.actions.num_values,), dtype='float32')
    }
    signature = adders.NStepTransitionAdder.signature(self._env_spec,
                                                      extra_spec)
    replay_table = reverb.Table(
        name=adders.DEFAULT_PRIORITY_TABLE,
        sampler=reverb.selectors.Uniform(),
        remover=reverb.selectors.Fifo(),
        max_size=self._max_replay_size,
        rate_limiter=limiter,
        signature=signature)
    return [replay_table]

  def learner(self, replay: reverb.Client, counter: counting.Counter):
    """The learning part of the agent."""
    # Create the networks.
    network = self._network_factory(self._env_spec.actions)

    tf2_utils.create_variables(network, [self._env_spec.observations])

    # The dataset object to learn from.
    dataset = datasets.make_reverb_dataset(
        server_address=replay.server_address,
        batch_size=self._batch_size,
        prefetch_size=self._prefetch_size)

    # Create the optimizer.
    optimizer = snt.optimizers.Adam(self._learning_rate)

    # Return the learning agent.
    return learning.AZLearner(
        network=network,
        discount=self._discount,
        dataset=dataset,
        optimizer=optimizer,
        counter=counter,
        checkpoint=self._checkpoint,
        save_directory=self._save_directory,
    )

  def actor(
      self,
      replay: reverb.Client,
      variable_source: acme.VariableSource,
      counter: counting.Counter,
  ) -> acme.EnvironmentLoop:
    """The actor process."""

    # Build environment, model, network.
    environment = self._environment_factory()
    network = self._network_factory(self._env_spec.actions)
    model = self._model_factory(environment)

    # Create variable client for communicating with the learner.
    tf2_utils.create_variables(network, [self._env_spec.observations])
    variable_client = tf2_variable_utils.VariableClient(
        client=variable_source,
        variables={'network': network.trainable_variables},
        update_period=self._variable_update_period)

    # Component to add things into replay.
    adder = adders.NStepTransitionAdder(
        client=replay,
        n_step=self._n_step,
        discount=self._discount,
    )

    # Create the agent.
    actor = acting.MCTSActor(
        environment_spec=self._env_spec,
        model=model,
        network=network,
        discount=self._discount,
        adder=adder,
        variable_client=variable_client,
        num_simulations=self._num_simulations,
        mcts_policy=self._mcts_policy,
        search_policy=self._search_policy,
        value_transform=self._value_transform,
        root_dirichlet_alpha=self._root_dirichlet_alpha,
        root_exploration_fraction=self._root_exploration_fraction,
        pb_c_base=self._pb_c_base,
        pb_c_init=self._pb_c_init,
        num_sampling_moves=self._num_sampling_moves,
        temperature=self._temperature,
    )

    # Create the loop to connect environment and agent.
    return acme.EnvironmentLoop(environment, actor, counter)

  def evaluator(
      self,
      variable_source: acme.VariableSource,
      counter: counting.Counter,
  ):
    """The evaluation process."""

    # Build environment, model, network.
    environment = self._environment_factory()
    network = self._network_factory(self._env_spec.actions)
    model = self._model_factory(environment)

    # Create variable client for communicating with the learner.
    tf2_utils.create_variables(network, [self._env_spec.observations])
    variable_client = tf2_variable_utils.VariableClient(
        client=variable_source,
        variables={'policy': network.trainable_variables},
        update_period=self._variable_update_period)

    # Create the agent.
    actor = acting.MCTSActor(
        environment_spec=self._env_spec,
        model=model,
        network=network,
        discount=self._discount,
        variable_client=variable_client,
        num_simulations=self._num_simulations,
        mcts_policy=self._mcts_policy,
        search_policy=self._search_policy,
        value_transform=self._value_transform,
        root_dirichlet_alpha=self._root_dirichlet_alpha,
        root_exploration_fraction=self._root_exploration_fraction,
        pb_c_base=self._pb_c_base,
        pb_c_init=self._pb_c_init,
        num_sampling_moves=self._num_sampling_moves,
        temperature=self._temperature,
    )

    # Create the run loop and return it.
    logger = loggers.make_default_logger('evaluator')
    return acme.EnvironmentLoop(
        environment, actor, counter=counter, logger=logger)

  def build(self, name='MCTS'):
    """Builds the distributed agent topology."""
    program = lp.Program(name=name)

    with program.group('replay'):
      replay = program.add_node(lp.ReverbNode(self.replay), label='replay')

    with program.group('counter'):
      counter = program.add_node(
          lp.CourierNode(counting.Counter), label='counter')

    with program.group('learner'):
      learner = program.add_node(
          lp.CourierNode(self.learner, replay, counter), label='learner')

    with program.group('evaluator'):
      program.add_node(
          lp.CourierNode(self.evaluator, learner, counter), label='evaluator')

    with program.group('actor'):
      program.add_node(
          lp.CourierNode(self.actor, replay, learner, counter), label='actor')

    return program


class DistributedOpenSpielMCTS:
  """Distributed MCTS agent for OpenSpiel games like Go."""

  def __init__(
      self,
      environment_factory: Callable[[], dm_env.Environment],
      network_factory: Callable[[specs.DiscreteArray], snt.Module],
      model_factory: Callable[[dm_env.Environment], models.Model],
      num_actors: int = 5000,
      num_simulations: int = 800,
      batch_size: int = 4096,
      prefetch_size: int = 4,
      target_update_period: int = 100,
      samples_per_insert: float = 32.0,
      min_replay_size: int = 1000,
      max_replay_size: int = 500000,
      importance_sampling_exponent: float = 0.2,
      priority_exponent: float = 0.6,
      n_step: int = 1,
      learning_rate: float = 1e-3,
      discount: float = 1., # AlphaZero uses no discount in Go
      environment_spec: Optional[specs.EnvironmentSpec] = None,
      save_logs: bool = False,
      variable_update_period: int = 1000,
      *,
      # AlphaZero optimizer parameters
      learning_rate_boundaries: Sequence[int] = (100000, 300000, 500000),
      learning_rate_values: Sequence[float] = (0.2, 0.02, 0.002, 0.0002),
      momentum: float = 0.9,
      # MCTS parameters
      mcts_policy: Callable = search.mcts,
      search_policy: Callable = search.puct,
      value_transform: Optional[Callable[[float, int, int], float]] = None,
      root_dirichlet_alpha: float = 0.03,
      root_exploration_fraction: float = 0.25,
      pb_c_base: float = 19652,
      pb_c_init: float = 1.25,
      num_sampling_moves: int = 30,
      temperature: float = 1.0,
      checkpoint: bool = True,
      save_directory: str = '~/acme',
  ):

    if environment_spec is None:
      environment_spec = specs.make_environment_spec(environment_factory())

    # These 'factories' create the relevant components on the workers.
    self._environment_factory = environment_factory
    self._network_factory = network_factory
    self._model_factory = model_factory
    self._env_spec = environment_spec

    # Internalize hyperparameters.
    self._num_actors = num_actors
    self._num_simulations = num_simulations
    self._batch_size = batch_size
    self._prefetch_size = prefetch_size
    self._target_update_period = target_update_period
    self._samples_per_insert = samples_per_insert
    self._min_replay_size = min_replay_size
    self._max_replay_size = max_replay_size
    self._importance_sampling_exponent = importance_sampling_exponent
    self._priority_exponent = priority_exponent
    self._n_step = n_step
    self._learning_rate = learning_rate
    self._discount = discount
    self._save_logs = save_logs
    self._variable_update_period = variable_update_period

    # AlphaZero specific
    self._learning_rate_boundaries = learning_rate_boundaries
    self._learning_rate_values = learning_rate_values
    self._momentum = momentum
    # Store MCTS configuration
    self._mcts_policy = mcts_policy
    self._search_policy = search_policy
    self._value_transform = value_transform

    # MCTS configuration
    self._root_dirichlet_alpha = root_dirichlet_alpha
    self._root_exploration_fraction = root_exploration_fraction
    self._pb_c_base = pb_c_base
    self._pb_c_init = pb_c_init
    self._num_sampling_moves = num_sampling_moves
    self._temperature = temperature
    self._checkpoint = checkpoint
    self._save_directory = save_directory

  def replay(self):
    """The replay storage worker."""
    limiter = reverb.rate_limiters.SampleToInsertRatio(
        min_size_to_sample=self._min_replay_size,
        samples_per_insert=self._samples_per_insert,
        error_buffer=self._batch_size)
    extra_spec = {
        'pi':
            specs.Array(
                shape=(self._env_spec.actions.num_values,), dtype='float32')
    }
    signature = adders.NStepTransitionAdder.signature(self._env_spec,
                                                      extra_spec)
    replay_table = reverb.Table(
        name=adders.DEFAULT_PRIORITY_TABLE,
        sampler=reverb.selectors.Uniform(),
        remover=reverb.selectors.Fifo(),
        max_size=self._max_replay_size,
        rate_limiter=limiter,
        signature=signature)
    return [replay_table]

  def learner(self, replay: reverb.Client, counter: counting.Counter):
    """The learning part of the agent."""
    # Create the networks.
    network = self._network_factory(self._env_spec.actions)

    tf2_utils.create_variables(network, [self._env_spec.observations])
    # Create learning rate schedule
    lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=self._learning_rate_boundaries,
        values=self._learning_rate_values
    )

    # Create optimizer
    optimizer = snt.optimizers.Momentum(
        learning_rate=lr_schedule,
        momentum=self._momentum
    )

    # Create dataset
    dataset = datasets.make_reverb_dataset(
        server_address=replay.server_address,
        batch_size=self._batch_size,
        prefetch_size=self._prefetch_size
    )

    # Return the learning agent.
    return learning.AZLearner(
        network=network,
        discount=self._discount,
        dataset=dataset,
        optimizer=optimizer,
        counter=counter,
        checkpoint=self._checkpoint,
        save_directory=self._save_directory,
    )

  def actor(
      self,
      replay: reverb.Client,
      variable_source: acme.VariableSource,
      counter: counting.Counter,
  ) -> acme.EnvironmentLoop:
    """The actor process."""

    # Build environment, model, network.
    environment = self._environment_factory()
    network = self._network_factory(self._env_spec.actions)
    model = self._model_factory(environment)

    # Create variable client for communicating with the learner.
    tf2_utils.create_variables(network, [self._env_spec.observations])
    variable_client = tf2_variable_utils.VariableClient(
        client=variable_source,
        variables={'network': network.trainable_variables},
        update_period=self._variable_update_period)

    # Component to add things into replay.
    adder = adders.NStepTransitionAdder(
        client=replay,
        n_step=self._n_step,
        discount=self._discount,
    )

    # Create the agent.
    actor = acting.MCTSActor(
        environment_spec=self._env_spec,
        model=model,
        network=network,
        discount=self._discount,
        adder=adder,
        variable_client=variable_client,
        num_simulations=self._num_simulations,
        mcts_policy=self._mcts_policy,
        search_policy=self._search_policy,
        value_transform=self._value_transform,
        root_dirichlet_alpha=self._root_dirichlet_alpha,
        root_exploration_fraction=self._root_exploration_fraction,
        pb_c_base=self._pb_c_base,
        pb_c_init=self._pb_c_init,
        num_sampling_moves=self._num_sampling_moves,
        temperature=self._temperature,
    )

    # Create the loop to connect environment and agent.
    return acme.EnvironmentLoop(environment, actor, counter)

  def evaluator(
      self,
      variable_source: acme.VariableSource,
      counter: counting.Counter,
  ):
    """The evaluation process."""

    # Build environment, model, network.
    environment = self._environment_factory()
    network = self._network_factory(self._env_spec.actions)
    model = self._model_factory(environment)

    # Create variable client for communicating with the learner.
    tf2_utils.create_variables(network, [self._env_spec.observations])
    variable_client = tf2_variable_utils.VariableClient(
        client=variable_source,
        variables={'policy': network.trainable_variables},
        update_period=self._variable_update_period)

    # Create the agent.
    actor = acting.MCTSActor(
        environment_spec=self._env_spec,
        model=model,
        network=network,
        discount=self._discount,
        variable_client=variable_client,
        num_simulations=self._num_simulations,
        mcts_policy=self._mcts_policy,
        search_policy=self._search_policy,
        value_transform=self._value_transform,
        root_dirichlet_alpha=0.0,  # No exploration in evaluation
        root_exploration_fraction=0.0,
        pb_c_base=self._pb_c_base,
        pb_c_init=self._pb_c_init,
        num_sampling_moves=0,  # No random sampling
        temperature=0.25,  # greedier
    )

    # Create the run loop and return it.
    logger = loggers.make_default_logger('evaluator', steps_key='evaluator_steps')
    return acme.EnvironmentLoop(
        environment, actor, counter=counter, logger=logger)

  def build(self, name='MCTS'):
    """Builds the distributed agent topology."""
    program = lp.Program(name=name)

    with program.group('replay'):
      replay = program.add_node(lp.ReverbNode(self.replay), label='replay')

    with program.group('counter'):
      counter = program.add_node(
          lp.CourierNode(counting.Counter), label='counter')

    with program.group('learner'):
      learner = program.add_node(
          lp.CourierNode(self.learner, replay, counter), label='learner')

    with program.group('evaluator'):
      program.add_node(
          lp.CourierNode(self.evaluator, learner, counter), label='evaluator')

    with program.group('actor'):
      # Add multiple actors
      for _ in range(self._num_actors):
        program.add_node(lp.CourierNode(self.actor, replay, learner, counter))

    return program
