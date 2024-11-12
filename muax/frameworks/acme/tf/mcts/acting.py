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

"""A MCTS actor."""

from typing import Optional, Tuple, List, Callable

import acme
from acme import adders
from acme import specs
from muax.frameworks.acme.tf.mcts import models
from muax.frameworks.acme.tf.mcts import search
from muax.frameworks.acme.tf.mcts import types
from acme.tf import variable_utils as tf2_variable_utils
from acme.tf import utils as tf2_utils

import dm_env
import numpy as np
from scipy import special
import sonnet as snt
import tensorflow as tf


class MCTSActor(acme.Actor):
  """Executes a policy- and value-network guided MCTS search."""

  _prev_timestep: dm_env.TimeStep

  def __init__(
      self,
      environment_spec: specs.EnvironmentSpec,
      model: models.Model,
      network: snt.Module,
      discount: float,
      num_simulations: int,
      adder: Optional[adders.Adder] = None,
      variable_client: Optional[tf2_variable_utils.VariableClient] = None,
      mcts_policy: Callable = search.mcts,
      search_policy: Callable = search.puct,
      value_transform: Optional[Callable[[float, int, int], float]] = None,
      root_dirichlet_alpha: float = 0.03,
      root_exploration_fraction: float = 0.25,
      pb_c_base: float = 19652,
      pb_c_init: float = 1.25,
      num_sampling_moves: Optional[int] = None,
      temperature: float = 1.0,
  ):

    # Internalize components: model, network, data sink and variable source.
    self._model = model
    self._network = tf.function(network)
    self._variable_client = variable_client
    self._adder = adder

    # Internalize hyperparameters.
    self._num_actions = environment_spec.actions.num_values
    self._num_simulations = num_simulations
    self._actions = list(range(self._num_actions))
    self._discount = discount
    self._mcts_policy = mcts_policy
    self._search_policy = search_policy
    self._value_transform = value_transform
    self._root_dirichlet_alpha = root_dirichlet_alpha
    self._root_exploration_fraction = root_exploration_fraction
    self._pb_c_base = pb_c_base
    self._pb_c_init = pb_c_init
    self._num_sampling_moves = np.inf if num_sampling_moves is None else num_sampling_moves
    self._temperature = temperature
    self._move_count = 0

    # We need to save the policy so as to add it to replay on the next step.
    self._probs = np.ones(
        shape=(self._num_actions,), dtype=np.float32) / self._num_actions

  def _forward(
      self, observation: types.Observation) -> Tuple[types.Probs, types.Value]:
    """Performs a forward pass of the policy-value network."""
    logits, value = self._network(tf2_utils.add_batch_dim(observation))

    # Convert to numpy & take softmax.
    logits = logits.numpy().squeeze(axis=0)
    value = value.numpy().item()
    probs = special.softmax(logits)

    return probs, value

  def select_action(self, observation: types.Observation) -> types.Action:
    """Computes the agent's policy via MCTS."""
    if self._model.needs_reset:
      self._model.reset(observation)
    
    # Base MCTS parameters
    mcts_kwargs = {
        'observation': observation,
        'model': self._model,
        'search_policy': self._search_policy,
        'evaluation': self._forward,
        'num_simulations': self._num_simulations,
        'num_actions': self._num_actions,
        'discount': self._discount,
        'root_dirichlet_alpha': self._root_dirichlet_alpha,
        'root_exploration_fraction': self._root_exploration_fraction,
        'pb_c_base': self._pb_c_base,
        'pb_c_init': self._pb_c_init,
    }

    # Add value_transform if using open_spiel_mcts
    if self._mcts_policy == search.open_spiel_mcts and self._value_transform is not None:
        mcts_kwargs['value_transform'] = self._value_transform

    # Compute fresh MCTS plan using selected policy
    root = self._mcts_policy(**mcts_kwargs)
    
    # Always sample if num_sampling_moves is np.inf, otherwise sample for the first num_sampling_moves
    if self._move_count < self._num_sampling_moves:
        probs = search.visit_count_policy(root, temperature=self._temperature)
        action = np.random.choice(self._actions, p=probs)
    else:
        action = np.argmax([child.visit_count for child in root.children.values()])

    self._move_count += 1

    # Save the policy probs so that we can add them to replay in `observe()`.
    self._probs = probs.astype(np.float32)

    return np.int32(action)

  def update(self, wait: bool = False):
    """Fetches the latest variables from the variable source, if needed."""
    if self._variable_client:
      self._variable_client.update(wait)

  def observe_first(self, timestep: dm_env.TimeStep):
    self._prev_timestep = timestep
    if self._adder:
      self._adder.add_first(timestep)
    self._move_count = 0

  def observe(self, action: types.Action, next_timestep: dm_env.TimeStep):
    """Updates the agent's internal model and adds the transition to replay."""
    self._model.update(self._prev_timestep, action, next_timestep)

    self._prev_timestep = next_timestep

    if self._adder:
      self._adder.add(action, next_timestep, extras={'pi': self._probs})


class ReprMCTSActor(acme.Actor):
  """Executes a policy- and value-network guided MCTS search with representation for raw observation(MuZero setting)."""

  _prev_timestep: dm_env.TimeStep

  def __init__(
      self,
      environment_spec: specs.EnvironmentSpec,
      model: models.Model,
      repr_network: snt.Module,
      eval_network: snt.Module,
      discount: float,
      num_simulations: int,
      adder: Optional[adders.Adder] = None,
      variable_client: Optional[tf2_variable_utils.VariableClient] = None,
  ):

    # Internalize components: model, network, data sink and variable source.
    self._model = model
    self._repr_network = tf.function(repr_network)
    self._eval_network = tf.function(eval_network)
    self._variable_client = variable_client
    self._adder = adder

    # Internalize hyperparameters.
    self._num_actions = environment_spec.actions.num_values
    self._num_simulations = num_simulations
    self._actions = list(range(self._num_actions))
    self._discount = discount

    # We need to save the policy so as to add it to replay on the next step.
    self._probs = np.ones(
        shape=(self._num_actions,), dtype=np.float32) / self._num_actions

  def _forward(
      self, hidden_state: types.Observation) -> Tuple[types.Probs, types.Value]:
    """Performs a forward pass of the policy-value network."""
    logits, value = self._eval_network(tf.expand_dims(hidden_state, axis=0))

    # Convert to numpy & take softmax.
    logits = logits.numpy().squeeze(axis=0)
    value = value.numpy().item()
    probs = special.softmax(logits)

    return probs, value

  def select_action(self, observation: types.Observation) -> types.Action:
    """Computes the agent's policy via MCTS."""
    if self._model.needs_reset:
      self._model.reset(observation)

    hidden_state = tf.squeeze(self._repr_network(tf.expand_dims(observation, axis=0)), axis=0)
    
    # Compute a fresh MCTS plan.
    root = search.mcts(
        hidden_state,
        model=self._model,
        search_policy=search.puct,
        evaluation=self._forward,
        num_simulations=self._num_simulations,
        num_actions=self._num_actions,
        discount=self._discount,
    )

    # The agent's policy is softmax w.r.t. the *visit counts* as in AlphaZero.
    probs = search.visit_count_policy(root)
    action = np.int32(np.random.choice(self._actions, p=probs))

    # Save the policy probs so that we can add them to replay in `observe()`.
    self._probs = probs.astype(np.float32)

    return action

  def update(self, wait: bool = False):
    """Fetches the latest variables from the variable source, if needed."""
    if self._variable_client:
      self._variable_client.update(wait)

  def observe_first(self, timestep: dm_env.TimeStep):
    self._prev_timestep = timestep
    if self._adder:
      self._adder.add_first(timestep)

  def observe(self, action: types.Action, next_timestep: dm_env.TimeStep):
    """Updates the agent's internal model and adds the transition to replay."""
    self._model.update(self._prev_timestep, action, next_timestep)

    self._prev_timestep = next_timestep

    if self._adder:
      self._adder.add(action, next_timestep, extras={'pi': self._probs})


class SampledMCTSActor(MCTSActor):
  """Executes a policy- and value-network guided MCTS search with sampled actions."""

  _prev_timestep: dm_env.TimeStep

  def __init__(
      self,
      environment_spec: specs.EnvironmentSpec,
      model: models.Model,
      network: snt.Module,
      discount: float,
      num_simulations: int,
      pi_shape: Tuple,
      num_samples: int,
      sampling_distribution: Callable[[types.Probs], types.Probs] = lambda x: x,
      adder: Optional[adders.Adder] = None,
      variable_client: Optional[tf2_variable_utils.VariableClient] = None,
  ):

    # Internalize components: model, network, data sink and variable source.
    self._model = model
    self._network = tf.function(network)
    self._variable_client = variable_client
    self._adder = adder

    # Internalize hyperparameters.
    self._action_spec = environment_spec.actions
    self._bins = self._bin_to_value(environment_spec, pi_shape[-1])
    self._actions = list(range(pi_shape[-1]))
    self._num_simulations = num_simulations
    self._discount = discount
    self._num_samples = num_samples
    self._sampling_distribution = sampling_distribution

    # We need to save the policy so as to add it to replay on the next step.
    self._probs = np.ones(
        shape=pi_shape, dtype=np.float32) / pi_shape[-1]  
        
  def _forward(
      self, observation: types.Observation) -> Tuple[types.Probs, types.Value]:
    """Performs a forward pass of the policy-value network."""
    logits, value = self._network(tf.expand_dims(observation, axis=0))

    # Convert to numpy & take softmax.
    logits = logits.numpy().squeeze(axis=0)
    value = value.numpy().item()
    probs = special.softmax(logits, axis=-1)

    return probs, value
  
  def _sample(self, prior: types.Probs, )-> Tuple[List[types.Action], List[types.Probs]]:
    dist = self._sampling_distribution(prior) 
    sampled_actions, priors = [], []
    for dim in range(len(prior)):
      dim_unique_actions, dim_priors = self._sample_per_dimension(prior[dim], dist[dim])
      sampled_actions.append(dim_unique_actions)
      priors.append(dim_priors)

    return sampled_actions, priors
  
  def _sample_per_dimension(
      self, dim_prior: np.ndarray, sampling_dist: np.ndarray,
  ) -> Tuple[np.ndarray, np.ndarray]:
    dim_sampled_actions = np.random.choice(len(dim_prior), size=self._num_samples, p=sampling_dist)
    dim_unique_actions, counts = np.unique(dim_sampled_actions, return_counts=True)
    empirical = counts / len(dim_sampled_actions)
    dim_priors = empirical / sampling_dist[dim_unique_actions] * dim_prior[dim_unique_actions]
    search.check_numerics(dim_priors)

    return dim_unique_actions, dim_priors
  
  def _convert(self, raw_action: List[types.Action]) -> types.Action:
    action = [self._bins[dim][a] for dim, a in enumerate(raw_action)]
    if len(action) == 1: return action[0]
    else: return np.asarray(action).reshape(self._action_spec.shape)

  def _bin_to_value(self, environment_spec: specs.EnvironmentSpec, k_bins: int,):
    _act_spec = environment_spec.actions
    if isinstance(_act_spec, specs.DiscreteArray):
      values = [list(range(_act_spec.num_values))]
    elif isinstance(_act_spec, specs.BoundedArray):
      num_dimensions = np.prod(_act_spec.shape)
      
      _min = _act_spec.minimum * np.ones(num_dimensions) if _act_spec.minimum.size == 1 else _act_spec.minimum.reshape(num_dimensions)
      _max = _act_spec.maximum * np.ones(num_dimensions) if _act_spec.maximum.size == 1 else _act_spec.maximum.reshape(num_dimensions)
      
      values = []
      for dim in range(num_dimensions):
        bin_edges = np.linspace(_min[dim], _max[dim], k_bins + 1)
        bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2
        values.append(bin_midpoints)

    return values 

  def select_action(self, observation: types.Observation) -> types.Action:
    """Computes the agent's policy via MCTS."""
    if self._model.needs_reset:
      self._model.reset(observation)

    # Compute a fresh MCTS plan.
    root = search.sampled_mcts(
        observation,
        model=self._model,
        search_policy=search.factored_puct,
        sample_policy=self._sample,
        evaluation=self._forward,
        num_simulations=self._num_simulations,
        discount=self._discount,
        action_converter=self._convert,
    )

    # The agent's policy is softmax w.r.t. the *visit counts* as in AlphaZero.
    dim_probs = search.factored_visit_count_policy(root)
    probs = np.zeros_like(self._probs)
    for dim, dim_prob in enumerate(dim_probs):
      probs[dim, list(root.sampled_action[dim].keys())] = dim_prob

    # Save the policy probs so that we can add them to replay in `observe()`.
    self._probs = probs.astype(np.float32)

    # Sample raw action and convert to action
    raw_action = [np.int32(np.random.choice(self._actions, p=p)) for p in probs]
    action = self._convert(raw_action)

    return action

