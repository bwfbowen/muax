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

"""A MCTS "AlphaZero-style" learner."""

from typing import List, Optional

import acme
from acme.tf import utils as tf2_utils
from acme.tf import savers as tf2_savers
from acme.utils import counting
from acme.utils import loggers
import numpy as np
import sonnet as snt
import tensorflow as tf


class AZLearner(acme.Learner):
  """AlphaZero-style learning."""

  def __init__(
      self,
      network: snt.Module,
      optimizer: snt.Optimizer,
      dataset: tf.data.Dataset,
      discount: float,
      logger: Optional[loggers.Logger] = None,
      counter: Optional[counting.Counter] = None,
  ):

    # Logger and counter for tracking statistics / writing out to terminal.
    self._counter = counting.Counter(counter, 'learner')
    self._logger = logger or loggers.TerminalLogger('learner', time_delta=30.)

    # Internalize components.
    # TODO(b/155086959): Fix type stubs and remove.
    self._iterator = iter(dataset)  # pytype: disable=wrong-arg-types
    self._optimizer = optimizer
    self._network = network
    self._variables = network.trainable_variables
    self._discount = np.float32(discount)

  @tf.function
  def _step(self) -> tf.Tensor:
    """Do a step of SGD on the loss."""

    inputs = next(self._iterator)
    o_t, _, r_t, d_t, o_tp1, extras = inputs.data
    pi_t = extras['pi']

    with tf.GradientTape() as tape:
      # Forward the network on the two states in the transition.
      logits, value = self._network(o_t)
      _, target_value = self._network(o_tp1)
      target_value = tf.stop_gradient(target_value)

      # Value loss is simply on-policy TD learning.
      value_loss = tf.square(r_t + self._discount * d_t * target_value - value)

      # Policy loss distills MCTS policy into the policy network.
      policy_loss = tf.nn.softmax_cross_entropy_with_logits(
          logits=logits, labels=pi_t)

      # Compute gradients.
      loss = tf.reduce_mean(value_loss + policy_loss)
      gradients = tape.gradient(loss, self._network.trainable_variables)
      
    self._optimizer.apply(gradients, self._network.trainable_variables)

    return loss

  def step(self):
    """Does a step of SGD and logs the results."""
    loss = self._step()
    self._logger.write({'loss': loss})

  def get_variables(self, names: List[str]) -> List[List[np.ndarray]]:
    """Exposes the variables for actors to update from."""
    return tf2_utils.to_numpy(self._repr_variables + self._eval_variables)


class MZLearner(acme.Learner):
  """MuZero-style learning."""

  def __init__(
      self,
      repr_network: snt.Module,
      eval_network: snt.Module,
      optimizer: snt.Optimizer,
      dataset: tf.data.Dataset,
      discount: float,
      logger: Optional[loggers.Logger] = None,
      counter: Optional[counting.Counter] = None,
      checkpoint: bool = True,
      save_directory: str = '~/acme',
  ):

    # Logger and counter for tracking statistics / writing out to terminal.
    self._counter = counting.Counter(counter, 'learner')
    self._logger = logger or loggers.TerminalLogger('learner', time_delta=30.)

    # Internalize components.
    # TODO(b/155086959): Fix type stubs and remove.
    self._iterator = iter(dataset)  # pytype: disable=wrong-arg-types
    self._optimizer = optimizer
    self._repr_network = repr_network
    self._eval_network = eval_network
    self._repr_variables = repr_network.trainable_variables
    self._eval_variables = eval_network.trainable_variables
    self._discount = np.float32(discount)

    self._checkpointer = None 
    self._snapshotter = None 
    if checkpoint:
      self._checkpointer = tf2_savers.Checkpointer(
          directory=save_directory,
          subdirectory='muzero_learner',
          objects_to_save={
            'repr_network': self._repr_network,
            'eval_network': self._eval_network,
            'optimizer': self._optimizer,
          }
      )
      self._snapshotter = tf2_savers.Snapshotter(
          directory=save_directory,
          objects_to_save={'repr_network': repr_network, 'eval_network': eval_network}
      )

  @tf.function
  def _step(self) -> tf.Tensor:
    """Do a step of SGD on the loss."""

    inputs = next(self._iterator)
    o_t, _, r_t, d_t, o_tp1, extras = inputs.data
    pi_t = extras['pi']

    with tf.GradientTape() as tape:
      # representation
      h_t = self._repr_network(o_t)
      h_tp1 = self._repr_network(o_tp1)
      # Forward the network on the two states in the transition.
      logits, value = self._eval_network(h_t)
      _, target_value = self._eval_network(h_tp1)
      target_value = tf.stop_gradient(target_value)

      # Value loss is simply on-policy TD learning.
      value_loss = tf.square(r_t + self._discount * d_t * target_value - value)

      # Policy loss distills MCTS policy into the policy network.
      policy_loss = tf.nn.softmax_cross_entropy_with_logits(
          logits=logits, labels=pi_t)

      # Compute gradients.
      loss = tf.reduce_mean(value_loss + policy_loss)
      gradients = tape.gradient(loss, self._eval_network.trainable_variables + self._repr_network.trainable_variables)
      
    self._optimizer.apply(gradients, self._eval_network.trainable_variables + self._repr_network.trainable_variables)

    return loss

  def step(self):
    """Does a step of SGD and logs the results."""
    loss = self._step()
    self._logger.write({'loss': loss})
    if self._checkpointer: self._checkpointer.save()
    if self._snapshotter: self._snapshotter.save()

  def get_variables(self, names: List[str]) -> List[List[np.ndarray]]:
    """Exposes the variables for actors to update from."""
    return tf2_utils.to_numpy(self._repr_variables + self._eval_variables)
  