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

"""A simulator model, which wraps a copy of the true environment."""
from typing import Optional
import copy
import dataclasses
import numpy as np
from acme import types
from acme.agents.tf.mcts import types as mtypes
from acme.agents.tf.mcts.models import base
import dm_env
from open_spiel.python import rl_environment


@dataclasses.dataclass
class OpenSpielCheckpoint:
  """Holds the checkpoint state for the open_spiel environment simulator."""
  needs_reset: bool 
  environment: rl_environment.Environment
  current_player: Optional[int] = None


@dataclasses.dataclass
class Checkpoint:
  """Holds the checkpoint state for the environment simulator."""
  needs_reset: bool
  environment: dm_env.Environment


class OpenSpielSimulator(base.Model):
  """A simulator model for OpenSpiel wrapper environments.

  Assumptions:
    - The environment (including RNG) is fully copyable via `deepcopy`.
    - Environment dynamics (modulo episode resets) are deterministic.
  """

  _checkpoint: OpenSpielCheckpoint
  _env: rl_environment.Environment

  def __init__(self, env: rl_environment.Environment):
    self._env = copy.deepcopy(env)
    self._needs_reset = True
    self._current_player = None
    self.save_checkpoint()

  def update(
      self,
      timestep: dm_env.TimeStep,
      action: mtypes.Action,
      next_timestep: dm_env.TimeStep,
  ) -> dm_env.TimeStep:
    # Call update() once per 'real' experience to keep this env in sync.
    return self.step([action])
  
  def save_checkpoint(self):
    self._checkpoint = OpenSpielCheckpoint(
      needs_reset=self._needs_reset,
      environment=copy.deepcopy(self._env),
      current_player=self._current_player,
    )
  
  def load_checkpoint(self):
    """Restores the simulator state from the saved checkpoint."""
    self._env = copy.deepcopy(self._checkpoint.environment)
    self._needs_reset = self._checkpoint.needs_reset
    self._current_player = self._checkpoint.current_player

  def step(self, action: types.NestedArray) -> dm_env.TimeStep:
    """Takes a step in the simulator.
    
    Args:
      action: Action to take.
    
    Returns:
      dm_env.TimeStep with observation from current player's perspective.
    
    Raises:
      ValueError: If simulator needs reset.
    """
    if self._needs_reset:
      raise ValueError('This model needs to be explicitly reset.')
    
    # Take step in environment 
    timestep = self._env.step(action)

    # Store current player after step
    self._current_player = self._env.current_player
    
    self._needs_reset = timestep.last()

    return dm_env.TimeStep(
        step_type=timestep.step_type,
        reward=timestep.reward[self._current_player],
        discount=timestep.discount[self._current_player],
        observation=timestep.observation[self._current_player]
      )
  
  def reset(self, *unused_args, **unused_kwargs):
    self._needs_reset = False
    timestep = self._env.reset()
    self._current_player = self._env.current_player
    return dm_env.TimeStep(
      step_type=timestep.step_type,
      reward=None,
      discount=None,
      observation=timestep.observation[self._current_player],
    )

  def observation_spec(self):
    return self._env.observation_spec()

  def action_spec(self):
    return self._env.action_spec()

  @property
  def needs_reset(self) -> bool:
    return self._needs_reset
  
  @property
  def current_player(self) -> int:
    return self._current_player
  

class Simulator(base.Model):
  """A simulator model, which wraps a copy of the true environment.

  Assumptions:
    - The environment (including RNG) is fully copyable via `deepcopy`.
    - Environment dynamics (modulo episode resets) are deterministic.
  """

  _checkpoint: Checkpoint
  _env: dm_env.Environment

  def __init__(self, env: dm_env.Environment):
    # Make a 'checkpoint' copy env to save/load from when doing rollouts.
    self._env = copy.deepcopy(env)
    self._needs_reset = True
    self.save_checkpoint()

  def update(
      self,
      timestep: dm_env.TimeStep,
      action: mtypes.Action,
      next_timestep: dm_env.TimeStep,
  ) -> dm_env.TimeStep:
    # Call update() once per 'real' experience to keep this env in sync.
    return self.step(action)

  def save_checkpoint(self):
    self._checkpoint = Checkpoint(
        needs_reset=self._needs_reset,
        environment=copy.deepcopy(self._env),
    )

  def load_checkpoint(self):
    self._env = copy.deepcopy(self._checkpoint.environment)
    self._needs_reset = self._checkpoint.needs_reset

  def step(self, action: mtypes.Action) -> dm_env.TimeStep:
    if self._needs_reset:
      raise ValueError('This model needs to be explicitly reset.')
    timestep = self._env.step(action)
    self._needs_reset = timestep.last()
    return timestep

  def reset(self, *unused_args, **unused_kwargs):
    self._needs_reset = False
    return self._env.reset()

  def observation_spec(self):
    return self._env.observation_spec()

  def action_spec(self):
    return self._env.action_spec()

  @property
  def needs_reset(self) -> bool:
    return self._needs_reset
