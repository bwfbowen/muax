"""
    MIT License

    Copyright (c) 2020 Microsoft Corporation.
    Copyright (c) 2021 github.com/coax-dev
    Copyright (c) 2022 bf2504@columbia.edu

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE
"""    

from abc import ABC, abstractmethod
from collections import deque 
from itertools import chain
import random
import numpy as np 
import jax 
from jax import numpy as jnp

from .utils import sliceable_deque
from .episode_tracer import Transition

   
class Trajectory:
    r"""
    A simple trajectory to hold episodical transitions.
    """
    def __init__(self):
      self.trajectory = sliceable_deque([])
      self._transition_weight = sliceable_deque([])
      self._batched_transitions = None
    
    def add(self, transition):
      """Adds a transition to the trajectory
      
      Parameters
      ----------
      transition: An instance of muax.episode_tracer.Transition. 
      
      """
      self.trajectory.append(transition)
      self._transition_weight.append(transition.w)

    
    def finalize(self):
      """"""
      batched_transitions = jax.tree_util.tree_transpose(
          outer_treedef=jax.tree_util.tree_structure([0 for i in self.trajectory]),
          inner_treedef=jax.tree_util.tree_structure(Transition()),
          pytree_to_transpose=self.trajectory
          )
      batched_transitions = Transition(*(np.expand_dims(np.vstack(_attr), axis=0)
          for _attr in batched_transitions))
      self._batched_transitions = batched_transitions

    
    def sample(self, num_samples: int = 1, k_steps: int = 5):
      """Samples consecutive transitions of k steps from the trajectory.
      
      Parameters
      ----------
      num_samples: int, positive int. Number of samples to draw from the trajectory. 
          Each sample is chosen given the weight(transition.w) as probability.
      k_steps: int, positive int. Determines the length of consecutive steps in each sample.
          If k_steps >= len(trajectory), no samples would be collected.
      
      Returns
      ----------
      samples: List[Transition]. For each transition in the list, 
          the attributes of which have the dimension `[B, L, ...]`, where B is the batch size(1) and 
          L is the length(k) of the consecutive transitions.
      """
      if len(self) <= k_steps: return []
      max_idx = len(self) - k_steps
      idxes = random.choices(range(max_idx), 
                            weights=self._transition_weight[:max_idx], 
                            k=num_samples)
                            
      if self.batched_transitions is None:
        self.finalize()

      samples = [self._get_sample(idx, k_steps) for idx in idxes]
      
      return samples

    @property
    def batched_transitions(self):
      return self._batched_transitions

    def _get_sample(self, idx, k_steps):
      end_idx = idx + k_steps 
      sample = self.batched_transitions[:, idx: end_idx, :]
      return sample

    def __getitem__(self, index):
      return self.trajectory[index]

    def __len__(self):
      return len(self.trajectory)

    def __repr__(self):
      return f'{type(self)}(len={len(self)})'
      

class BaseReplayBuffer(ABC):

    @property
    @abstractmethod
    def capacity(self):
        pass

    @abstractmethod
    def add(self, transition_batch):
        pass

    @abstractmethod
    def sample(self, batch_size=32):
        pass

    @abstractmethod
    def clear(self):
        pass

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __bool__(self):
        pass

    @abstractmethod
    def __iter__(self):
        pass


class TrajectoryReplayBuffer(BaseReplayBuffer):
    r"""
    A simple ring buffer for experience replay.
    Parameters
    ----------
    capacity : positive int
        The capacity of the experience replay buffer.
    random_seed : int, optional
        To get reproducible results.
    """
    def __init__(self, capacity, random_seed=None):
        self._capacity = int(capacity)
        random.seed(random_seed)
        self._random_state = random.getstate()
        self.clear()  # sets self._storage

    @property
    def capacity(self):
        return self._capacity

    def add(self, trajectory, w=1.):
        r"""
        Add a trajectory to the experience replay buffer.
        Parameters
        ----------
        trajectory : Trajectory
            A :class: `Trajectory` object.
        w: float
            sample probability weight of input trajectory
        """
        self._storage.append(trajectory)
        self._trajectory_weight.append(w)

    def sample(self, 
               batch_size=32, 
               num_trajectory: int = None,
               k_steps: int = 5,
               sample_per_trajectory: int = 1):
        r"""
        Get a batch of transitions to be used for bootstrapped updates.
        Parameters
        ----------
        batch_size : positive int, optional
            The desired batch size of the sample. One sample from a single trajectory.
        num_trajectory: positive int, optional
            Number of trajectory to be sampled. Either num_trajectory or batch_size 
            need to be given.
        k_steps: positive int
            Consecutive k steps from each trajectory. k steps unrolled for training. 
        sample_per_trajectory: positive int, optional
            Number of Transition to be sampled. Used when num_trajectory is provided.
            The batch size will be num_trajectory * sample_per_trajectory.
        Returns
        -------
        transitions : Batch of consecutive transitions.
        """
        if batch_size is None and num_trajectory is None: 
          raise ValueError('Either num_trajectory or batch_size need to be given.')
        elif batch_size is not None and num_trajectory is None:
          num_trajectory = batch_size 
          sample_per_trajectory = 1
        # sandwich sample in between setstate/getstate in case global random state was tampered with
        random.setstate(self._random_state)
        trajectories = random.choices(self._storage, 
                                      weights=self._trajectory_weight,
                                      k=num_trajectory)
        self._random_state = random.getstate()
        batch = list(chain.from_iterable(traj.sample(num_samples=sample_per_trajectory, k_steps=k_steps) 
                     for traj in trajectories
                     ))
        
        batch = jax.tree_util.tree_transpose(
          outer_treedef=jax.tree_util.tree_structure([0 for i in batch]),
          inner_treedef=jax.tree_util.tree_structure(Transition()),
          pytree_to_transpose=batch)
        batch = Transition(*(np.vstack(v) for v in batch))
        return batch

    def clear(self):
        r""" Clear the experience replay buffer. """
        self._storage = sliceable_deque([], maxlen=self.capacity)
        self._trajectory_weight = sliceable_deque([], maxlen=self.capacity)

    def __len__(self):
        return len(self._storage)

    def __bool__(self):
        return bool(len(self))

    def __iter__(self):
        return iter(self._storage)
