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
from typing import Any, Tuple
import dataclasses
from dataclasses import dataclass
from collections import deque 
from itertools import islice
import jax
from jax import numpy as jnp

from .utils import sliceable_deque


@dataclass
class Transition:
    obs: Any = 0.
    a: int = 0
    r: float = 0.
    done: bool = False
    Rn: float = 0. 
    v: float = 0. 
    pi: Any = 0.
    w: float = 1.

    def __iter__(self):
      for field in dataclasses.fields(self):
        yield getattr(self, field.name)

    def __getitem__(self, index):
      return Transition(*(_attr[index] for _attr in self))

def flatten_transition_func(transition: Transition) -> Tuple:
  return iter(transition), None 

def unflatten_transition_func(treedef, leaves) -> Transition:
  return Transition(*leaves)

jax.tree_util.register_pytree_node(
    Transition,
    flatten_func=flatten_transition_func,
    unflatten_func=unflatten_transition_func
)  
    

class BaseTracer(ABC):

    @abstractmethod
    def reset(self):
        r"""
        Reset the cache to the initial state.
        """
        pass

    @abstractmethod
    def add(self, obs, a, r, done, v=0.0, pi=0.0, w=1.0):
        r"""
        Add a transition to the experience cache.
        Parameters
        ----------
        obs : state observation
            A single state observation.
        a : action
            A single action.
        r : float
            A single observed reward.
        done : bool
            Whether the episode has finished.
        v : search tree root node value.
        pi : float, optional
            The action weights.
        w : float, optional
            Sample weight associated with the given state-action pair.
        """
        pass

    @abstractmethod
    def pop(self):
        r"""
        Pop a single transition from the cache.
        Returns
        -------
        transition : 
            
        """
        pass


class NStep(BaseTracer):
    r"""
    A short-term cache for :math:`n`-step bootstrapping.

    Parameters
    ----------
    n : positive int

        The number of steps over which to bootstrap.

    gamma : float between 0 and 1

        The amount by which to discount future rewards.

    """

    def __init__(self, n, gamma):
        self.n = int(n)
        self.gamma = float(gamma)
        self.reset()

    def reset(self):
        self._deque_s = sliceable_deque([])
        self._deque_r = sliceable_deque([])
        self._done = False
        self._gammas = jnp.power(self.gamma, jnp.arange(self.n))
        self._gamman = jnp.power(self.gamma, self.n)

    def add(self, obs, a, r, done, v=0.0, pi=0.0, w=1.0):
        # if self._done and len(self):
        #     raise EpisodeDoneError(
        #         "please flush cache (or repeatedly call popleft) before appending new transitions")

        self._deque_s.append((obs, a, v, pi, w))
        self._deque_r.append(r)
        self._done = bool(done)

    def __len__(self):
        return len(self._deque_s)

    def __bool__(self):
        return bool(len(self)) and (self._done or len(self) > self.n)

    def pop(self):
        # if not self:
        #     raise InsufficientCacheError(
        #         "cache needs to receive more transitions before it can be popped from")

        # pop state-action (propensities) pair
        obs, a, v, pi, w = self._deque_s.popleft()

        # n-step partial return
        rs = jnp.array(self._deque_r[:self.n])
        Rn = jnp.sum(self._gammas[:len(rs)] * rs).item()
        r = self._deque_r.popleft()

        # keep in mind that we've already popped 
        if len(self) >= self.n:
            obs_next, a_next, v_next, pi_next, _ = self._deque_s[self.n - 1]
            done = False
            gamman = self._gamman
        else:
            # no more bootstrapping
            obs_next, a_next, v_next, pi_next, done = obs, a, v, pi, True
            gamman = self._gammas[len(rs)]
        
        Rn += v_next * gamman

        return Transition(obs=obs, a=a, r=r, done=done, Rn=Rn, v=v, pi=pi, w=w)


class PNStep(NStep):
    r"""
    A short-term cache for :math:`n`-step bootstrapping with priority.

    Parameters
    ----------
    n : positive int

        The number of steps over which to bootstrap.

    gamma : float between 0 and 1

        The amount by which to discount future rewards.

    alpha: float between 0 and 1
        The PER alpha.
    """
    def __init__(self, n, gamma, alpha: float = 0.5):
      self.alpha = float(alpha)
      super().__init__(n, gamma)
    
    def pop(self):
        # if not self:
        #     raise InsufficientCacheError(
        #         "cache needs to receive more transitions before it can be popped from")

        # pop state-action (propensities) pair
        obs, a, v, pi, w = self._deque_s.popleft()

        # n-step partial return
        rs = jnp.array(self._deque_r[:self.n])
        Rn = jnp.sum(self._gammas[:len(rs)] * rs).item()
        r = self._deque_r.popleft()

        # keep in mind that we've already popped 
        if len(self) >= self.n:
            obs_next, a_next, v_next, pi_next, _ = self._deque_s[self.n - 1]
            done = False
            gamman = self._gamman
        else:
            # no more bootstrapping
            obs_next, a_next, v_next, pi_next, done = obs, a, v, pi, True
            gamman = self._gammas[len(rs)]
        
        Rn += v_next * gamman

        w = abs(v - Rn) ** self.alpha

        return Transition(obs=obs, a=a, r=r, done=done, Rn=Rn, v=v, pi=pi, w=w)