from abc import ABC, abstractmethod

from collections import deque 
from itertools import islice
from jax import numpy as jnp

class BaseTracer(ABC):

    @abstractmethod
    def reset(self):
        r"""
        Reset the cache to the initial state.
        """
        pass

    @abstractmethod
    def add(self, s, a, r, done, logp=0.0, w=1.0):
        r"""
        Add a transition to the experience cache.
        Parameters
        ----------
        s : state observation
            A single state observation.
        a : action
            A single action.
        r : float
            A single observed reward.
        done : bool
            Whether the episode has finished.
        logp : float, optional
            The log-propensity :math:`\log\pi(a|s)`.
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

    record_extra_info : bool, optional
    """

    def __init__(self, n, gamma, record_extra_info=False):
        self.n = int(n)
        self.gamma = float(gamma)
        self.record_extra_info = record_extra_info
        self.reset()

    def reset(self):
        self._deque_s = deque([])
        self._deque_r = deque([])
        self._done = False
        self._gammas = jnp.power(self.gamma, jnp.arange(self.n))
        self._gamman = jnp.power(self.gamma, self.n)

    def add(self, s, a, r, done, logp=0.0, w=1.0):
        # if self._done and len(self):
        #     raise EpisodeDoneError(
        #         "please flush cache (or repeatedly call popleft) before appending new transitions")

        self._deque_s.append((s, a, logp, w))
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
        s, a, logp, w = self._deque_s.popleft()

        # n-step partial return
        zipped = zip(self._gammas, self._deque_r)
        rn = sum(x * r for x, r in islice(zipped, self.n))
        r = self._deque_r.popleft()

        # keep in mind that we've already popped (s, a, logp)
        if len(self) >= self.n:
            s_next, a_next, logp_next, _ = self._deque_s[self.n - 1]
            done = False
        else:
            # no more bootstrapping
            s_next, a_next, logp_next, done = s, a, logp, True

        extra_info = self._extra_info(
            s, a, r, len(self) == 0, logp, w) if self.record_extra_info else None

        return TransitionBatch.from_single(
            s=s, a=a, logp=logp, r=rn, done=done, gamma=self._gamman,
            s_next=s_next, a_next=a_next, logp_next=logp_next, w=w, extra_info=extra_info)

    def _extra_info(self, s, a, r, done, logp, w):
        return