from typing import List, Dict, Any, Generator, Optional, Union
import copy 
import random 
import warnings
import numpy as np
import jax
from gymnasium import spaces

from stable_baselines3.common.buffers import BaseBuffer, RolloutBuffer, DictRolloutBuffer, ReplayBuffer, DictReplayBuffer
from stable_baselines3.common.vec_env import VecNormalize

from .type_aliases import MuaxRolloutBufferSamples

try:
    # Check memory used by replay buffer when possible
    import psutil
except ImportError:
    psutil = None


# class MuaxReplayBuffer(ReplayBuffer):
#     def __init__(
#         self,
#         buffer_size: int,
#         observation_space: spaces.Space,
#         action_space: spaces.Space,
#         device: jax.Device = None,
#         n_envs: int = 1,
#         optimize_memory_usage: bool = False,
#         handle_timeout_termination: bool = True,
#     ):
#         super().__init__(buffer_size=buffer_size, 
#                          observation_space=observation_space, 
#                          action_space=action_space, 
#                          n_envs=n_envs, 
#                          optimize_memory_usage=optimize_memory_usage,
#                          handle_timeout_termination=handle_timeout_termination)
#         self.next_observations = None  # In Muax we do not need next_obs information
#         self.Rn = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
#         self.values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
#         self.pi = np.zeros((self.buffer_size, self.n_envs, *action_space.shape), dtype=np.float32)
#         self.weights = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

#         if psutil is not None:
#             total_memory_usage = self.observations.nbytes + self.actions.nbytes + self.rewards.nbytes + self.dones.nbytes + self.Rn.nbytes + self.values.nbytes + self.pi.nbytes + self.weights.nbytes

#             if total_memory_usage > mem_available:
#                 # Convert to GB
#                 total_memory_usage /= 1e9
#                 mem_available /= 1e9
#                 warnings.warn(
#                     "This system does not have apparently enough memory to store the complete "
#                     f"muax replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB"
#                 )
    
#     def add(
#         self, 
#         obs: np.ndarray,
#         next_obs: np.ndarray,
#         action: np.ndarray,
#         reward: np.ndarray,
#         done: np.ndarray,
#         infos: List[Dict[str, Any]]
#     ) -> None:
#         # Reshape needed when using multiple envs with discrete observations
#         # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
#         if isinstance(self.observation_space, spaces.Discrete):
#             obs = obs.reshape((self.n_envs, *self.obs_shape))
#             next_obs = next_obs.reshape((self.n_envs, *self.obs_shape))

#         # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
#         action = action.reshape((self.n_envs, self.action_dim))

#         # Copy to avoid modification by reference
#         self.observations[self.pos] = np.array(obs).copy()

#         if self.optimize_memory_usage:
#             self.observations[(self.pos + 1) % self.buffer_size] = np.array(next_obs).copy()
#         else:
#             self.next_observations[self.pos] = np.array(next_obs).copy()

#         self.actions[self.pos] = np.array(action).copy()
#         self.rewards[self.pos] = np.array(reward).copy()
#         self.dones[self.pos] = np.array(done).copy()

#         if self.handle_timeout_termination:
#             self.timeouts[self.pos] = np.array([info.get("TimeLimit.truncated", False) for info in infos])

#         self.pos += 1
#         if self.pos == self.buffer_size:
#             self.full = True
#             self.pos = 0


class MuaxRolloutBuffer(BaseBuffer):

    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    Rn: np.ndarray
    values: np.ndarray
    pi: np.ndarray
    weights: np.ndarray
    episode_starts: np.ndarray

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: jax.Device = None,
        n_envs: int = 1,
        k_steps: int = 5,
        n_step_bootstrapping: int = 10,
        lambda_t: float = 1.,
        gamma_t: float = 0.99,
        prioritized_sampling: bool = False,
        prioritized_alpha: float = 1.,
        prioritized_beta: float = 1.
    ):
        super().__init__(buffer_size=buffer_size, observation_space=observation_space, action_space=action_space, n_envs=n_envs)
        self.device = device 
        self.k_steps = k_steps
        self.n_step_bootstrapping = n_step_bootstrapping
        self.lambda_t = lambda_t
        self.gamma_t = gamma_t
        self.prioritized_sampling = prioritized_sampling
        self.prioritized_alpha = prioritized_alpha
        self.prioritized_beta = prioritized_beta
        self.reset()
    
    @staticmethod
    def flatten(arr: np.ndarray) -> np.ndarray:
        """
        Flatten axes 0 (buffer_size) and 1 (n_envs)
        """
        shape = arr.shape 
        if len(shape) < 3:
            shape = (*shape, 1)
        return arr.reshape(-1, *shape[2:])        
    
    def reset(self) -> None:
        self.observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.Rn = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.pi = np.zeros((self.buffer_size, self.n_envs, *self.action_space.shape), dtype=np.float32)
        self.weights = np.ones((self.buffer_size, self.n_envs), dtype=np.float32)
        self.episode_starts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.generator_ready = False
        super().reset()
    
    def compute_Rn_and_weights(
        self, 
        last_values: np.ndarray, 
        dones: np.ndarray,
        n: Optional[int] = None,
        lambda_t: Optional[float] = None,
        gamma_t: Optional[float] = None 
    ) -> None:
        n = self.n_step_bootstrapping if n is None else n 
        lambda_t = self.lambda_t if lambda_t is None else lambda_t
        gamma_t = self.gamma_t if gamma_t is None else gamma_t

        next_non_terminal = 1.0 - dones 
        next_values = last_values

        r = np.concatenate([self.rewards, np.zeros((n, self.n_envs))])
        v = np.concatenate([self.values, np.ones((n, self.n_envs)) * last_values])
        ep_starts = np.concatenate([self.episode_starts, np.ones((n, self.n_envs)) * dones])
        
        for step in reversed(range(self.buffer_size)):
            next_non_terminal = 1.0 - ep_starts[step + n]
            next_values = v[step + n]
            G = next_non_terminal * next_values
            for t in reversed(range(step, step + n)):
                non_terminal = 1.0 - ep_starts[t + 1]
                G = r[t] + gamma_t * non_terminal * (lambda_t * G + (1 - lambda_t) * v[t + 1]) 
            self.Rn[step] = G 
        self.weights = np.abs(self.values - self.Rn) ** self.prioritized_alpha
            
    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        value: np.ndarray,
        pi: np.ndarray,
        episode_start: np.ndarray
    ) -> None:
        
        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))
        
        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        self.observations[self.pos] = np.array(obs).copy()
        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.episode_starts[self.pos] = np.array(episode_start).copy()
        self.values[self.pos] = np.array(value).copy()
        self.pi[self.pos] = np.array(pi).copy()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
    
    def get(
        self, 
        batch_size: Optional[int] = None, 
        k_steps: Optional[int] = None
    ) -> Generator[MuaxRolloutBufferSamples, None, None]:
        assert self.full, ""

        # feasible indices
        k_steps = self.k_steps if k_steps is None else k_steps
        flat_starts = self.episode_starts.ravel()
        flat_starts_indices = np.nonzero(flat_starts)[0]
        steps_indices = np.arange(len(flat_starts))
        diff = flat_starts_indices.reshape(-1, 1) - steps_indices
        sample_mask = ~np.any((diff > 0) & (diff < k_steps), axis=0)
        sample_mask[-k_steps+1:] = False 
        indices = np.nonzero(sample_mask)[0]
        sample_indices = indices[:, np.newaxis] + np.arange(k_steps)

        if not self.generator_ready:
            _array_names = [
                'observations',
                'actions',
                'rewards',
                'Rn',
                'pi',
                'weights'
            ]

            for array in _array_names:
                self.__dict__[array] = self.flatten(self.__dict__[array])
            self.generator_ready = True 
        
        if batch_size is None:
            batch_size = len(indices)
        
        if not self.prioritized_sampling:
            np.random.shuffle(sample_indices)
            start_idx = 0
            while start_idx < len(indices):
                yield self._get_samples(sample_indices[start_idx: start_idx + batch_size])
                start_idx += batch_size
        else:
            # prioritized sampling weights
            flat_weights = copy.deepcopy(self.weights.ravel())
            flat_weights = flat_weights[indices]
            start_idx = 0 
            while start_idx < len(indices):
                _sampled_indices = random.choices(sample_indices, weights=flat_weights, k=batch_size)
                yield self._get_samples(_sampled_indices)
                start_idx += batch_size

    def _get_samples(
        self, 
        batch_inds: np.ndarray, 
        env: Optional[VecNormalize] = None
    ) -> MuaxRolloutBufferSamples:
        
        # importance sampling ratio
        _weights = self.weights[batch_inds]
        weights = (
            (1 / len(_weights))  # 1 / N
            * (np.sum(_weights, axis=0) / _weights)  # 1/P
            ) ** self.prioritized_beta
        
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.rewards[batch_inds],
            self.Rn[batch_inds],
            self.pi[batch_inds],
            weights
        )
        return MuaxRolloutBufferSamples(*data)