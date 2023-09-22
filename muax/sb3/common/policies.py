from typing import Dict, Tuple, Union, no_type_check, Optional, Callable, Any, List
import copy 
import numpy as np
import optax
import haiku as hk
from gymnasium import spaces
import gymnasium as gym 
from stable_baselines3.common.policies import BasePolicy, BaseModel
from stable_baselines3.common.preprocessing import is_image_space, maybe_transpose
from stable_baselines3.common.utils import is_vectorized_observation
from stable_baselines3.common.type_aliases import Schedule

from .utils import obs_as_tensor



class MuaxPolicy(BasePolicy):
    def __init__(
        self, 
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        features_extractor_class=None,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        optimizer_class: Callable[..., optax.GradientTransformation] = optax.adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        squash_output: bool = True,
        ):

        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=squash_output
        )

    @no_type_check
    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        self.set_training_mode(False)

        observation, vectorized_env = self.prepare_obs(observation)

        actions = self._predict(observation, deterministic=deterministic)

        # Convert to numpy, and reshape to the original action shape
        actions = np.array(actions).reshape((-1, *self.action_space.shape))

        if isinstance(self.action_space, spaces.Box):
            if self.squash_output:
                # Clip due to numerical instability
                actions = np.clip(actions, -1, 1)
                # Rescale to proper domain when using squashing
                actions = self.unscale_action(actions)
            else:
                # Actions could be on arbitrary scale, so clip the actions to avoid
                # out of bound error (e.g. if sampling from a Gaussian distribution)
                actions = np.clip(actions, self.action_space.low, self.action_space.high)

        # Remove batch dimension if needed
        if not vectorized_env:
            actions = actions.squeeze(axis=0)  # type: ignore[call-overload]

        return actions, state

    def prepare_obs(self, observation: Union[np.ndarray, Dict[str, np.ndarray]]) -> Tuple[np.ndarray, bool]:
        vectorized_env = False
        is_dict_obs = isinstance(observation, dict)
        if is_dict_obs:
            # need to copy the dict as the dict in VecFrameStack will become a torch tensor
            observation = copy.deepcopy(observation)
            for key, obs in observation.items():
                obs_space = self.observation_space.spaces[key]
                if is_image_space(obs_space):
                    obs_ = maybe_transpose(obs, obs_space)
                else:
                    obs_ = np.array(obs)
                vectorized_env = vectorized_env or is_vectorized_observation(obs_, obs_space)
                # Add batch dimension if needed
                observation[key] = obs_.reshape((-1, *self.observation_space[key].shape))

        elif is_image_space(self.observation_space):
            # Handle the different cases for images
            # as PyTorch use channel first format
            observation = maybe_transpose(observation, self.observation_space)

        else:
            observation = np.array(observation)

        if not isinstance(self.observation_space, spaces.Dict):
            vectorized_env = is_vectorized_observation(observation, self.observation_space)
            # Add batch dimension if needed
            observation = observation.reshape((-1, *self.observation_space.shape))  # type: ignore[misc]

        observation = obs_as_tensor(obs, self.device, is_dict_obs)
        return observation, vectorized_env
    
    def set_training_mode(self, mode: bool) -> None:
        self.training = mode 