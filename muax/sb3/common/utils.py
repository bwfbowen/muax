from typing import Union, Dict
import numpy as np 
import jax 

from .type_aliases import Array, ArrayDict

def obs_as_tensor(obs: Union[np.ndarray, Dict[str, np.ndarray]], device: jax.Device = None, is_dict_obs: bool = False) -> Union[Array, ArrayDict]:
    """
    Moves the observation to the given device.

    :param obs:
    :param device: Jax device
    :return: Jax numpy array of the observation on a desired device.
    """
    if not is_dict_obs:
        return jax.device_put(obs, device=device)
    else:
        return {key: jax.device_put(_obs, device=device) for (key, _obs) in obs.items()}
    