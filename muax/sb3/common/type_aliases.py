import numpy as np 
from typing import NamedTuple, Dict
import jaxlib

Array = jaxlib.xla_extension.ArrayImpl
ArrayDict = Dict[str, Array]


class MuaxRolloutBufferSamples(NamedTuple):
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    Rn: np.ndarray
    pi: np.ndarray
    weights: np.ndarray


class DictMuaxRolloutBufferSamples(NamedTuple):
    observations: Dict[str, np.ndarray]
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    Rn: np.ndarray
    values: np.ndarray
    pi: np.ndarray
    weights: np.ndarray

