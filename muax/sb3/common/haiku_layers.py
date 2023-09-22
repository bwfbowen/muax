from typing import Union, List, Dict, Type
import gymnasium as gym 
import haiku as hk 
import jax 
from jax import numpy as jnp 
from ..common import min_max_normalize


class Representation(hk.Module):
    def __init__(
        self, 
        embedding_dim: int, 
        net_arch: List[int],
        activation_fn: jax.nn,
        name='representation'
    ):
        super().__init__(name=name)
        repr_func = []
        for curr_layer_dim in net_arch:
            repr_func.append(hk.Linear(curr_layer_dim))
            repr_func.append(activation_fn)
            repr_func.append(hk.Linear(embedding_dim))
        self.repr_func = hk.Sequential(repr_func)

    def __call__(self, obs):
        s = self.repr_func(obs)
        s = min_max_normalize(s)
        return s 


class Prediction(hk.Module):
    def __init__(
        self, 
        num_actions: int, 
        full_support_size: int, 
        net_arch: Union[List[int], Dict[str, List[int]]],
        activation_fn: jax.nn,
        name='prediction'
    ):
        super().__init__(name=name)        
        v_func = []
        pi_func = []
        
        self.v_func = hk.Sequential([
        hk.Linear(16), jax.nn.elu,
        hk.Linear(full_support_size)
    ])
        self.pi_func = hk.Sequential([
        hk.Linear(16), jax.nn.elu,
        hk.Linear(num_actions)
    ])
  
    def __call__(self, s):
        v = self.v_func(s)
        logits = self.pi_func(s)
        # logits = jax.nn.softmax(logits, axis=-1)
        return v, logits


class Dynamic(hk.Module):
    def __init__(self, embedding_dim, num_actions, full_support_size, name='dynamic'):
        super().__init__(name=name)
    
        self.ns_func = hk.Sequential([
        hk.Linear(16), jax.nn.elu,
        hk.Linear(embedding_dim)
    ])
        self.r_func = hk.Sequential([
        hk.Linear(16), jax.nn.elu,
        hk.Linear(full_support_size)
    ])
        self.cat_func = jax.jit(lambda s, a: 
                            jnp.concatenate([s, jax.nn.one_hot(a, num_actions)],
                                            axis=1)
                            )
  
    def __call__(self, s, a):
        sa = self.cat_func(s, a)
        r = self.r_func(sa)
        ns = self.ns_func(sa)
        ns = min_max_normalize(ns)
        return r, ns


class BaseFeaturesExtractorJax(hk.Module):
    """
    Base class that represents a features extractor.

    :param observation_space:
    :param features_dim: Number of features extracted.
    """
    def __init__(self, observation_space: gym.Space, features_dim: int = 0) -> None:
        super().__init__()
        assert features_dim > 0
        self._observation_space = observation_space
        self._features_dim = features_dim
    
    @property
    def features_dim(self) -> int:
        return self._features_dim
    

