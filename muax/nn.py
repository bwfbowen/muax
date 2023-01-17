import jax
from jax import numpy as jnp 
import haiku as hk


class Representation(hk.Module):
  def __init__(self, embedding_dim, name='representation'):
    super().__init__(name=name)

    self.repr_func = hk.Sequential([
        hk.Linear(embedding_dim), jax.nn.relu
    ])

  def __call__(self, obs):
    s = self.repr_func(obs)
    return s 


class Prediction(hk.Module):
  def __init__(self, num_actions, name='prediction'):
    super().__init__(name=name)        
    
    self.v_func = hk.Sequential([
        hk.Linear(16), jax.nn.relu,
        hk.Linear(1)
    ])
    self.pi_func = hk.Sequential([
        hk.Linear(16), jax.nn.relu,
        hk.Linear(num_actions)
    ])
  
  def __call__(self, s):
    v = self.v_func(s)
    logits = self.pi_func(s)
    logits = jax.nn.softmax(logits, axis=-1)
    return v, logits


class Dynamic(hk.Module):
  def __init__(self, embedding_dim, num_actions, name='dynamic'):
    super().__init__(name=name)
    
    self.ns_func = hk.Sequential([
        hk.Linear(16), jax.nn.relu,
        hk.Linear(embedding_dim)
    ])
    self.r_func = hk.Sequential([
        hk.Linear(16), jax.nn.relu,
        hk.Linear(1)
    ])
    self.cat_func = jax.jit(lambda s, a: 
                            jnp.concatenate([s, jax.nn.one_hot(a, num_actions)],
                                            axis=1)
                            )
  
  def __call__(self, s, a):
    sa = self.cat_func(s, a)
    r = self.r_func(sa)
    ns = self.ns_func(sa)
    return r, ns
    