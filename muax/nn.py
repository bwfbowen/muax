import jax
from jax import numpy as jnp 
import haiku as hk


class Representation(hk.Module):
  def __init__(self, embedding_dim, name='representation'):
    super().__init__(name=name)
    self.ffn = hk.Sequential([
        hk.Linear(8), jax.nn.relu,
        hk.Linear(8), jax.nn.relu,
        hk.Linear(8), jax.nn.relu
    ])
    self.repr_func = hk.Linear(embedding_dim)

  def __call__(self, obs):
    obs = self.ffn(obs)
    s = self.repr_func(obs)
    return s 


class Prediction(hk.Module):
  def __init__(self, num_actions, name='prediction'):
    super().__init__(name=name)        
    self.ffn = hk.Sequential([
        hk.Linear(8), jax.nn.relu,
        hk.Linear(8), jax.nn.relu,
        hk.Linear(8), jax.nn.relu
    ])
    self.v_func = hk.Linear(1)
    self.pi_func = hk.Sequential([
        hk.Linear(num_actions)
        ])
  
  def __call__(self, s):
    s = self.ffn(s)
    v = self.v_func(s)
    logits = self.pi_func(s)
    logits = jax.nn.softmax(logits, axis=-1)
    return v, logits


class Dynamic(hk.Module):
  def __init__(self, embedding_dim, num_actions, name='dynamic'):
    super().__init__(name=name)
    self.ffn = hk.Sequential([
        hk.Linear(8), jax.nn.relu,
        hk.Linear(8), jax.nn.relu,
        hk.Linear(8), jax.nn.relu
    ])
    self.ns_func = hk.Linear(embedding_dim)
    self.r_func = hk.Linear(1)
    self.cat_func = jax.jit(lambda s, a: 
                            jnp.concatenate([s, jax.nn.one_hot(a, num_actions)],
                                            axis=1)
                            )
  
  def __call__(self, s, a):
    sa = self.cat_func(s, a)
    sa = self.ffn(sa)
    r = self.r_func(sa)
    ns = self.ns_func(sa)
    return r, ns
    