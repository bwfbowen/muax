from collections import deque
from itertools import islice
import jax
from jax import numpy as jnp


class sliceable_deque(deque):
    r"""A class implemented slice for collections.deque
    
    Reference: https://stackoverflow.com/questions/10003143/how-to-slice-a-deque
    """
    def __getitem__(self, index):
        if isinstance(index, slice):
            return type(self)(islice(self, index.start,
                                               index.stop, index.step))
        return deque.__getitem__(self, index)

jax.tree_util.register_pytree_node(
  sliceable_deque,
  flatten_func=lambda sd: (sd, None),
  unflatten_func=lambda treedef, leaves: sliceable_deque(leaves)
)


def scale_gradient(g, scale: float = 1):
    return g * scale + jax.lax.stop_gradient(g) * (1. - scale)


def min_max(state):
    _min = jax.lax.stop_gradient(jnp.min(state))
    _max = jax.lax.stop_gradient(jnp.max(state))
    return (state - _min) / (_max - _min)   


def _scaling(x, eps: float = 1e-3):
  """Reference: https://arxiv.org/abs/1805.11593"""
  return jnp.sign(x) * (jnp.sqrt(jnp.abs(x) + 1) - 1) + eps * x


def _inv_scaling(x, eps: float = 1e-3):
  """The inverse scaling of the output of `_scaling`"""
  return jnp.sign(x) * (
        ((jnp.sqrt(1 + 4 * eps * (jnp.abs(x) + 1 + eps)) - 1) / (2 * eps))
        ** 2
        - 1
    )


def scalar_to_support(x, support_size):
    x = _scaling(x)
    x = jnp.clip(x, -support_size, support_size)
    low = jnp.floor(x).astype(jnp.int32) 
    high = jnp.ceil(x).astype(jnp.int32)
