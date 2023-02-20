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
    return (state-jnp.min(state)) / (jnp.max(state)-jnp.min(state))   