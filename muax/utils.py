"""
MIT License

Copyright (c) 2022 Zeyu Zheng
Copyright (c) 2023 bf2504@columbia.edu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from collections import deque
from itertools import islice
from functools import partial
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
    r"""Scales the gradient while remaining the identical value during the forward pass."""
    return g * scale + jax.lax.stop_gradient(g) * (1. - scale)


@partial(jax.jit, static_argnums=(1, 2,))
def min_max(state, _min: float, _max: float):
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
    """Scalar to support"""
    x = _scaling(x)
    x = jnp.clip(x, -support_size, support_size)
    low = jnp.floor(x).astype(jnp.int32) 
    high = jnp.ceil(x).astype(jnp.int32)
    prob_high = x - low
    prob_low = 1. - prob_high
    idx_low = low + support_size
    idx_high = high + support_size
    support_low = jax.nn.one_hot(idx_low, 2 * support_size + 1) * prob_low[..., None]
    support_high = jax.nn.one_hot(idx_high, 2 * support_size + 1) * prob_high[..., None]
    return support_low + support_high


def support_to_scalar(probs, support_size):
    """Support to scalar"""

    x = jnp.sum(
      (jnp.arange(2*support_size+1) - support_size)
       * probs,
        axis=-1)
    x = _inv_scaling(x)
    return x 