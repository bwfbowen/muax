"""
MIT License

Copyright 2019 DeepMind Technologies Limited. 
Copyright (c) 2020 Microsoft Corporation.
Copyright (c) 2021 github.com/coax-dev
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
from scipy.linalg import pascal


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


def diff_transform_matrix(num_frames, dtype='float32'):
    r"""
    A helper function that implements discrete differentiation for stacked
    state observations.
    Let's say we have a feature vector :math:`X` consisting of four stacked
    frames, i.e. the shape would be: ``[batch_size, height, width, 4]``.
    The corresponding diff-transform matrix with ``num_frames=4`` is a
    :math:`4\times 4` matrix given by:
    .. math::
        M_\text{diff}^{(4)}\ =\ \begin{pmatrix}
            -1 &  0 &  0 & 0 \\
             3 &  1 &  0 & 0 \\
            -3 & -2 & -1 & 0 \\
             1 &  1 &  1 & 1
        \end{pmatrix}
    such that the diff-transformed feature vector is readily computed as:
    .. math::
        X_\text{diff}\ =\ X\, M_\text{diff}^{(4)}
    The diff-transformation preserves the shape, but it reorganizes the frames
    in such a way that they look more like canonical variables. You can think
    of :math:`X_\text{diff}` as the stacked variables :math:`x`,
    :math:`\dot{x}`, :math:`\ddot{x}`, etc. (in reverse order). These
    represent the position, velocity, acceleration, etc. of pixels in a single
    frame.
    Parameters
    ----------
    num_frames : positive int
        The number of stacked frames in the original :math:`X`.
    dtype : dtype, optional
        The output data type.
    Returns
    -------
    M : 2d-Tensor, shape: [num_frames, num_frames]
        A square matrix that is intended to be multiplied from the left, e.g.
        ``X_diff = K.dot(X_orig, M)``, where we assume that the frames are
        stacked in ``axis=-1`` of ``X_orig``, in chronological order.
    """
    assert isinstance(num_frames, int) and num_frames >= 1
    s = jnp.diag(jnp.power(-1, jnp.arange(num_frames)))  # alternating sign
    m = s.dot(pascal(num_frames, kind='upper'))[::-1, ::-1]
    return m.astype(dtype)


def diff_transform(X, dtype='float32'):
    r"""
    A helper function that implements discrete differentiation for stacked state observations. See
    :func:`diff_transform_matrix` for a detailed description.
    .. code:: python
        M = diff_transform_matrix(num_frames=X.shape[-1])
        X_transformed = np.dot(X, M)
    Parameters
    ----------
    X : ndarray
        An array whose shape is such that the last axis is the frame-stack axis, i.e.
        :code:`X.shape[-1] == num_frames`.
    Returns
    -------
    X_transformed : ndarray
        The shape is the same as the input shape, but the last axis are mixed to represent position,
        velocity, acceleration, etc.
    """
    M = diff_transform_matrix(num_frames=X.shape[-1], dtype=dtype)
    return jnp.dot(X, M)


def n_step_bootstrapped_returns(
    r_t,
    discount_t,
    v_t,
    n: int,
    lambda_t = 1.,
    stop_target_gradients: bool = False,
):
  """Computes strided n-step bootstrapped return targets over a sequence.
  The returns are computed according to the below equation iterated `n` times:
     Gₜ = rₜ₊₁ + γₜ₊₁ [(1 - λₜ₊₁) vₜ₊₁ + λₜ₊₁ Gₜ₊₁].
  When lambda_t == 1. (default), this reduces to
     Gₜ = rₜ₊₁ + γₜ₊₁ * (rₜ₊₂ + γₜ₊₂ * (... * (rₜ₊ₙ + γₜ₊ₙ * vₜ₊ₙ ))).
  Args:
    r_t: rewards at times [1, ..., T].
    discount_t: discounts at times [1, ..., T].
    v_t: state or state-action values to bootstrap from at time [1, ...., T].
    n: number of steps over which to accumulate reward before bootstrapping.
    lambda_t: lambdas at times [1, ..., T]. Shape is [], or [T-1].
    stop_target_gradients: bool indicating whether or not to apply stop gradient
      to targets.
  Returns:
    estimated bootstrapped returns at times [0, ...., T-1]
  """
  
  seq_len = r_t.shape[0]

  # Maybe change scalar lambda to an array.
  lambda_t = jnp.ones_like(discount_t) * lambda_t

  # Shift bootstrap values by n and pad end of sequence with last value v_t[-1].
  pad_size = min(n - 1, seq_len)
  targets = jnp.concatenate([v_t[n - 1:], jnp.array([v_t[-1]] * pad_size)])

  # Pad sequences. Shape is now (T + n - 1,).
  r_t = jnp.concatenate([r_t, jnp.zeros(n - 1)])
  discount_t = jnp.concatenate([discount_t, jnp.ones(n - 1)])
  lambda_t = jnp.concatenate([lambda_t, jnp.ones(n - 1)])
  v_t = jnp.concatenate([v_t, jnp.array([v_t[-1]] * (n - 1))])

  # Work backwards to compute n-step returns.
  for i in reversed(range(n)):
    r_ = r_t[i:i + seq_len]
    discount_ = discount_t[i:i + seq_len]
    lambda_ = lambda_t[i:i + seq_len]
    v_ = v_t[i:i + seq_len]
    targets = r_ + discount_ * ((1. - lambda_) * v_ + lambda_ * targets)

  return jax.lax.select(stop_target_gradients,
                        jax.lax.stop_gradient(targets), targets)    


def action2plane(action: int, shape):
    return jnp.broadcast_to(action, shape)