from functools import partial
import jax 
from jax import numpy as jnp
import optax

from .utils import scalar_to_support, scale_gradient


@partial(jax.jit, static_argnums=(0, ))
def default_loss_fn(muzero_instance, params, batch):
    r"""
    Computes loss for MuZero model. 
    Uses `scalar_to_support` for reward `r` and n-step bootstrapping value `Rn`. A technique mentioned in the paper's Appendix.
    The loss is the sum of `cross_entropy(u, r)`, `cross_entropy(Rn, v)` and `cross_entropy(logits, pi)`, and is regularised by L2.

    Parameters
    ----------
    muzero_instance : An instance of `MuZero` class

    params : `muzero_instance.params`

        The parameters of each of the three neural networks(representation, prediction and dynamic)

    batch: An instance of `Transition`.

        A batch from `replay_buffer.sample`. For each `field` in batch, it is of the shape `[B, L, E]`, 
        where `B` is the batch size, `L` is the length of the sample trajectory, `E` is the dimension of this `field`.
        For instance, the shape of `batch.r` could be `[32, 10, 1]`, which represents there are 32 trajectories sampled, each has a length of 10,
        and the reward corresponding to each step is a scalar.
    
    Returns
    -------
    loss: jnp.array. The loss calculated.
    """
    # Use muzero_instance to access required methods and attributes
    loss = 0
    c = 1e-4
    B, L, _ = batch.a.shape
    batch.r = scalar_to_support(batch.r, muzero_instance._support_size).reshape(B, L, -1)
    batch.Rn = scalar_to_support(batch.Rn, muzero_instance._support_size).reshape(B, L, -1)
    s = muzero_instance._repr_apply(params['representation'], batch.obs[:, 0, :])
    # TODO: jax.lax.scan (or stay with fori_loop ?)
    def body_func(i, loss_s):
      loss, s = loss_s
      v, logits = muzero_instance._pred_apply(params['prediction'], s)
      # Appendix G, scale the gradient at the start of the dynamics function by 1/2 
      s = scale_gradient(s, 0.5)
      r, s = muzero_instance._dy_apply(params['dynamic'], s, batch.a[:, i, :].flatten())
      # losses: reward
      loss_r = jnp.mean(
        optax.softmax_cross_entropy(r, 
        jax.lax.stop_gradient(batch.r[:, i, :])
        ))
      # losses: value
      loss_v = jnp.mean(
        optax.softmax_cross_entropy(v, 
        jax.lax.stop_gradient(batch.Rn[:, i, :])
        ))
      # losses: action weights
      loss_pi = jnp.mean(
        optax.softmax_cross_entropy(logits, 
        jax.lax.stop_gradient(batch.pi[:, i, :])
        ))

      loss += loss_r + loss_v + loss_pi 
      loss_s = (loss, s)
      return loss_s 
    loss, _ = jax.lax.fori_loop(1, L, body_func, (loss, s))
    # Appendix G Training: "irrespective of how many steps we unroll for"
    loss /= L 

    # L2 regulariser
    l2_regulariser = 0.5 * sum(
      jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(params))
    loss += c * jnp.sum(l2_regulariser)
    # print(f'loss2: {loss}')
    return loss 
