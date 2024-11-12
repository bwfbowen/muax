from functools import partial
import jax 
from jax import numpy as jnp
import optax

from muax import utils as mx_utils
from muax.model import MuZero, MZNetworkParams


@partial(jax.jit, static_argnums=(0,))
def default_loss_fn(muzero_instance: MuZero, params: MZNetworkParams, batch):
    r"""
    Computes the loss for the MuZero model using JAX.

    This function calculates the combined loss for the representation, dynamics, and prediction networks
    of the MuZero model. It uses the scalar_to_support function for reward `r` and n-step bootstrapping
    value `Rn`, as mentioned in the MuZero paper's Appendix.

    The loss is computed as the sum of three components:
    1. Reward prediction loss: cross_entropy(predicted_reward, actual_reward)
    2. Value prediction loss: cross_entropy(predicted_value, n_step_return)
    3. Policy prediction loss: cross_entropy(predicted_policy, actual_policy)

    The function also applies L2 regularization to all model parameters.

    Parameters
    ----------
    muzero_instance : MuZero
        An instance of the `MuZero` class containing the model architecture and hyperparameters.

    params : dict
        The parameters of the three neural networks (representation, prediction, and dynamics).

    batch : Transition
        A batch of transitions from the replay buffer. Each field in the batch has shape [B, L, ...],
        where B is the batch size, L is the trajectory length, and the remaining dimensions are
        specific to each field.

    Returns
    -------
    loss : jnp.array
        The computed loss value, combining prediction losses and L2 regularization.

    Implementation Details
    ----------------------
    - Uses jax.lax.scan for efficient loop processing over the trajectory steps.
    - Applies gradient scaling to the hidden state for improved training stability.
    - Computes losses using optax.softmax_cross_entropy for each prediction type.
    - Applies L2 regularization with a coefficient of 1e-4.
    """
    B, L = batch.a.shape
    batch.r = mx_utils.scalar_to_support(batch.r, muzero_instance._support_size).reshape(B, L, -1)
    batch.Rn = mx_utils.scalar_to_support(batch.Rn, muzero_instance._support_size).reshape(B, L, -1)
    
    initial_s = muzero_instance.repr_func.apply(params.representation, batch.obs[:, 0])

    def body_func(i, loss_s):
      loss, s = loss_s
      v, logits = muzero_instance.pred_func.apply(params.prediction, s)
      # Appendix G, scale the gradient at the start of the dynamics function by 1/2 
      s = mx_utils.scale_gradient(s, 0.5)
      r, ns = muzero_instance.dy_func.apply(params.dynamic, s, batch.a[:, i].flatten())
      # losses: reward
      loss_r = jnp.mean(
        optax.softmax_cross_entropy(r, 
        jax.lax.stop_gradient(batch.r[:, i])
        ))
      # losses: value
      loss_v = jnp.mean(
        optax.softmax_cross_entropy(v, 
        jax.lax.stop_gradient(batch.Rn[:, i])
        ))
      # losses: action weights
      loss_pi = jnp.mean(
        optax.softmax_cross_entropy(logits, 
        jax.lax.stop_gradient(batch.pi[:, i])
        ))

      loss += loss_r + loss_v + loss_pi 
      loss_s = (loss, ns)
      return loss_s 

    loss, _ = jax.lax.fori_loop(0, L, body_func, (loss, initial_s))

    l2_regularizer = 0.5 * sum(jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(params))
    loss += 1e-4 * l2_regularizer

    return loss
