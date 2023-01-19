from functools import partial
import jax 
from jax import numpy as jnp
import optax 


@partial(jax.jit, static_argnums=(0,))
def _update(optimizer, params, optimizer_state, grads):
  updates, optimizer_state = optimizer.update(grads, optimizer_state)
  params = optax.apply_updates(params, updates)
  return params, optimizer_state


@partial(jax.jit, static_argnums=(0, 3))
def _loss_fn(model, params, batch, c: float = 1e-4):
  loss = 0
  B, L, _ = batch.a.shape
  s = model._repr_apply(params['representation'], batch.obs[:, 0, :])
  # TODO: jax.lax.scan (or stay with fori_loop ?)
  def body_func(i, loss_s):
    loss, s = loss_s
    v, logits = model._pred_apply(params['prediction'], s)
    r, s = model._dy_apply(params['dynamic'], s, batch.a[:, i, :].flatten())
    # losses: reward
    loss_r = jnp.mean(optax.l2_loss(r, 
                                    jax.lax.stop_gradient(batch.r[:, i, :])
                                    ))
    # losses: value
    loss_v = jnp.mean(optax.l2_loss(v, 
                                    jax.lax.stop_gradient(batch.Rn[:, i, :])
                                    ))
    # losses: action weights
    loss_pi = jnp.mean(optax.softmax_cross_entropy(logits, 
                                                    jax.lax.stop_gradient(batch.pi[:, i, :])
                                                    ))

    loss += loss_r + loss_v + loss_pi 
    loss_s = (loss, s)
    return loss_s 
  loss, _ = jax.lax.fori_loop(1, L, body_func, (loss, s))
  # Appendix G Training: "irrespective of how many steps we unroll for"
  loss /= L 

  # L2 regularizer
  # TODO: BUG all nan
  # flatten_params, _ = jax.flatten_util.ravel_pytree(
  #     jax.tree_map(lambda theta: jnp.linalg.norm(theta, ord=2), params)
  #     )
  
  # loss += c * jnp.sum(flatten_params)
  # print(f'loss2: {loss}')
  return loss


def train(model, batch, optimizer, loss_fn=_loss_fn, update_fn=_update, c: float = 1e-4):
  loss, grads = jax.value_and_grad(loss_fn)(model, model.params, batch, c)
  model._params, model._opt_state = update_fn(optimizer, model._params, model._opt_state, grads)
  loss_metric = {'loss': loss.item()}
  return loss_metric  