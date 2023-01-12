from functools import partial
import jax
from jax import numpy as jnp 
import mctx 
import optax
import haiku as hk 


class MuZero:
  def __init__(self, 
               embedding_dim,
               num_actions,
               representation_module,
               prediction_module,
               dynamic_module,
               optimizer = optax.adam(0.01),
               discount: float = 0.99
               ):
    self.repr_func = self._init_representation_func(representation_module, 
                                                    embedding_dim) 
    self.repr_func = hk.without_apply_rng(hk.transform(self.repr_func))

    self.pred_func = self._init_prediction_func(prediction_module, 
                                                num_actions)
    self.pred_func = hk.without_apply_rng(hk.transform(self.pred_func))

    self.dy_func = self._init_dynamic_func(dynamic_module, 
                                           embedding_dim, num_actions)
    self.dy_func = hk.without_apply_rng(hk.transform(self.dy_func))
    
    self._optimizer = optimizer 
    self._discount = discount
  
  def init(self, rng_key, sample_input):
    # TODO doc sample_input shape requirement
    repr_params = self.repr_func.init(rng_key, sample_input)
    s = self.repr_func.apply(repr_params, sample_input)
    pred_params = self.pred_func.init(rng_key, s)
    dy_params = self.dy_func.init(rng_key, s, jnp.zeros(s.shape[0]))
    self._params = {'representation': repr_params, 
                   'prediction': pred_params, 
                   'dynamic': dy_params}
    self._opt_state = self._optimizer.init(self._params)
    return self._params 

  def representation(self, obs):
    s = self._repr_apply(self.params['representation'], obs)
    return s 
  
  def prediction(self, s):
    v, logits = self._pred_apply(self.params['prediction'], s)
    return v, logits

  def dynamic(self, s, a):
    r, ns = self._dy_apply(self.params['dynamic'], s, a)
    return r, ns

  @partial(jax.jit, static_argnums=(0, 3, 4, 5))
  def plan(self, rng_key, obs,
           with_value: bool = True,
           obs_from_batch: bool = False,
           num_simulations: int = 5,
           *args, **kwargs):
    params = self._params
    root = self._root_inference(params, rng_key, obs, obs_from_batch)
    plan_output = mctx.muzero_policy(params, rng_key, root, self._recurrent_inference,
                                     num_simulations=num_simulations,
                                     *args, **kwargs)
    
    if not with_value: return plan_output
    else: return plan_output, root.value

  def act(self, rng_key, obs,
          with_pi: bool = True,
          with_value: bool = True,
          obs_from_batch: bool = False,
          num_simulations: int = 5,
          *args, **kwargs):
    if with_value:
      plan_output, root_value = self.plan(rng_key, obs, with_value, 
                                          obs_from_batch, num_simulations,
                                          *args, **kwargs)
      root_value = root_value.item() if not obs_from_batch else root_value
    else:
      plan_output = self.plan(rng_key, obs, with_value, 
                              obs_from_batch, num_simulations,
                              *args, **kwargs)
    action = plan_output.action.item() if not obs_from_batch else plan_output.action

    if with_pi and with_value: return action, plan_output.action_weights, root_value
    elif not with_pi and with_value: return action, root_value
    elif with_pi and not with_value: return action, plan_output.action_weights
    else: return action

  def train(self, batch, c: float = 1e-4):
    loss, grads = jax.value_and_grad(self._loss_fn)(self._params, batch, c)
    updates, self._opt_state = self._optimizer.update(grads, self._opt_state)
    self._params = optax.apply_updates(self._params, updates)
    loss_metric = {'loss': loss.item()}
    return loss_metric

  @property
  def params(self):
    return self._params

  @property
  def optimizer_state(self):
    return self._opt_state
  
  @partial(jax.jit, static_argnums=(0, 3))
  def _loss_fn(self, params, batch, c: float = 1e-4):
    loss = 0
    B, L, _ = batch.a.shape
    s = self._repr_apply(params['representation'], batch.obs[:, 0, :])
    # TODO: jax.lax.scan (or stay with fori_loop ?)
    def body_func(i, loss_s):
      loss, s = loss_s
      v, logits = self._pred_apply(params['prediction'], s)
      r, s = self._dy_apply(params['dynamic'], s, batch.a[:, i, :].flatten())
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

  @partial(jax.jit, static_argnums=(0, 4))
  def _root_inference(self, params, rng_key, obs, obs_from_batch: bool = False):
    s = self._repr_apply(params['representation'], obs)
    v, logits = self._pred_apply(params['prediction'], s)
    if not obs_from_batch:
      s = jnp.expand_dims(s, axis=0)
      logits = jnp.expand_dims(logits, axis=0)
    else:
      v = v.flatten()
    root = mctx.RootFnOutput(
        prior_logits=logits,
        value=v,
        embedding=s
    )
    return root 

  @partial(jax.jit, static_argnums=(0,))
  def _recurrent_inference(self, params, rng_key, action, embedding):
    r, next_embedding = self._dy_apply(params['dynamic'], embedding, action)
    v, logits = self._pred_apply(params['prediction'], embedding)
    r, v = r.flatten(), v.flatten()
    discount = jnp.ones_like(r) * self._discount
    recurrent_output = mctx.RecurrentFnOutput(
        reward=r,
        discount=discount,
        prior_logits=logits,
        value=v 
    )
    return recurrent_output, next_embedding
  
  @partial(jax.jit, static_argnums=(0,))
  def _repr_apply(self, repr_params, obs):
    s = self.repr_func.apply(repr_params, obs)
    return s

  @partial(jax.jit, static_argnums=(0,))
  def _pred_apply(self, pred_params, s):
    v, logits = self.pred_func.apply(pred_params, s)
    return v, logits

  @partial(jax.jit, static_argnums=(0,))
  def _dy_apply(self, dy_params, s, a):
    r, ns = self.dy_func.apply(dy_params, s, a)
    return r, ns

  def _init_representation_func(self, representation_module, embedding_dim):
    def representation_func(obs):
      repr_model = representation_module(embedding_dim)
      return repr_model(obs)
    return representation_func
  
  def _init_prediction_func(self, prediction_module, num_actions):
    def prediction_func(s):
      pred_model = prediction_module(num_actions)
      return pred_model(s)
    return prediction_func

  def _init_dynamic_func(self, dynamic_module, embedding_dim, num_actions):
    def dynamic_func(s, a):
      dy_model = dynamic_module(embedding_dim, num_actions)
      return dy_model(s, a)
    return dynamic_func 