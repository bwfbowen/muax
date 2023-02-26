from functools import partial
import jax
from jax import numpy as jnp 
import mctx 
import optax
import haiku as hk 

import warnings

from .utils import scale_gradient, scalar_to_support, support_to_scalar, min_max


class MuZero:
  r"""Muzero algorithm
  
    TODO: more args, more flexible for repr, pred, dy modules
    
    Parameters
    ----------
    embedding_dim: Any. 
        The embedding dimension of hidden state `s`. Depends on the representation module. 
    num_actions: Any. 
        The maximum number of actions the policy can take. Depends on the prediction module and dynamic module.
    representation_fn: A function initialized from a class which inherents hk.Module, 
        which takes raw observation `obs` from the environment as input and outputs the hidden state `s`.
        `s` will be the input of prediction_module and dynamic_module.
    prediction_fn: A function initialized from a class which inherents hk.Module, 
        which takes hidden state `s` as input and outputs prior logits `logits` and value `v` of the state.
    dynamic_fn: A function initialized from a class which inherents hk.Module,
        which takes hidden state `s` and action `a` as input and outputs reward `r` and next hidden state `ns`.
    policy: str, value in `['muzero', 'gumbel']`. Determines which muzero policy in mctx to use. 
    optimizer: Optimizer to update the parameters of `representation_module`, `prediction_module` and `dynamic_module`.
    discount: Any. Used for mctx.RecurrentFnOutput.
  """
  def __init__(self, 
               representation_fn,
               prediction_fn,
               dynamic_fn,
               policy='muzero',
               optimizer = optax.chain(
                 optax.clip_by_global_norm(1.0),  
                 optax.scale_by_adam(),  
                 optax.scale_by_schedule(
                   optax.warmup_exponential_decay_schedule(
                     init_value=0, 
                     peak_value=2e-2,
                     end_value=1e-4,
                     warmup_steps=10000,
                     transition_steps=10000,
                     decay_rate=0.8)
                     ),  
                optax.scale(-1.0)
                ),
               discount: float = 0.99,
               support_size: int = 10
               ):
     
    self.repr_func = hk.without_apply_rng(hk.transform(representation_fn))
    self.pred_func = hk.without_apply_rng(hk.transform(prediction_fn))
    self.dy_func = hk.without_apply_rng(hk.transform(dynamic_fn))
    
    self._policy = self._init_policy(policy)
    self._optimizer = optimizer 
    self._discount = discount
    self._support_size = support_size
  
  def init(self, rng_key, sample_input):
    """Inits `representation`, `prediction` and `dynamic` functions and optimizer
    
    Parameters
    ----------
    rng_key: jax.random.PRNGKey.
    sample_input: Array. The dimension is `[B, ...]` where B is the batch dimension.
    
    Returns
    ----------
    params: dict. {'representation': repr_params, 'prediction': pred_params, 'dynamic': dy_params}
    """
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

  def act(self, rng_key, obs,
          with_pi: bool = False,
          with_value: bool = False,
          obs_from_batch: bool = False,
          num_simulations: int = 5,
          *args, **kwargs):
    """Acts given environment's observations.
    
    Parameters
    ----------
    rng_key: jax.random.PRNGKey.
    obs: Array. The raw observations from environemnt.
    with_pi: bool.
    with_value: bool.
    obs_from_batch: bool.
    num_simulations: int, positive int. Argument for mctx.muzero_policy
    args, kwargs: Arguments for mctx.muzero_policy
    
    Returns
    ----------
    
    """
    if not obs_from_batch:
      obs = jnp.expand_dims(obs, axis=0)
    plan_output, root_value = self._plan(self.params, rng_key, obs, num_simulations,
                                        *args, **kwargs)
    root_value = root_value.item() if not obs_from_batch else root_value
    action = plan_output.action.item() if not obs_from_batch else plan_output.action

    if with_pi and with_value: return action, plan_output.action_weights, root_value
    elif not with_pi and with_value: return action, root_value
    elif with_pi and not with_value: return action, plan_output.action_weights
    else: return action

  def update(self, batch, c: float = 1e-4):
    loss, grads = jax.value_and_grad(self._loss_fn)(self._params, batch, c)
    self._params, self._opt_state = self._update(self._params, self._opt_state, grads)
    loss_metric = {'loss': loss.item()}
    return loss_metric
  
  def save(self, file):
    """Saves model parameters and optimizer state to the file"""
    to_save = {'params': self.params, 'optimizer_state': self.optimizer_state}
    jnp.save(file, to_save)
  
  def load(self, file):
    """Loads model parameters and optimizer state from the saved file"""
    saved = jnp.load(file, allow_pickle=True).item()
    self._params, self._opt_state = saved['params'], saved['optimizer_state']

  @property
  def params(self):
    return self._params

  @property
  def optimizer_state(self):
    return self._opt_state

  def _plan(self, params, rng_key, obs,
           num_simulations: int = 5,
           *args, **kwargs):
    root = self._root_inference(params, rng_key, obs)
    plan_output = self._policy(params, rng_key, root, self._recurrent_inference,
                               num_simulations=num_simulations,
                               *args, **kwargs)
    return plan_output, root.value
    
  @partial(jax.jit, static_argnums=(0,))
  def _update(self, params, optimizer_state, grads):
    updates, optimizer_state = self._optimizer.update(grads, optimizer_state)
    params = optax.apply_updates(params, updates)
    return params, optimizer_state
  
  @partial(jax.jit, static_argnums=(0, 3))
  def _loss_fn(self, params, batch, c: float = 1e-4):
    loss = 0
    B, L, _ = batch.a.shape
    batch.r = scalar_to_support(batch.r, self._support_size)
    batch.Rn = scalar_to_support(batch.Rn, self._support_size)
    s = self._repr_apply(params['representation'], batch.obs[:, 0, :])
    # TODO: jax.lax.scan (or stay with fori_loop ?)
    def body_func(i, loss_s):
      loss, s = loss_s
      v, logits = self._pred_apply(params['prediction'], s)
      # Appendix G, scale the gradient at the start of the dynamics function by 1/2 
      s = scale_gradient(s, 0.5)
      r, s = self._dy_apply(params['dynamic'], s, batch.a[:, i, :].flatten())
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

  @partial(jax.jit, static_argnums=(0,))
  def _root_inference(self, params, rng_key, obs):
    s = self._repr_apply(params['representation'], obs)
    v, logits = self._pred_apply(params['prediction'], s)  
    v = support_to_scalar(jax.nn.softmax(v), self._support_size).flatten()
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
    r = support_to_scalar(jax.nn.softmax(r), self._support_size).flatten()
    v = support_to_scalar(jax.nn.softmax(v), self._support_size).flatten()
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

  def _init_policy(self, policy):
    if policy == 'muzero':
      policy_func = mctx.muzero_policy
    elif policy == 'gumbel':
      policy_func = mctx.gumbel_muzero_policy
    else:
      warnings.warn(f"{policy} is not in ['muzero', 'gumbel'], uses muzero policy instead")
      policy_func = mctx.muzero_policy
    return jax.jit(policy_func, static_argnums=(3, 4, ), backend='cpu')


