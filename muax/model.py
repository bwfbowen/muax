from functools import partial
import numpy as np
import jax
from jax import numpy as jnp 
import mctx 
import optax
import haiku as hk 

from muax import utils as mx_utils 
from muax.loss import default_loss_fn
from muax.policy import MuZeroPolicy
from muax.nn import MZNetwork, MZNetworkParams
from muax.optimizers import create_optimizer


class MuZero:
  r"""Muzero algorithm
    
    Parameters
    ----------
    representation_fn: A function initialized from a class which inherents hk.Module, 
        which takes raw observation `obs` from the environment as input and outputs the hidden state `s`.
        `s` will be the input of prediction_fn and dynamic_fn. 
        The first dimension of the `obs` is the batch dimension.
    
    prediction_fn: A function initialized from a class which inherents hk.Module, 
        which takes hidden state `s` as input and outputs prior logits `logits` and value `v` of the state.
    
    dynamic_fn: A function initialized from a class which inherents hk.Module,
        which takes hidden state `s` and action `a` as input and outputs reward `r` and next hidden state `ns`.
    
    policy_class: A subclass of `muax.policy.Policy`. Determines which muzero policy to use.
    
    optimizer: Optimizer to update the parameters of `representation_fn`, `prediction_fn` and `dynamic_fn`.
    
    loss_fn: Callable, computes loss for the MuZero model. The default is `default_loss_fn`.
    
    discount: Any. Used for mctx.RecurrentFnOutput.

    support_size: int, the `support_size` for `scalar_to_support`, 
        the scale is nearly square root, that is, if the scalar is ~100, `support_size`=10 might be sufficient.
  """
  def __init__(self, 
               network: MZNetwork,
               policy_class=MuZeroPolicy,
               optimizer = create_optimizer(),
               loss_fn = default_loss_fn,
               discount: float = 0.99,
               support_size: int = 10
               ):
     
    self.repr_func = hk.without_apply_rng(hk.transform(network.representation_fn))
    self.pred_func = hk.without_apply_rng(hk.transform(network.prediction_fn))
    self.dy_func = hk.without_apply_rng(hk.transform(network.dynamic_fn))
    
    self._policy = policy_class()
    self._optimizer = optimizer 
    self.loss_fn = partial(loss_fn, self)
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
    params: MZNetworkParams. {'representation': repr_params, 'prediction': pred_params, 'dynamic': dy_params}
    """
    repr_params = self.repr_func.init(rng_key, sample_input)
    s = self.repr_func.apply(repr_params, sample_input)
    pred_params = self.pred_func.init(rng_key, s)
    dy_params = self.dy_func.init(rng_key, s, jnp.zeros(s.shape[0]))
    self._params = MZNetworkParams(repr_params, pred_params, dy_params)
    self._opt_state = self._optimizer.init(self._params)
    return self._params 

  def act(self, rng_key, obs,
          with_pi: bool = False,
          with_value: bool = False,
          obs_from_batch: bool = False,
          num_simulations: int = 5,
          temperature: float = 1.,
          invalid_actions=None,
          max_depth: int = None, 
          loop_fn = jax.lax.fori_loop,
          qtransform=None, 
          dirichlet_fraction: float = 0.25, 
          dirichlet_alpha: float = 0.3, 
          pb_c_init: float = 1.25, 
          pb_c_base: float = 19652,
        ):
    r"""Acts given environment's observations.
    
    Parameters
    ----------
    rng_key: jax.random.PRNGKey.

    obs: Array. The raw observations from environemnt.

    with_pi: bool. If True, the action logits `action weight` will be returned.
    
    with_value: bool. If True, the value of the root node (corresponding to the observation) will be returned.

    obs_from_batch: bool. If True, the first dimension should be the batch dimension. 
    If False, `jnp.expand_dims(obs, axis=0)` will be applied.
    
    num_simulations: int, positive int. Argument for mctx.muzero_policy and mctx.gumbel_muzero_policy. 
    The number of simulations.

    temperature: float,  Argument for mctx.muzero_policy. temperature for acting proportionally to
    `visit_counts**(1 / temperature)`.

    invalid_actions: Array. A mask with invalid actions. Argument for mctx.muzero_policy and mctx.gumbel_muzero_policy. 
    Invalid actions have ones, valid actions have zeros in the mask. Shape `[B, num_actions]`.

    max_depth: int, positive integer. Argument for mctx.muzero_policy and mctx.gumbel_muzero_policy. 
    Maximum search tree depth allowed during simulation.

    loop_fn: Callable. Argument for mctx.muzero_policy and mctx.gumbel_muzero_policy. 
    Function used to run the simulations. It may be required to pass
    `hk.fori_loop` if using this function inside a Haiku module.

    qtransform: Callable. Argument for mctx.muzero_policy and mctx.gumbel_muzero_policy.
    Function to obtain completed Q-values for a node. By default, the qtransform for mctx.muzero_policy is `qtransform_by_parent_and_siblings`, 
    and `qtransform_completed_by_mix_value` for mctx.gumbel_muzero_policy.

    dirichlet_fraction: float, from 0 to 1. Argument for mctx.muzero_policy. Interpolating between using only the
    prior policy or just the Dirichlet noise.

    dirichlet_alpha: float. Argument for mctx.muzero_policy. Concentration parameter to parametrize the Dirichlet
    distribution.

    pb_c_init: float. Argument for mctx.muzero_policy. constant c_1 in the PUCT formula.

    pb_c_base: float. Argument for mctx.muzero_policy. constant c_2 in the PUCT formula.

    max_num_considered_actions: int, positive integer. Argument for mctx.gumbel_muzero_policy.
    the maximum number of actions expanded at the root node. 
    A smaller number of actions will be expanded if the number of valid actions is smaller.

    gumbel_scale: float. Argument for mctx.gumbel_muzero_policy. scale for the Gumbel noise. 
    Evalution on perfect-information games can use gumbel_scale=0.0.
    
    
    Returns
    ----------
    action: int or array. If `obs_from_batch` is True, returns array. Else returns int
    
    action_weights: Array. If `with_pi` is True, the `action_weights` will be returned.

    root_value: floar or array. If `with_value` is True, the `root_value` will be returned.
    If `obs_from_batch` is True, returns array. Else returns int
    
    """
    if not obs_from_batch:
      obs = jnp.expand_dims(obs, axis=0)
    plan_output, root_value = self._plan(self.params, rng_key, obs, 
                                         num_simulations=num_simulations, 
                                         temperature=temperature,
                                         invalid_actions=invalid_actions,
                                         max_depth=max_depth, 
                                         loop_fn=loop_fn,
                                         qtransform=qtransform, 
                                         dirichlet_fraction=dirichlet_fraction, 
                                         dirichlet_alpha=dirichlet_alpha, 
                                         pb_c_init=pb_c_init, 
                                         pb_c_base=pb_c_base,)
    root_value = root_value.item() if not obs_from_batch else root_value
    action = plan_output.action.item() if not obs_from_batch else np.asarray(plan_output.action)

    if with_pi and with_value: return action, plan_output.action_weights, root_value
    elif not with_pi and with_value: return action, root_value
    elif with_pi and not with_value: return action, plan_output.action_weights
    else: return action

  def update(self, batch, *args, **kwargs):
    r"""Updates model parameters given a batch of trajectories.
    
    Parameters
    ----------
    batch: An instance of `Transition`.

        A batch from `replay_buffer.sample`. For each `field` in batch, it is of the shape `[B, L, E]`, 
        where `B` is the batch size, `L` is the length of the sample trajectory, `E` is the dimension of this `field`.
        For instance, the shape of `batch.r` could be `[32, 10, 1]`, which represents there are 32 trajectories sampled, each has a length of 10,
        and the reward corresponding to each step is a scalar.

    Returns
    -------
    loss_metric: dict. The key is 'loss', the value is float.
    """
    loss, grads = jax.value_and_grad(self.loss_fn)(self._params, batch, *args, **kwargs)
    self._params, self._opt_state = self._update(self._params, self._opt_state, grads)
    loss_metric = {'loss': loss.item()}
    
    return loss_metric
  
  def save_load(self, file, save=True):
    """Saves or loads model parameters and optimizer state to/from the file"""
    if save:
      to_save = {'params': self.params, 'optimizer_state': self.optimizer_state}
      jnp.save(file, to_save)
    else:
      if not file.endswith('.npy'):
        file = f'{file}.npy'
      saved = jnp.load(file, allow_pickle=True).item()
      self._params, self._opt_state = saved['params'], saved['optimizer_state']

  @property
  def params(self):
    return self._params

  @property
  def optimizer_state(self):
    return self._opt_state
  
  @partial(jax.jit, static_argnums=(0, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13))
  def _plan(self, params, rng_key, obs, 
            num_simulations=5, temperature=1., invalid_actions=None, 
            max_depth=None, loop_fn=jax.lax.fori_loop, 
            qtransform=None, dirichlet_fraction=0.25, 
            dirichlet_alpha=0.3, pb_c_init=1.25, pb_c_base=19652):
    
    root = self._root_inference(params, rng_key, obs)
    if qtransform is None:
      qtransform = mctx.qtransform_by_parent_and_siblings
    plan_output = self._policy(params, rng_key, root, self._recurrent_inference,
                               num_simulations=num_simulations,
                               temperature=temperature,
                               invalid_actions=invalid_actions,
                               max_depth=max_depth,
                               loop_fn=loop_fn,
                               qtransform=qtransform,
                               dirichlet_fraction=dirichlet_fraction,
                               dirichlet_alpha=dirichlet_alpha,
                               pb_c_init=pb_c_init,
                               pb_c_base=pb_c_base)
    return plan_output, root.value
    
  @partial(jax.jit, static_argnums=(0,))
  def _update(self, params, optimizer_state, grads):
    updates, optimizer_state = self._optimizer.update(grads, optimizer_state)
    params = optax.apply_updates(params, updates)
    return params, optimizer_state

  @partial(jax.jit, static_argnums=(0,))
  def _root_inference(self, params, rng_key, obs):
    r"""Given the observation, a (prior_logits, value, embedding) RootFnOutput is estimated. The
    prior_logits are from a policy network. The shapes are ([B, num_actions], [B], [B, ...]), respectively."""
    s = self.repr_func.apply(params.representation, obs)
    v, logits = self.pred_func.apply(params.prediction, s)
    v = mx_utils.support_to_scalar(jax.nn.softmax(v), self._support_size).flatten()
    root = mctx.RootFnOutput(
        prior_logits=logits,
        value=v,
        embedding=s
    )
    return root 

  @partial(jax.jit, static_argnums=(0,))
  def _recurrent_inference(self, params, rng_key, action, embedding):
    r"""To be called on the leaf nodes and unvisited actions retrieved by the simulation step,
    which takes as args (params, rng_key, action, embedding) and returns a `RecurrentFnOutput` and the new state embedding.
    The rng_key argument is consumed.
    """
    r, next_embedding = self.dy_func.apply(params.dynamic, embedding, action)
    v, logits = self.pred_func.apply(params.prediction, next_embedding)
    r = mx_utils.support_to_scalar(jax.nn.softmax(r), self._support_size).flatten()
    v = mx_utils.support_to_scalar(jax.nn.softmax(v), self._support_size).flatten()
    discount = jnp.ones_like(r) * self._discount
    recurrent_output = mctx.RecurrentFnOutput(
        reward=r,
        discount=discount,
        prior_logits=logits,
        value=v 
    )
    return recurrent_output, next_embedding

