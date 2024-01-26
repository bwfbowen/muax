import numpy as np 
import jax 
from jax import numpy as jnp
from acme.adders.reverb import base as reverb_base
from acme.agents.jax import actor_core as actor_core_lib


def temperature_fn(max_training_steps, training_steps):
  r"""Determines the randomness for the action taken by the model"""
  if training_steps < 0.5 * max_training_steps:
      return 1.0
  elif training_steps < 0.75 * max_training_steps:
      return 0.5
  else:
      return 0.25


def get_priority_fn_with_reanalyse(policy: actor_core_lib.ActorCore) -> reverb_base.PriorityFn:
    
    def priority_fn(trajectory: reverb_base.PriorityFnInput):
        return 
    
    return  


def get_priority_fn_with_replaybuffer(policy: actor_core_lib.ActorCore) -> reverb_base.PriorityFn:
    
    def priority_fn(trajectory: reverb_base.PriorityFnInput):
        return 

    return  


@jax.jit
def min_max_normalize(s):
  s_min = s.min(axis=1, keepdims=True)
  s_max = s.max(axis=1, keepdims=True)
  s_scale = s_max - s_min 
  s_scale = jnp.where(s_scale < 1e-5, s_scale + 1e-5, s_scale)
  s_normed = (s - s_min) / (s_scale)
  return s_normed


def _fetch_devicearray(x):
  if isinstance(x, jax.Array):
    return np.asarray(x)
  return x


def get_from_first_device(nest, as_numpy: bool = True):
  """Gets the first array of a nest of `jax.pxla.ShardedDeviceArray`s."""
  # TODO(abef): remove this when fake_pmap is fixed or acme error is removed.

  def _slice_and_maybe_to_numpy(x):
    x = x[0]
    return _fetch_devicearray(x) if as_numpy else x

  return jax.tree_map(_slice_and_maybe_to_numpy, nest)


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


def scale_gradient(g, scale: float = 1):
    r"""Scales the gradient while remaining the identical value during the forward pass."""
    return g * scale + jax.lax.stop_gradient(g) * (1. - scale)