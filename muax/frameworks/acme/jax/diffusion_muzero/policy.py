"""Search policy for Diffusion MuZero"""
from typing import Any, Callable, Generic, TypeVar, Tuple
import functools
from typing import Optional, Tuple

import chex
import jax
import jax.numpy as jnp

from mctx._src import action_selection
from mctx._src import base
from mctx._src import qtransforms
from mctx._src import search
from mctx._src import seq_halving
from mctx._src import policies

from muax.frameworks.acme.jax.diffusion_muzero import types as dmz_types


def diffusion_muzero(
    params: chex.ArrayTree,
    rng_key: chex.PRNGKey,
    root: base.RootFnOutput,
    decision_recurrent_fn: base.DecisionRecurrentFn,
    chance_recurrent_fn: base.ChanceRecurrentFn,  # Should return samples and their log-likelihoods
    num_simulations: int,
    num_samples: int,  # Number of diffusion samples (num_chance_outcomes)
    invalid_actions: Optional[chex.Array] = None,
    max_depth: Optional[int] = None,
    loop_fn: base.LoopFn = jax.lax.fori_loop,
    *,
    qtransform: base.QTransform = qtransforms.qtransform_by_parent_and_siblings,
    dirichlet_fraction: chex.Numeric = 0.25,
    dirichlet_alpha: chex.Numeric = 0.3,
    pb_c_init: chex.Numeric = 1.25,
    pb_c_base: chex.Numeric = 19652,
    temperature: chex.Numeric = 1.0) -> base.PolicyOutput[None]:
  """Runs Diffusion MuZero with diffusion model handling stochastic transitions and actions.

  Args:
    params: Parameters forwarded to recurrent functions.
    rng_key: Random number generator state, which is consumed.
    root: A `RootFnOutput` containing the root prior_logits, value, and embedding.
    decision_recurrent_fn: Callable for decision nodes and generates samples and their log-likehoods.
    chance_recurrent_fn: Callable for chance nodes that generates samples and their log-likelihoods.
    num_simulations: The number of simulations.
    num_samples: The number of diffusion samples (chance outcomes).
    invalid_actions: Mask with invalid actions.
    max_depth: Maximum search tree depth allowed during simulation.
    loop_fn: Function used to run the simulations.
    qtransform: Function to obtain completed Q-values for a node.
    dirichlet_fraction: Fraction of Dirichlet noise to add to the priors.
    dirichlet_alpha: Alpha parameter for the Dirichlet distribution.
    pb_c_init: Constant c1 in the PUCT formula.
    pb_c_base: Constant c2 in the PUCT formula.
    temperature: Temperature parameter for action selection.

  Returns:
    `PolicyOutput` containing the proposed action, action_weights, and the search tree.
  """

  # Splitting RNG keys
  rng_key, dirichlet_rng_key, search_rng_key = jax.random.split(rng_key, 3)

  # Adding Dirichlet noise to the root prior logits
  noisy_logits = policies._get_logits_from_probs(
      policies._add_dirichlet_noise(
          dirichlet_rng_key,
          jax.nn.softmax(root.prior_logits),
          dirichlet_fraction=dirichlet_fraction,
          dirichlet_alpha=dirichlet_alpha))
  root = root.replace(
      prior_logits=policies._mask_invalid_actions(noisy_logits, invalid_actions))

  batch_size = jax.tree_util.tree_leaves(root.embedding)[0].shape[0]

  # Initialize is_decision_node flag
  is_decision_node = jnp.ones([batch_size], dtype=bool)

  # Create initial StochasticRecurrentState
  root_state = base.StochasticRecurrentState(
      state_embedding=root.embedding,
      afterstate_embedding=None,
      is_decision_node=is_decision_node)

  # Adjust root prior_logits to include chance logits (set to -inf for root)
  num_actions = root.prior_logits.shape[-1]
  num_chance_outcomes = num_samples
  extended_prior_logits = jnp.concatenate([
      root.prior_logits,
      jnp.full([batch_size, num_chance_outcomes], fill_value=-jnp.inf)
  ], axis=-1)
  root = root.replace(
      prior_logits=extended_prior_logits,
      embedding=root_state)

  # Define the recurrent function with diffusion model for chance nodes
  recurrent_fn = _make_diffusion_recurrent_fn(
      decision_node_fn=decision_recurrent_fn,
      chance_node_fn=chance_recurrent_fn,
      num_actions=num_actions,
      num_chance_outcomes=num_chance_outcomes,
      num_samples=num_samples)

  # Define action selection functions
  interior_decision_node_selection_fn = functools.partial(
      action_selection.muzero_action_selection,
      pb_c_base=pb_c_base,
      pb_c_init=pb_c_init,
      qtransform=qtransform)

  interior_action_selection_fn = policies._make_stochastic_action_selection_fn(
      interior_decision_node_selection_fn,
      num_actions=num_actions)
  root_action_selection_fn = functools.partial(
      interior_action_selection_fn, depth=0)

  # Run the search
  search_tree = search.search(
      params=params,
      rng_key=search_rng_key,
      root=root,
      recurrent_fn=recurrent_fn,
      root_action_selection_fn=root_action_selection_fn,
      interior_action_selection_fn=interior_action_selection_fn,
      num_simulations=num_simulations,
      max_depth=max_depth,
      invalid_actions=invalid_actions,
      loop_fn=loop_fn)

  # Sampling the proposed action proportionally to the visit counts (decision nodes)
  search_tree = policies._mask_tree(search_tree, num_actions, 'decision')
  summary = search_tree.summary()
  action_weights = summary.visit_probs
  action_logits = policies._apply_temperature(
      policies._get_logits_from_probs(action_weights), temperature)
  action = jax.random.categorical(rng_key, action_logits)
  return base.PolicyOutput(
      action=action, action_weights=action_weights, search_tree=search_tree)


def _make_diffusion_recurrent_fn(
    decision_node_fn: base.DecisionRecurrentFn,
    chance_node_fn: base.ChanceRecurrentFn,
    num_actions: int,
    num_samples: int,
) -> base.RecurrentFn:
  """Creates a RecurrentFn integrating diffusion model."""

  def diffusion_recurrent_fn(
      params: base.Params,
      rng: chex.PRNGKey,
      action_or_chance: base.Action,  # [B]
      state: dmz_types.DiffusionRecurrentState
  ) -> Tuple[base.RecurrentFnOutput, dmz_types.DiffusionRecurrentState]:
    batch_size = jax.tree_util.tree_leaves(state.state_embedding)[0].shape[0]

    # Internally we assume that there are `A' = A + C` "actions";
    # action_or_chance can take on values in `{0, 1, ..., A' - 1}`,.
    # To interpret it as an action we can leave it as is:
    action = action_or_chance - 0
    # To interpret it as a chance outcome we subtract num_actions:
    chance_outcome = action_or_chance - num_actions

    decision_output, next_state_embeddings = decision_node_fn(
        params, rng, action, state.state_embedding)
    # Outputs from DecisionRecurrentFunction produce chance logits with
    # dim `C`, to respect our internal convention that there are `A' = A + C`
    # "actions" we pad with `A` dummy logits which are ultimately ignored:
    # see `_mask_tree`.
    output_if_decision_node = base.RecurrentFnOutput(
        prior_logits=jnp.concatenate([
            jnp.full([batch_size, num_actions], fill_value=-jnp.inf),
            decision_output.chance_logits], axis=-1),
        value=decision_output.afterstate_value,
        reward=jnp.zeros_like(decision_output.afterstate_value),
        discount=jnp.ones_like(decision_output.afterstate_value))

    chance_output, state_embedding = chance_node_fn(params, rng, chance_outcome,
                                                    state.next_state_embeddings)
    # Outputs from ChanceRecurrentFunction produce action logits with dim `A`,
    # to respect our internal convention that there are `A' = A + C` "actions"
    # we pad with `C` dummy logits which are ultimately ignored: see
    # `_mask_tree`.
    output_if_chance_node = base.RecurrentFnOutput(
        prior_logits=jnp.concatenate([
            chance_output.action_logits,
            jnp.full([batch_size, num_samples], fill_value=-jnp.inf)
            ], axis=-1),
        value=chance_output.value,
        reward=chance_output.reward,
        discount=chance_output.discount)

    new_state = dmz_types.DiffusionRecurrentState(
        state_embedding=state_embedding,
        next_state_embeddings=next_state_embeddings,
        is_decision_node=jnp.logical_not(state.is_decision_node))

    def _broadcast_where(decision_leaf, chance_leaf):
      extra_dims = [1] * (len(decision_leaf.shape) - 1)
      expanded_is_decision = jnp.reshape(state.is_decision_node,
                                         [-1] + extra_dims)
      return jnp.where(
          # ensure state.is_decision node has appropriate shape.
          expanded_is_decision,
          decision_leaf, chance_leaf)

    output = jax.tree_map(_broadcast_where,
                          output_if_decision_node,
                          output_if_chance_node)
    return output, new_state

  return diffusion_recurrent_fn