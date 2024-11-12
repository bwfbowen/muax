from abc import ABC, abstractmethod
import mctx
import jax
from mctx import qtransform_by_parent_and_siblings, qtransform_completed_by_mix_value


class Policy(ABC):
    @abstractmethod
    def __call__(self, params, rng_key, root, recurrent_fn=None, decision_recurrent_fn=None, chance_recurrent_fn=None, **kwargs):
        pass


class MuZeroPolicy(Policy):
    def __init__(self):
        self.policy_func = mctx.muzero_policy

    def __call__(self, params, rng_key, root, recurrent_fn, **kwargs):
        return self.policy_func(
            params, rng_key, root, recurrent_fn,
            num_simulations=kwargs.get('num_simulations', 5),
            temperature=kwargs.get('temperature', 1.),
            invalid_actions=kwargs.get('invalid_actions'),
            max_depth=kwargs.get('max_depth'),
            loop_fn=kwargs.get('loop_fn', jax.lax.fori_loop),
            qtransform=kwargs.get('qtransform', qtransform_by_parent_and_siblings),
            dirichlet_fraction=kwargs.get('dirichlet_fraction', 0.25),
            dirichlet_alpha=kwargs.get('dirichlet_alpha', 0.3),
            pb_c_init=kwargs.get('pb_c_init', 1.25),
            pb_c_base=kwargs.get('pb_c_base', 19652)
        )


class GumbelMuZeroPolicy(Policy):
    def __init__(self):
        self.policy_func = mctx.gumbel_muzero_policy

    def __call__(self, params, rng_key, root, recurrent_fn=None, decision_recurrent_fn=None, chance_recurrent_fn=None, **kwargs):
        return self.policy_func(
            params, rng_key, root, recurrent_fn,
            num_simulations=kwargs.get('num_simulations', 5),
            invalid_actions=kwargs.get('invalid_actions'),
            max_depth=kwargs.get('max_depth'),
            loop_fn=kwargs.get('loop_fn', jax.lax.fori_loop),
            qtransform=kwargs.get('qtransform', qtransform_completed_by_mix_value),
            max_num_considered_actions=kwargs.get('max_num_considered_actions', 16),
            gumbel_scale=kwargs.get('gumbel_scale', 1)
        )


class StochasticMuZeroPolicy(Policy):
    def __init__(self):
        self.policy_func = mctx.stochastic_muzero_policy

    def __call__(self, params, rng_key, root, recurrent_fn=None, decision_recurrent_fn=None, chance_recurrent_fn=None, **kwargs):
        return self.policy_func(
            params, rng_key, root, decision_recurrent_fn, chance_recurrent_fn,
            num_simulations=kwargs.get('num_simulations', 5),
            temperature=kwargs.get('temperature', 1.),
            invalid_actions=kwargs.get('invalid_actions'),
            max_depth=kwargs.get('max_depth'),
            loop_fn=kwargs.get('loop_fn', jax.lax.fori_loop),
            qtransform=kwargs.get('qtransform', qtransform_by_parent_and_siblings),
            dirichlet_fraction=kwargs.get('dirichlet_fraction', 0.25),
            dirichlet_alpha=kwargs.get('dirichlet_alpha', 0.3),
            pb_c_init=kwargs.get('pb_c_init', 1.25),
            pb_c_base=kwargs.get('pb_c_base', 19652)
        )
