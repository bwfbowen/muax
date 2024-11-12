from typing import NamedTuple, Optional, Union, Tuple, Sequence

import dataclasses
import numpy as np
import jax 
from jax import numpy as jnp
import haiku as hk
import haiku.initializers as hk_init 
from acme import specs
from acme.jax import networks as networks_lib
from acme.jax import utils 
from muax.frameworks.acme.jax.diffusion_muzero import types
from muax.frameworks.acme.jax.diffusion_muzero import utils as dmz_utils

import jax.nn as nn


class DMZNetworkParams(NamedTuple):
    encoder: Optional[hk.Params] = None
    representation: Optional[hk.Params] = None
    prediction: Optional[hk.Params] = None
    decision: Optional[hk.Params] = None
    chance: Optional[hk.Params] = None
    temperature: float = 1.

@dataclasses.dataclass
class DMZNetworks:
    """Networks and pure functions for Stochastic MuZero agent."""
    encoder_network: networks_lib.FeedForwardNetwork
    representation_network: networks_lib.FeedForwardNetwork
    prediction_network: networks_lib.FeedForwardNetwork
    decision_network: networks_lib.FeedForwardNetwork
    chance_network: networks_lib.FeedForwardNetwork

def init_params(
    networks: DMZNetworks,
    spec: specs.EnvironmentSpec,
    random_key: types.RNGKey,
    max_chance_size: int = 32,
    add_batch_dim: bool = True, 
) -> DMZNetworkParams:
    rng_keys = jax.random.split(random_key, 5)
    observations, actions = utils.zeros_like((spec.observations, spec.actions))
    dummy_chance_outcome = jnp.zeros(max_chance_size, dtype=jnp.int32)
    if add_batch_dim:
        observations, actions, dummy_chance_outcome = utils.add_batch_dim((observations, actions, dummy_chance_outcome))
    
    encoder_params = networks.encoder_network.init(rng_keys[0], observations)
    representation_params = networks.representation_network.init(rng_keys[1], observations)
    state_embedding = networks.representation_network.apply(representation_params, observations)
    prediction_params = networks.prediction_network.init(rng_keys[2], state_embedding)
    decision_params = networks.decision_network.init(rng_keys[2], state_embedding, actions)
    afterstate_embedding = networks.decision_network.apply(decision_params, state_embedding, actions)
    chance_params = networks.chance_network.init(rng_keys[3], dummy_chance_outcome, afterstate_embedding)

    params = DMZNetworkParams(
        encoder=encoder_params,
        representation=representation_params,
        prediction=prediction_params,
        decision=decision_params,
        chance=chance_params
    )
    
    return params

def make_mlp_networks(
    environment_spec: specs.EnvironmentSpec,
    embedding_dim: int = 64,
    max_chance_code_book_size: int = 32,
    *,
    encoder_layer_sizes: Sequence[int] = (256, 256),  
    representation_layer_sizes: Sequence[int] = (256, 256, 256),
    prediction_layer_sizes: Sequence[int] = (256, 256, 256),
    decision_layer_sizes: Sequence[int] = (256, 256, 256),
    chance_layer_sizes: Sequence[int] = (256, 256, 256),
    full_support_size: int = 51,
    vmin: float = -150., # Used by 2hot critics
    vmax: float = 150., # Used by 2hot critics
) -> DMZNetworks:
    num_dimensions = environment_spec.actions.num_values

    # Add Encoder class
    class Encoder(hk.Module):
        def __init__(self, max_chance_code_book_size: int, name='encoder'):
            super().__init__(name=name)
            self.encoder_func = hk.Sequential([
                networks_lib.LayerNormMLP(encoder_layer_sizes, activate_final=True),
                hk.Linear(max_chance_code_book_size, w_init=hk_init.TruncatedNormal(1))
                ])

        def __call__(self, obs):
            return self.encoder_func(obs)

    def encoder_fn(observation: types.Observation) -> types.Embedding:
        return Encoder(max_chance_code_book_size)(observation)

    class Representation(hk.Module):
        def __init__(self, embedding_dim: int, name='representation'):
            super().__init__(name=name)
            self.repr_func = hk.Sequential([
                networks_lib.LayerNormMLP(representation_layer_sizes, activate_final=True),
                hk.Linear(embedding_dim, w_init=hk_init.TruncatedNormal(1))
                ])

        def __call__(self, obs):
            s = self.repr_func(obs)
            s = dmz_utils.min_max_normalize(s)
            return s 
    
    def representation_fn(observation: types.Observation) -> types.Embedding:
        return Representation(embedding_dim)(observation)
    
    class Decision(hk.Module):
        def __init__(self, embedding_dim, num_actions, max_chance_code_book_size, name='decision'):
            super().__init__(name=name)
            self.afterstate_func = hk.Sequential([
                networks_lib.LayerNormMLP(decision_layer_sizes, activate_final=True),
                hk.Linear(embedding_dim)
            ])
            self.chance_logits_func = hk.Sequential([
                networks_lib.LayerNormMLP(decision_layer_sizes, activate_final=True),
                hk.Linear(max_chance_code_book_size)
            ])
            self.afterstate_value_func = hk.Sequential([
                networks_lib.LayerNormMLP(decision_layer_sizes, activate_final=True),
                networks_lib.CategoricalValueHead(num_bins=full_support_size, vmin=vmin, vmax=vmax)
            ])
            self.cat_func = jax.jit(lambda s, a: 
                                    jnp.concatenate([s, jax.nn.one_hot(a, num_actions)],
                                                    axis=1)
                                    )
        
        def __call__(self, s, a):
            sa = self.cat_func(s, a)
            afterstate_embedding = self.afterstate_func(sa)
            chance_logits = self.chance_logits_func(afterstate_embedding)
            afterstate_value = self.afterstate_value_func(afterstate_embedding)
            return afterstate_embedding, chance_logits, afterstate_value
    
    def decision_fn(embedding: types.Embedding, action: types.Action):
        return Decision(embedding_dim, num_dimensions, max_chance_code_book_size)(embedding, action)
    
    class Chance(hk.Module):
        def __init__(self, embedding_dim, value_support_size, name='chance'):
            super().__init__(name=name)
            self.state_func = hk.Sequential([
                networks_lib.LayerNormMLP(chance_layer_sizes, activate_final=True),
                hk.Linear(embedding_dim)
            ])
            
            self.reward_func = hk.Sequential([
                networks_lib.LayerNormMLP(chance_layer_sizes, activate_final=True),
                networks_lib.CategoricalValueHead(num_bins=value_support_size, vmin=vmin, vmax=vmax)
            ])

        def __call__(self, afterstate_embedding, chance_outcome):
            input_tensor = jnp.concatenate([afterstate_embedding, chance_outcome], axis=-1)
            state_embedding = self.state_func(input_tensor)
            state_embedding = dmz_utils.min_max_normalize(state_embedding)
            reward = self.reward_func(state_embedding)
            return state_embedding, reward 
        
    class Prediction(hk.Module):
        def __init__(self, num_actions, full_support_size, name='prediction'):
            super().__init__(name=name)
            self.value_func = hk.Sequential([
                networks_lib.LayerNormMLP(prediction_layer_sizes, activate_final=True),
                networks_lib.CategoricalValueHead(num_bins=full_support_size, vmin=vmin, vmax=vmax)
            ])
            self.action_logits_func = hk.Sequential([
                networks_lib.LayerNormMLP(prediction_layer_sizes, activate_final=True),
                hk.Linear(num_actions)
            ])

        def __call__(self, state_embedding):
            value = self.value_func(state_embedding)
            action_logits = self.action_logits_func(state_embedding)
            return value, action_logits
    
    def prediction_fn(state_embedding):
        return Prediction(num_dimensions, full_support_size)(state_embedding)

    def chance_fn(afterstate_embedding, chance_outcome):
        state_embedding, reward = Chance(embedding_dim, full_support_size)(afterstate_embedding, chance_outcome)
        value, action_logits = prediction_fn(state_embedding)
        return state_embedding, action_logits, value, reward
    
    networks = DMZNetworks(
        encoder_network=hk.without_apply_rng(hk.transform(encoder_fn)),
        representation_network=hk.without_apply_rng(hk.transform(representation_fn)),
        prediction_network=hk.without_apply_rng(hk.transform(prediction_fn)),
        decision_network=hk.without_apply_rng(hk.transform(decision_fn)),
        chance_network=hk.without_apply_rng(hk.transform(chance_fn)),
    )    

    return networks
