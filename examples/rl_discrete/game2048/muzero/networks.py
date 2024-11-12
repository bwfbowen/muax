from typing import NamedTuple, Optional, Union, Tuple, Sequence

import dataclasses
import numpy as np
import jax 
from jax import numpy as jnp
import haiku as hk
import haiku.initializers as hk_init 
from acme.jax import networks as networks_lib
from acme import specs 
from acme.jax import utils 
from muax.frameworks.acme.jax.muzero import types
from muax.frameworks.acme.jax.muzero import utils as mz_utils


class MZNetworkParams(NamedTuple):
    representation: Optional[hk.Params] = None
    prediction: Optional[hk.Params] = None
    dynamic: Optional[hk.Params] = None
    temperature: float = 1.


@dataclasses.dataclass
class MZNetworks:
    """Networks and pure functions for MuZero agent."""
    representation_network: networks_lib.FeedForwardNetwork
    prediction_network: networks_lib.FeedForwardNetwork
    dynamic_network: networks_lib.FeedForwardNetwork


def init_params(
    networks: MZNetworks,
    spec: specs.EnvironmentSpec,
    random_key: types.RNGKey,
    add_batch_dim: bool = True, 
) -> MZNetworkParams:
    rng_keys = jax.random.split(random_key, 3)
    observations, actions = utils.zeros_like((spec.observations, spec.actions))
    if add_batch_dim:
        observations, actions = utils.add_batch_dim((observations, actions))
    representation_params = networks.representation_network.init(rng_keys[0], observations['board'])
    embeddings = networks.representation_network.apply(representation_params, observations['board'])
    prediction_params = networks.prediction_network.init(rng_keys[1], embeddings)
    dynamic_params = networks.dynamic_network.init(rng_keys[2], embeddings, actions)

    params = MZNetworkParams(
        representation=representation_params,
        prediction=prediction_params,
        dynamic=dynamic_params,
        temperature=np.array(1.))
    
    return params


def make_mlp_networks(
    environment_spec: specs.EnvironmentSpec,
    embedding_dim: int = 64,
    *,
    representation_layer_sizes: Sequence[int] = (256, 256, 256),
    prediction_layer_sizes: Sequence[int] = (256, 256, 256),
    dynamic_layer_sizes: Sequence[int] = (256, 256, 256),
    full_support_size: int = 51,
    vmin: float = -150., # Used by 2hot critics
    vmax: float = 150., # Used by 2hot critics
) -> MZNetworks:
    num_dimensions = environment_spec.actions.num_values
    class Representation(hk.Module):
        def __init__(self, embedding_dim: int, name='representation'):
            super().__init__(name=name)
            self.repr_func = hk.Sequential([
                networks_lib.LayerNormMLP(representation_layer_sizes, activate_final=True),
                hk.Linear(embedding_dim, w_init=hk_init.TruncatedNormal(1))
                ])

        def __call__(self, obs):
            s = self.repr_func(obs)
            s = mz_utils.min_max_normalize(s)
            return s 
    
    def representation_fn(observation: types.Observation) -> types.Embedding:
        return Representation(embedding_dim)(observation)
    
    class Prediction(hk.Module):
        def __init__(self, num_actions, value_support_size, name='prediction'):
            super().__init__(name=name)        
            
            self.v_func = hk.Sequential([
                networks_lib.LayerNormMLP(prediction_layer_sizes, activate_final=True),
                networks_lib.CategoricalCriticHead(num_bins=value_support_size, vmin=vmin, vmax=vmax)
            ])
            self.pi_func = hk.Sequential([
                networks_lib.LayerNormMLP(prediction_layer_sizes, activate_final=True),
                hk.Linear(num_actions)
            ])
        
        def __call__(self, s):
            v = self.v_func(s)
            logits = self.pi_func(s)
            return v, logits
        
    def prediction_fn(embedding: types.Embedding):
        return Prediction(num_dimensions, full_support_size)(embedding)
    
    class Dynamic(hk.Module):
        def __init__(self, embedding_dim, num_actions, reward_support_size, name='dynamic'):
            super().__init__(name=name)
            
            self.ns_func = hk.Sequential([
                networks_lib.LayerNormMLP(dynamic_layer_sizes, activate_final=True),
                hk.Linear(embedding_dim, w_init=hk_init.TruncatedNormal(1))
            ])
            self.r_func = hk.Sequential([
                networks_lib.LayerNormMLP(dynamic_layer_sizes, activate_final=True),
                networks_lib.CategoricalCriticHead(num_bins=reward_support_size, vmin=vmin, vmax=vmax)
            ])
            self.cat_func = jax.jit(lambda s, a: 
                                    jnp.concatenate([s, jax.nn.one_hot(a, num_actions)],
                                                    axis=1)
                                    )
        
        def __call__(self, s, a):
            sa = self.cat_func(s, a)
            r = self.r_func(sa)
            ns = self.ns_func(sa)
            ns = mz_utils.min_max_normalize(ns)
            return r, ns
    
    def dynamic_fn(embedding: types.Embedding, action: types.Action):
        return Dynamic(embedding_dim, num_dimensions, full_support_size)(embedding, action)
    
    networks = MZNetworks(
        representation_network=hk.without_apply_rng(hk.transform(representation_fn)),
        prediction_network=hk.without_apply_rng(hk.transform(prediction_fn)),
        dynamic_network=hk.without_apply_rng(hk.transform(dynamic_fn))
    )
    return networks

def make_fully_connect_resnet_networks(
    environment_spec: specs.EnvironmentSpec,
    embedding_dim: int = 256,
    num_blocks: int = 10,
    full_support_size: int = 601,
    vmin: float = 0.,
    vmax: float = 600.,
) -> MZNetworks:
    num_actions = environment_spec.actions.num_values

    class ResNetBlock(hk.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = hk.Linear(embedding_dim)
            self.linear2 = hk.Linear(embedding_dim)
            self.projection = hk.Linear(embedding_dim)

        def __call__(self, x):
            residual = self.projection(x)
            x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
            x = jax.nn.relu(x)
            x = self.linear1(x)
            x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
            x = jax.nn.relu(x)
            x = self.linear2(x)
            return x + residual

    class ResNetTower(hk.Module):
        def __init__(self, num_blocks: int):
            super().__init__()
            self.blocks = [ResNetBlock() for _ in range(num_blocks)]

        def __call__(self, x):
            for block in self.blocks:
                x = block(x)
            return x

    def representation_network(obs):
        x = hk.Linear(embedding_dim)(obs)
        x = ResNetTower(num_blocks)(x)
        x = mz_utils.min_max_normalize(x)
        return x

    def prediction_network(state):
        tower = ResNetTower(num_blocks)(state)
        policy_head = hk.Sequential([
            hk.LayerNorm(axis=-1, create_scale=True, create_offset=True),
            jax.nn.relu,
            hk.Linear(num_actions)
        ])
        value_head = hk.Sequential([
            hk.LayerNorm(axis=-1, create_scale=True, create_offset=True),
            jax.nn.relu,
            networks_lib.CategoricalCriticHead(num_bins=full_support_size, vmin=vmin, vmax=vmax)
        ])
        return value_head(tower), policy_head(tower)

    def dynamics_network(state, action):
        action_embed = jax.nn.one_hot(action, num_actions)
        x = jnp.concatenate([state, action_embed], axis=-1)
        x = ResNetTower(num_blocks)(x)
        reward_head = hk.Sequential([
            hk.LayerNorm(axis=-1, create_scale=True, create_offset=True),
            jax.nn.relu,
            networks_lib.CategoricalCriticHead(num_bins=full_support_size, vmin=vmin, vmax=vmax)
        ])
        next_state_head = hk.Sequential([
            hk.LayerNorm(axis=-1, create_scale=True, create_offset=True),
            jax.nn.relu,
            hk.Linear(embedding_dim)
        ])
        reward = reward_head(x)
        next_state = next_state_head(x)
        return reward, next_state

    return MZNetworks(
        representation_network=hk.without_apply_rng(hk.transform(representation_network)),
        prediction_network=hk.without_apply_rng(hk.transform(prediction_network)),
        dynamic_network=hk.without_apply_rng(hk.transform(dynamics_network))
    )