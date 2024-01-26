from typing import NamedTuple, Tuple, Mapping, Union
import numpy as np
import jax 
from jax import numpy as jnp 
import mctx 
import rlax 
from acme import specs 
from acme.jax import networks as networks_lib
from acme.agents.jax import actor_core as actor_core_lib
from acme.jax import types as jax_types
from acme.jax import utils as jax_utils
from muax.frameworks.acme.jax.muzero import config as mz_config
from muax.frameworks.acme.jax.muzero import networks as mz_networks
from muax.frameworks.acme.jax.muzero import types


POLICY_PROBS_KEY = 'pi'
RAW_VALUES_KEY = 'value'


class ActorState(NamedTuple):
  key: jax_types.PRNGKey
  step: int
  value: Union[jnp.ndarray, Tuple[()]] = ()
  pi: Union[jnp.ndarray, Tuple[()]] = ()
      

def make_actor_core(networks: mz_networks.MZNetworks,
                    config: mz_config.MuZeroConfig,
                    environment_spec: specs.EnvironmentSpec,
                    evaluation: bool = False,
                    with_pi: bool = True,
                    with_value: bool = True) -> actor_core_lib.ActorCore:
    
    def init(key: jax_types.PRNGKey) -> ActorState:
        next_key, key = jax.random.split(key, 2)

        return ActorState(
            key=next_key,
            step=0,
            value=np.zeros(shape=(), dtype=np.float32),
            pi=np.zeros(shape=(environment_spec.actions.num_values,), dtype=np.float32)
        )

    def get_extras(state: ActorState) -> Mapping[str, jnp.ndarray]:
        extras = {}
        if with_pi:
            extras[POLICY_PROBS_KEY] = state.pi 
        if with_value:
            extras[RAW_VALUES_KEY] = state.value
        return extras 

    def get_select_action_policy(
        networks: mz_networks.MZNetworks,
        config: mz_config.MuZeroConfig) -> networks_lib.FeedForwardNetwork:
        """Returns a function that computes actions."""

        def get_root_inference(
            networks: mz_networks.MZNetworks,
            config: mz_config.MuZeroConfig) -> networks_lib.FeedForwardNetwork:
            repr_fn = networks.representation_network.apply
            pred_fn = networks.prediction_network.apply

            def root_inference(params: mz_networks.MZNetworkParams, key, obs):
                embedding = repr_fn(params.representation, obs)
                value_output, logits = pred_fn(params.prediction, embedding)
                value = rlax.transform_from_2hot(
                    jax.nn.softmax(value_output.logits), 
                    value_output.values.min(), 
                    value_output.values.max(), 
                    config.full_support_size).flatten()
                root = mctx.RootFnOutput(
                    prior_logits=logits,
                    value=value,
                    embedding=embedding,
                )
                return root 
            return root_inference

        def get_recurrent_inference(
            networks: mz_networks.MZNetworks,
            config: mz_config.MuZeroConfig) -> mctx.RecurrentFn:
            dyna_fn = networks.dynamic_network.apply
            pred_fn = networks.prediction_network.apply

            def recurrent_inference(params: mz_networks.MZNetworkParams, key, action, embedding):
                reward_output, next_embedding = dyna_fn(params.dynamic, embedding, action)
                # Thanks to SaeedAnas
                value_output, logits = pred_fn(params.prediction, next_embedding)
                reward = rlax.transform_from_2hot(
                    jax.nn.softmax(reward_output.logits), 
                    reward_output.values.min(), 
                    reward_output.values.max(), 
                    config.full_support_size).flatten()
                value = rlax.transform_from_2hot(
                    jax.nn.softmax(value_output.logits), 
                    value_output.values.min(), 
                    value_output.values.max(), 
                    config.full_support_size).flatten()
                discount = jnp.ones_like(reward) * config.discount
                recurrent_output = mctx.RecurrentFnOutput(
                    reward=reward, 
                    discount=discount, 
                    prior_logits=logits, 
                    value=value,
                )
                return recurrent_output, next_embedding
            return recurrent_inference
        
        root_inference_fn = jax.jit(get_root_inference(networks, config))
        recurrent_inference_fn = jax.jit(get_recurrent_inference(networks, config))
        policy = jax.jit(mctx.muzero_policy, static_argnames=config.policy_jit_static_argnames)

        def select_action_policy(params: mz_networks.MZNetworkParams, 
                                observations: types.Observation, 
                                state: ActorState) -> Tuple[types.Action, ActorState]:
            next_key, key = jax.random.split(state.key, 2)
            observations = jax_utils.add_batch_dim(observations)
            root = root_inference_fn(params, key, observations) 
            plan_output = policy(params, key, root, recurrent_inference_fn,
                                num_simulations=config.num_simulations,
                                temperature=params.temperature if not evaluation else 0.,
                                invalid_actions=config.invalid_actions,
                                max_depth=config.max_depth, 
                                loop_fn=config.loop_fn,
                                qtransform=config.qtransform, 
                                dirichlet_fraction=config.dirichlet_fraction, 
                                dirichlet_alpha=config.dirichlet_alpha, 
                                pb_c_init=config.pb_c_init, 
                                pb_c_base=config.pb_c_base)
            
            return jax_utils.squeeze_batch_dim(plan_output.action), ActorState(
                key=next_key,
                step=state.step + 1,
                value=jax_utils.squeeze_batch_dim(root.value) if with_value else (),
                pi=jax_utils.squeeze_batch_dim(plan_output.action_weights) if with_pi else ())
            
        return select_action_policy
    
    return actor_core_lib.ActorCore(
        init=init, 
        select_action=get_select_action_policy(networks=networks, config=config),
        get_extras=get_extras
    )