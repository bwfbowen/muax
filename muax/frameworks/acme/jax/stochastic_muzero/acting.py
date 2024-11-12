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
from muax.frameworks.acme.jax.stochastic_muzero import config as smz_config
from muax.frameworks.acme.jax.stochastic_muzero import networks as smz_networks
from muax.frameworks.acme.jax.stochastic_muzero import types


POLICY_PROBS_KEY = 'pi'
RAW_VALUES_KEY = 'value'


class ActorState(NamedTuple):
  key: jax_types.PRNGKey
  step: int
  value: Union[jnp.ndarray, Tuple[()]] = ()
  pi: Union[jnp.ndarray, Tuple[()]] = ()
      

def make_actor_core(networks: smz_networks.SMZNetworks,
                    config: smz_config.StochasticMuZeroConfig,
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
        networks: smz_networks.SMZNetworks,
        config: smz_config.StochasticMuZeroConfig) -> networks_lib.FeedForwardNetwork:
        """Returns a function that computes actions."""

        def get_root_inference(
            networks: smz_networks.SMZNetworks,
            config: smz_config.StochasticMuZeroConfig) -> networks_lib.FeedForwardNetwork:
            repr_fn = networks.representation_network.apply
            pred_fn = networks.prediction_network.apply

            def root_inference(params: smz_networks.SMZNetworkParams, key, obs):
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

        def get_decision_recurrent_inference(
            networks: smz_networks.SMZNetworks,
            config: smz_config.StochasticMuZeroConfig) -> mctx.RecurrentFn:
            decision_fn = networks.decision_network.apply

            def decision_recurrent_fn(params: smz_networks.SMZNetworkParams, key, action, embedding):
                afterstate_embedding, chance_logits, afterstate_value = decision_fn(params.decision, embedding, action)
                afterstate_value = rlax.transform_from_2hot(
                    jax.nn.softmax(afterstate_value.logits),
                    afterstate_value.values.min(),
                    afterstate_value.values.max(),
                    config.full_support_size).flatten()
                decision_output = mctx.DecisionRecurrentFnOutput(
                    chance_logits=chance_logits,
                    afterstate_value=afterstate_value
                )
                return decision_output, afterstate_embedding 

            return decision_recurrent_fn
        
        def get_chance_recurrent_inference(
            networks: smz_networks.SMZNetworks,
            config: smz_config.StochasticMuZeroConfig) -> mctx.RecurrentFn:
            chance_fn = networks.chance_network.apply

            def chance_recurrent_fn(params: smz_networks.SMZNetworkParams, key, chance_outcome, afterstate_embedding):
                next_embedding, action_logits, value_output, reward_output = chance_fn(params.chance, afterstate_embedding, chance_outcome)
                value = rlax.transform_from_2hot(
                    jax.nn.softmax(value_output.logits),
                    value_output.values.min(),
                    value_output.values.max(),
                    config.full_support_size).flatten()
                reward = rlax.transform_from_2hot(
                    jax.nn.softmax(reward_output.logits),
                    reward_output.values.min(),
                    reward_output.values.max(),
                    config.full_support_size).flatten()
                discount = jnp.ones_like(reward) * config.discount
                chance_output = mctx.ChanceRecurrentFnOutput(
                    action_logits=action_logits,
                    value=value,
                    reward=reward,
                    discount=discount
                )
                return chance_output, next_embedding
            
            return chance_recurrent_fn
        
        root_inference_fn = jax.jit(get_root_inference(networks, config))
        decision_recurrent_fn = jax.jit(get_decision_recurrent_inference(networks, config))
        chance_recurrent_fn = jax.jit(get_chance_recurrent_inference(networks, config))
        policy = jax.jit(mctx.stochastic_muzero_policy, static_argnames=config.policy_jit_static_argnames)

        def select_action_policy(params: smz_networks.SMZNetworkParams, 
                                observations: types.Observation, 
                                state: ActorState) -> Tuple[types.Action, ActorState]:
            next_key, key = jax.random.split(state.key, 2)
            observations = jax_utils.add_batch_dim(observations)
            root = root_inference_fn(params, key, observations) 
            plan_output = policy(params, key, root, decision_recurrent_fn, chance_recurrent_fn,
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