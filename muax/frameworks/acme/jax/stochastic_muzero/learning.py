
"""Learner for the Stochastic MuZero agent."""

from typing import Iterator, List, Optional, NamedTuple, Sequence, Dict, Any, Tuple 

import functools
from absl import logging
import time 
import numpy as np
import acme 
from acme import specs 
from acme.adders import reverb as adders
from acme.types import NestedArray 
import reverb
import jax 
from jax import numpy as jnp 
import optax 
import rlax 
from acme.jax import types as jax_types
from acme.jax import utils 
from acme.jax import networks as network_lib
from acme.utils import counting
from acme.utils import loggers
from acme.utils import async_utils
from muax.frameworks.acme.jax.stochastic_muzero import networks as smz_networks
from muax.frameworks.acme.jax.stochastic_muzero import config as smz_config
from muax.frameworks.acme.jax.stochastic_muzero import utils as smz_utils
from muax.frameworks.acme.jax.stochastic_muzero import RAW_VALUES_KEY, POLICY_PROBS_KEY
import tree

_PMAP_AXIS_NAME = 'data'


class TrainingState(NamedTuple):
    params: smz_networks.SMZNetworkParams
    opt_state: optax.OptState
    step: int
    random_key: jax_types.PRNGKey


class ReverbUpdate(NamedTuple):
  """Tuple for updating reverb priority information."""
  keys: jnp.ndarray
  priorities: jnp.ndarray


class SMZLearner(acme.Learner):

    _state: TrainingState

    def __init__(
        self,
        config: smz_config.StochasticMuZeroConfig,
        networks: smz_networks.SMZNetworks,
        environment_spec: specs.EnvironmentSpec,
        iterator: Iterator[reverb.ReplaySample],
        optimizer: optax.GradientTransformation,
        random_key: jax_types.PRNGKey,
        replay_client: Optional[reverb.Client] = None,
        counter: Optional[counting.Counter] = None,
        logger: Optional[loggers.Logger] = None,
        devices: Optional[Sequence[jax.Device]] = None,
    ):
        self._config = config 
        self._networks = networks
        self._counter = counter or counting.Counter()
        self._logger = logger
        self._iterator = iterator
        self._optimizer = optimizer 

        self._loss_l2_reg_scale = config.loss_l2_reg_scale
        self._vqvae_beta = config.vqvae_beta

        process_id = jax.process_index()
        local_devices = jax.local_devices()
        self._devices = devices or local_devices
        logging.info('Learner process id: %s. Devices passed: %s', process_id,
                    devices)
        logging.info('Learner process id: %s. Local devices from JAX API: %s',
                    process_id, local_devices)
        
        self._local_devices = [d for d in self._devices if d in local_devices]
        random_key, key = jax.random.split(random_key)
        network_params = smz_networks.init_params(
            self._networks,
            environment_spec,
            key,
            add_batch_dim=True,
        )
        opt_state = self._optimizer.init(network_params)
        state = TrainingState(
            params=network_params,
            opt_state=opt_state,
            step=0,
            random_key=random_key,
        )
        self._state = utils.replicate_in_all_devices(state, self._local_devices)

        # Log how many parameters the network has.
        sizes = tree.map_structure(jnp.size, network_params)._asdict()
        num_params_by_component_str = ' | '.join(
            [f'{key}: {sum(tree.flatten(size))}' for key, size in sizes.items()])
        logging.info('Number of params by network component: %s',
                    num_params_by_component_str)
        logging.info('Total number of params: %d',
                    sum(tree.flatten(sizes.values())))
        
        # TODO: 
        self._priority_name = 'updated_priority'
        # Update replay priorities
        def update_priorities(reverb_update: ReverbUpdate) -> None:
            if replay_client is None:
                return
            keys, priorities = tree.map_structure(
                # Fetch array and combine device and batch dimensions.
                lambda x: utils.fetch_devicearray(x).reshape((-1,) + x.shape[2:]),
                (reverb_update.keys, reverb_update.priorities))
            replay_client.mutate_priorities(
                table=adders.DEFAULT_PRIORITY_TABLE,
                updates=dict(zip(keys, priorities)))
        self._replay_client = replay_client
        self._async_priority_updater = async_utils.AsyncExecutor(update_priorities)

        def postprocess_aux(aux):
            return {k: jnp.mean(v) if k != self._priority_name else v for k, v in aux.items()}

        self._current_step = 0
        self._timestamp = None 
        self._gammas = np.power(self._config.discount, np.arange(1, self._config.sequence_length + 1))  # discount at 1,...,T

        gradient_steps = utils.process_multiple_batches(
            self._gradient_step, 
            self._config.gradient_steps_per_learner_step,
            postprocess_aux=postprocess_aux)
        self._gradient_steps = jax.pmap(gradient_steps, axis_name=_PMAP_AXIS_NAME, devices=self._devices)

    def _gradient_step(
            self,
            state: TrainingState,
            batch: adders.Step,
    ) -> Tuple[TrainingState, Dict[str, Any]]:
        keys = jax.random.split(state.random_key, num=2)
        random_key, keys = keys[0], keys[1]
        loss_vfn = jax.vmap(self._loss_fn, in_axes=(None, 0))
        safe_mean = lambda x: jnp.mean(x) if x is not None else x
        # loss_fn = lambda *a, **k: tree.map_structure(safe_mean, loss_vfn(*a, **k))
        def loss_fn(*a, **k):
            loss, loss_log_dict = loss_vfn(*a, **k) 
            return safe_mean(loss), loss_log_dict
        loss_and_grad = jax.value_and_grad(loss_fn, has_aux=True)
        (_, loss_log_dict), all_gradients = loss_and_grad(state.params, batch)
        gradients = jax.lax.pmean(all_gradients, _PMAP_AXIS_NAME)
        gradients_norm = optax.global_norm(gradients)
        updates, opt_state = self._optimizer.update(
            gradients, state.opt_state, state.params)
        params = optax.apply_updates(state.params, updates)
        step = state.step + 1
        new_params = smz_networks.SMZNetworkParams(
            representation=params.representation,
            prediction=params.prediction,
            decision=params.decision,
            chance=params.chance,
            temperature=self._config.temperature_fn(self._config.num_steps, self._current_step)
        )
        
        new_state = TrainingState(
            params=new_params,
            step=step,
            opt_state=opt_state,
            random_key=random_key
        )

        metrics = {f'loss/{k}' if k != self._priority_name else k: safe_mean(v) if k != self._priority_name else v for k, v in loss_log_dict.items()}
        metrics.update({'opt/grad_norm': gradients_norm})

        return new_state, metrics

    def _loss_fn(
            self,
            params: smz_networks.SMZNetworkParams,
            batch: adders.Step,
    ) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        batch = utils.add_batch_dim(batch) 
        c = self._loss_l2_reg_scale
        vqvae_beta = self._vqvae_beta
        value_target_scalar = self._compute_value_target(batch)
        reward = rlax.transform_to_2hot(
            scalar=batch.reward, 
            min_value=self._config.vmin, 
            max_value=self._config.vmax, 
            num_bins=self._config.full_support_size)
        value_target = rlax.transform_to_2hot(
            scalar=value_target_scalar,
            min_value=self._config.vmin,
            max_value=self._config.vmax,
            num_bins=self._config.full_support_size)
        policy = batch.extras[POLICY_PROBS_KEY]
        
        s = self._networks.representation_network.apply(params.representation, batch.observation[:, 0])
        
        init_v, init_policy_logits = self._networks.prediction_network.apply(params.prediction, s)
        loss_init_v = jnp.mean(
            optax.softmax_cross_entropy(
                init_v.logits, jax.lax.stop_gradient(value_target[:, 0]))
        )
        loss_init_pi = jnp.mean(
            optax.softmax_cross_entropy(
                init_policy_logits, jax.lax.stop_gradient(policy[:, 0]))
        )
        loss = loss_init_v + loss_init_pi

        value_scalar = rlax.transform_from_2hot(
                    jax.nn.softmax(init_v.logits), 
                    init_v.values.min(), 
                    init_v.values.max(), 
                    self._config.full_support_size)
        updated_priority = jnp.mean(jnp.abs(value_scalar - value_target_scalar[:, 0]) ** self._config.priority_alpha)

        def unroll_func(i, loss_s):
            loss, s = loss_s 
            # chance code
            # Apply straight-through
            code_encoded = self._networks.encoder_network.apply(params.encoder, batch.observation[:, i])
            code_quantized = smz_networks.select_closest_code(code_encoded)
            code = code_encoded + jax.lax.stop_gradient(code_quantized - code_encoded)
            # Appendix G, scale the gradient at the start of the dynamics function by 1/2 
            s = smz_utils.scale_gradient(s, 0.5)
            ae, c_logit, av = self._networks.decision_network.apply(params.decision, s, batch.action[:, i - 1])
            ns, a_logits, v, r = self._networks.chance_network.apply(params.chance, ae, code)
            
            # losses: reward
            loss_r = jnp.mean(
                optax.softmax_cross_entropy(
                    r.logits, jax.lax.stop_gradient(reward[:, i - 1]))
            )
            # losses: value 
            loss_v = jnp.mean(
                optax.softmax_cross_entropy(
                    v.logits, jax.lax.stop_gradient(value_target[:, i]))
            )
            # losses: policy
            loss_pi = jnp.mean(
                optax.softmax_cross_entropy(
                    a_logits, policy[:, i])
            )
            # losses: afterstate chance outcome
            loss_chance_outcome = jnp.mean(
                optax.softmax_cross_entropy(
                    logits=c_logit, 
                    labels=jax.lax.stop_gradient(code)
                )
            )
            # losses: afterstate value 
            loss_after_value = jnp.mean(
                optax.softmax_cross_entropy(
                    av.logits, jax.lax.stop_gradient(value_target[:, i - 1])
                )
            )
            # losses: vq-vae commitment cost
            loss_vqvae = jnp.mean(jnp.square(code_encoded - code))
            
            loss += loss_r + loss_v + loss_pi + loss_chance_outcome + loss_after_value + (loss_vqvae * vqvae_beta)
            loss_s = (loss, ns)
            return loss_s 
        loss, _ = jax.lax.fori_loop(
            1, 
            self._config.num_unroll_steps, 
            unroll_func, 
            (loss, s))
        # Appendix G Training: "irrespective of how many steps we unroll for"
        loss /= self._config.num_unroll_steps
        # L2 regulariser
        l2_regulariser = 0.5 * sum(
            jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(params))
        loss += c * jnp.sum(l2_regulariser)
        loss_logging_dict = {
            'loss': loss,
            'root_policy_loss': loss_init_pi,
            'root_value_loss': loss_init_v,
            self._priority_name: updated_priority,}
        return loss, loss_logging_dict
    
    def _compute_value_target(self, batch: adders.Step) -> jnp.ndarray:
        reward = batch.reward
        value = batch.extras[RAW_VALUES_KEY]
        n_step_return_fn = functools.partial(smz_utils.n_step_bootstrapped_returns, 
                                discount_t=self._gammas, 
                                n=self._config.num_bootstrapping,
                                lambda_t=self._config.bootstrapping_lambda,
                                stop_target_gradients=True)
        bootstrapped = jax.vmap(n_step_return_fn)(r_t=reward[..., :-1], v_t=value[..., 1:])
        return bootstrapped

    def step(self):
        """Perform one learner step."""
        with jax.profiler.StepTraceAnnotation('step', step_num=self._current_step):
            sample = next(self._iterator)
            reverb_keys = sample.info.key
            minibatch = adders.Step(*sample.data)
            # minibatch is of shape (device?, batch * gradient steps, sequence length, data dimensions)
            self._state, metrics = self._gradient_steps(self._state, minibatch)
            self._current_step, metrics = smz_utils.get_from_first_device(
                (self._state.step, metrics))
            
            timestamp = time.time()
            elapsed_time = timestamp - self._timestamp if self._timestamp else 0
            self._timestamp = timestamp

            if self._replay_client and metrics is not None:
                reverb_update = ReverbUpdate(reverb_keys, metrics[self._priority_name])
                self._async_priority_updater.put(reverb_update)

            # Increment counts and record the current time
            counts = self._counter.increment(
                steps=self._config.gradient_steps_per_learner_step, 
                walltime=elapsed_time)
            
            del metrics[self._priority_name]
            metrics['steps_per_second'] = (
                self._config.gradient_steps_per_learner_step / elapsed_time if elapsed_time > 0
                else 0.)
            
            if self._logger:
                self._logger.write({**metrics, **counts})

        return super().step()
    
    def get_variables(self, names: List[str]) -> network_lib.Params:
        params = smz_utils.get_from_first_device(self._state.params)
        variables = {
            'policy': params,
        }
        return [variables[name] for name in names]
    
    def save(self) -> TrainingState:
        return jax.tree_map(smz_utils.get_from_first_device, self._state)
    
    def restore(self, state: TrainingState):
        self._state = utils.replicate_in_all_devices(state, self._local_devices)