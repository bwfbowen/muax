from typing import Generic, Iterator, List, Optional, Dict

import functools
import jax 
import optax 
import chex 
import tensorflow as tf 
from absl import logging
from acme import core, specs
from acme.adders import base
from acme.agents.jax import builders
from acme.agents.jax import actors 
from acme.agents.jax import actor_core as actor_core_lib
from acme.agents.jax.builders import Networks, Policy, Sample
from acme.adders import reverb as adders
from acme.datasets import reverb as datasets
from acme.datasets import image_augmentation as img_aug
from acme.utils import counting, loggers
from muax.frameworks.acme.jax.stochastic_muzero import networks as smz_networks
from muax.frameworks.acme.jax.stochastic_muzero import config as smz_config
from muax.frameworks.acme.jax.stochastic_muzero import acting
from muax.frameworks.acme.jax.stochastic_muzero import learning
from acme.jax import networks as networks_lib, observation_stacking as obs_stacking
from acme.jax import types as jax_types
from acme.jax import variable_utils
from acme.jax import utils 
import reverb
import tree 


_POLICY_KEY = 'policy'
_ONLINE_TABLE_NAME = 'online_table'


class SMZBuilder(builders.ActorLearnerBuilder):

    def __init__(
        self, 
        config: smz_config.StochasticMuZeroConfig,
        extra_spec: Dict,
    ):
        self.config = config 
        self.extra_spec = extra_spec
        self._adjust_sequence_length = (self.config.sequence_length 
                                        + self.config.num_stacked_observations
                                        - 1)
    
    def make_policy(
        self, 
        networks: smz_networks.SMZNetworks, 
        environment_spec: specs.EnvironmentSpec, 
        evaluation: bool = False
    ) -> actor_core_lib.ActorCore:
        actor_core = acting.make_actor_core(
            networks=networks,
            config=self.config,
            environment_spec=environment_spec,
            evaluation=evaluation
        )
        if self.config.num_stacked_observations > 1:
            # actor-side observation stacking
            actor_core = obs_stacking.wrap_actor_core(
                actor_core=actor_core,
                observation_spec=environment_spec.observations,
                num_stacked_observations=self.config.num_stacked_observations)
        
        return actor_core
    
    def make_actor(
        self, 
        random_key: jax_types.PRNGKey, 
        policy: actor_core_lib.ActorCore, 
        environment_spec: specs.EnvironmentSpec, 
        variable_source: Optional[core.VariableSource] = None, 
        adder: Optional[base.Adder] = None
    ) -> core.Actor:
        
        del environment_spec 
        variable_client = variable_utils.VariableClient(
            client=variable_source,
            key=_POLICY_KEY,
            update_period=self.config.variable_update_period
        )

        return actors.GenericActor(
            actor=policy,
            random_key=random_key,
            variable_client=variable_client,
            adder=adder,
            backend=self.config.variable_client_backend)
    
    def make_adder(
            self, 
            replay_client: reverb.Client, 
            environment_spec: Optional[specs.EnvironmentSpec], 
            policy: Optional[actor_core_lib.ActorCore]
        ) -> Optional[base.Adder]:
        del environment_spec
        priority_fns = {}
        if self.config.offline_fraction > 0:
            priority_fns[adders.DEFAULT_PRIORITY_TABLE] = self.config.get_offline_priority_fn(
                discount=self.config.discount,
                num_bootstrapping=self.config.num_bootstrapping,
                sequence_length=self.config.sequence_length,
                bootstrapping_lambda=self.config.bootstrapping_lambda,
                priority_alpha=self.config.priority_alpha)
        if self.config.offline_fraction < 1:
            priority_fns[_ONLINE_TABLE_NAME] = self.config.get_online_priority_fn(
                discount=self.config.discount,
                num_bootstrapping=self.config.num_bootstrapping,
                sequence_length=self.config.sequence_length,
                bootstrapping_lambda=self.config.bootstrapping_lambda,
                priority_alpha=self.config.priority_alpha)
        if self.config.adder_type == adders.SequenceAdder:
            
            return adders.SequenceAdder(
                client=replay_client,
                sequence_length=self._adjust_sequence_length,
                period=self.config.period,
                end_of_episode_behavior=adders.EndBehavior.WRITE,
                priority_fns=priority_fns)
        elif self.config.adder_type == adders.EpisodeAdder: 
            return adders.EpisodeAdder(
                client=replay_client,
                max_sequence_length=self.config.sequence_length,
                priority_fns=priority_fns)

    def make_replay_tables(
            self, 
            environment_spec: specs.EnvironmentSpec, 
            policy: actor_core_lib.ActorCore
        ) -> List[reverb.Table]:
        dummy_actor_state = policy.init(jax.random.PRNGKey(0))
        extras_spec = policy.get_extras(dummy_actor_state)

        if self.config.adder_type == adders.SequenceAdder:
            signature = adders.SequenceAdder.signature(
                environment_spec=environment_spec,
                extras_spec=extras_spec,
                sequence_length=self._adjust_sequence_length)
        elif self.config.adder_type == adders.EpisodeAdder:
            signature = adders.EpisodeAdder.signature(
                environment_spec=environment_spec,
                extras_spec=extras_spec,
                sequence_length=self.config.sequence_length)
        
        if self.config.samples_per_insert:
            samples_per_insert_tolerance = (
                self.config.samples_per_insert_tolerance_ratio
                * self.config.samples_per_insert)
            error_buffer = self.config.min_replay_size * samples_per_insert_tolerance
            limiter = reverb.rate_limiters.SampleToInsertRatio(
                min_size_to_sample=self.config.min_replay_size,
                samples_per_insert=self.config.samples_per_insert,
                error_buffer=max(error_buffer, 2 * self.config.samples_per_insert))
        else:
            limiter = reverb.rate_limiters.MinSize(self.config.min_replay_size)
        
        offline_extensions, online_extensions = [], []

        tables = []
        if self.config.offline_fraction > 0:
            offline_table = reverb.Table(
                name=adders.DEFAULT_PRIORITY_TABLE,
                sampler=reverb.selectors.Uniform(),
                remover=reverb.selectors.Fifo(),
                max_size=self.config.max_replay_size, 
                rate_limiter=limiter,
                extensions=offline_extensions,
                signature=signature)
            tables.append(offline_table)
            logging.info(f'''
                Creating off-policy replay buffer with 
                fraction {self.config.offline_fraction} 
                of batch {self.config.batch_size}''')
        if self.config.offline_fraction < 1:
            online_table = reverb.Table.queue(
                name=_ONLINE_TABLE_NAME,
                max_size=self.config.online_queue_capacity,
                extensions=online_extensions,
                signature=signature)
            tables.append(online_table)
            logging.info(f"""
                Creating online replay queue with 
                fraction {1. - self.config.offline_fraction} 
                of batch {self.config.batch_size}""")
        return tables
       
    def make_dataset_iterator(
            self,
            replay_client: reverb.Client
        ) -> Iterator[reverb.ReplaySample]:

        if self.config.num_stacked_observations > 1:
            maybe_stack_observations = functools.partial(
                obs_stacking.stack_reverb_observation,
                stack_size=self.config.num_stacked_observations)
        else:
            maybe_stack_observations = None
        
        dataset = datasets.make_reverb_dataset(
            server_address=replay_client.server_address,
            batch_size=self.config.batch_size // jax.device_count(), 
            table={
                adders.DEFAULT_PRIORITY_TABLE: self.config.offline_fraction,
                _ONLINE_TABLE_NAME: 1. - self.config.offline_fraction
            },
            num_parallel_calls=self.config.dataset_num_parallel_calls,
            max_in_flight_samples_per_worker=(
                self.config.gradient_steps_per_learner_step
                * self.config.batch_size
                // jax.device_count()), 
            postprocess=maybe_stack_observations)
        
        if self.config.observation_transform:
            transform = img_aug.make_transform(
                observation_transform=self.config.observation_transform, 
                transform_next_observation=False, 
            )
            dataset = dataset.map(
                transform, 
                num_parallel_calls=self.config.dataset_num_parallel_calls,
                deterministic=False)
        
        if self.config.gradient_steps_per_learner_step > 1:
            dataset = dataset.batch(self.config.gradient_steps_per_learner_step, drop_remainder=True)
            batch_flatten = lambda t: tf.reshape(t, [-1] + t.shape[2:].as_list())
            dataset = dataset.map(lambda x: tree.map_structure(batch_flatten, x))

        return utils.multi_device_put(dataset.as_numpy_iterator(), jax.local_devices())

    def make_learner(
            self,
            random_key: jax_types.PRNGKey, 
            networks: smz_networks.SMZNetworks, 
            dataset: Iterator[reverb.ReplaySample], 
            logger_fn: loggers.LoggerFactory, 
            environment_spec: specs.EnvironmentSpec, 
            replay_client: Optional[reverb.Client] = None,
            counter: Optional[counting.Counter] = None
    ) -> core.Learner:
        
        if self.config.batch_size % jax.device_count() > 0:
            raise ValueError(
                'Batch size must divide evenly by the number of learner devices.'
                f' Passed a batch size of {self.config.batch_size} and the number of'
                f' available learner devices is {jax.device_count()}. Specifically,'
                f' devices: {jax.devices()}.')
        
        agent_environment_spec = environment_spec
        if self.config.num_stacked_observations > 1:
            agent_environment_spec = obs_stacking.get_adjusted_environment_spec(
            environment_spec, self.config.num_stacked_observations)
        
        logger = logger_fn(
            'learner',
            steps_key=counter.get_steps_key() if counter else 'learner_steps')
        
        optimizer = optax.adamw(
            self.config.learning_rate,
            b1=self.config.adam_b1,
            b2=self.config.adam_b2,
            weight_decay=self.config.weight_decay)
        
        with chex.fake_pmap_and_jit(not self.config.jit_learner,
                                    not self.config.jit_learner):
            learner = learning.SMZLearner(
                config=self.config,
                networks=networks,
                environment_spec=agent_environment_spec,
                iterator=dataset,
                optimizer=optimizer,
                logger=logger,
                random_key=random_key,
                replay_client=replay_client,
            )
        return learner
    
    