from typing import Callable, Tuple, Optional, Union

import dataclasses
import jax 
import mctx
from acme.adders import reverb 
from acme.adders.reverb import base as reverb_base
from muax.frameworks.acme.jax.muzero import types
from muax.frameworks.acme.jax.muzero import utils
from muax.utils import support_to_scalar, scalar_to_support


@dataclasses.dataclass
class MuZeroConfig:
    """MuZero config"""
    num_stacked_observations: int = 1
    num_steps: int = int(1e4)
    num_simulations: int = 200
    invalid_actions: types.Action = None
    max_depth: int = None
    loop_fn: Callable = jax.lax.fori_loop
    qtransform: Callable = mctx.qtransform_by_parent_and_siblings
    temperature_fn: Callable = utils.temperature_fn
    pb_c_init: float = 1.25
    pb_c_base: float = 19652
    dirichlet_fraction: float = 0.25
    dirichlet_alpha: float = 0.3
    discount: float = 0.99  # Discount rate applied to value per timestep.
    support_size: int = 20
    support_to_scalar_fn: Callable = support_to_scalar
    scalar_to_support_fn: Callable = scalar_to_support
    policy_jit_static_argnames: Tuple[str] = ('temperature', 'recurrent_fn', 'num_simulations', 'loop_fn', 'qtransform', 'max_depth', 'dirichlet_fraction', 'dirichlet_alpha', 'pb_c_init', 'pb_c_base')

    batch_size: int = 256
    gradient_steps_per_learner_step: int = 16
    learning_rate: Union[float, Callable[[int], float]] = 1e-4

    variable_update_period: int = 1000
    variable_client_backend: str = 'cpu'

    offline_fraction: float = 0.5

    adder_type: reverb_base.ReverbAdder = reverb.SequenceAdder
    sequence_length: int = 10
    period: int = 1
    # priority based on value prediction error
    get_offline_priority_fn: Callable = utils.get_priority_fn_with_replaybuffer
    get_online_priority_fn: Callable = utils.get_priority_fn_with_replaybuffer

    samples_per_insert: Optional[float] = 32.
    samples_per_insert_tolerance_ratio: Optional[float] = 0.1
    min_replay_size: int = int(1e3)
    max_replay_size: int = int(1e6)

    online_queue_capacity: int = int(1e3)

    dataset_num_parallel_calls: int = max(16, 4 * jax.local_device_count())
    observation_transform: Optional[Callable[[types.Observation], types.Observation]] = None