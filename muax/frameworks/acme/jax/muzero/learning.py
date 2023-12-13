
"""Learner for the MuZero agent."""

from typing import Iterator

import acme 
import reverb
import optax 
from acme.jax import types as jax_types
from muax.frameworks.acme.jax.muzero import networks as mz_networks


class MZLearner(acme.Learner):
    def __init__(
        self,
        networks: mz_networks.MZNetworks,
        iterator: Iterator[reverb.ReplaySample],
        optimizer: optax.GradientTransformation,
        random_key: jax_types.PRNGKey
    ):
        pass 