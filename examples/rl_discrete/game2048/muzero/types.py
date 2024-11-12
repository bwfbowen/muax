from jax import numpy as jnp
from acme import types

Observation = types.NestedArray
Embedding = types.NestedArray
Action = jnp.ndarray

RNGKey = jnp.ndarray