from jax import numpy as jnp
from acme import types
import chex 

Observation = types.NestedArray
Embedding = types.NestedArray
Action = jnp.ndarray

RNGKey = jnp.ndarray

@chex.dataclass(frozen=True)
class DiffusionRecurrentState:
  """Wrapper that enables different treatment of decision and chance nodes.

  Addtionally handle next_state_embeddings to StochasticRecurrentState.

  Attributes:
    state_embedding: `[B ...]` an optionally meaningful state embedding.
    next_state_embeddings: `[B ...]` an optionally meaningful afterstate
      embedding.
    is_decision_node: `[B]` whether the node is a decision or chance node. 
  """
  state_embedding: chex.ArrayTree  # [B, ...]
  next_state_embeddings: chex.ArrayTree  # [B, num_samples, ...]
  is_decision_node: chex.Array  # [B]