from typing import NamedTuple, Optional, Union, Tuple

import dataclasses
import haiku as hk
from acme.jax import networks as networks_lib


class MZNetworkParams(NamedTuple):
    representation: Optional[hk.Params] = None
    prediction: Optional[hk.Params] = None
    dynamic: Optional[hk.Params] = None


@dataclasses.dataclass
class MZNetworks:
    """Networks and pure functions for MuZero agent."""
    representation_network: networks_lib.FeedForwardNetwork
    prediction_network: networks_lib.FeedForwardNetwork
    dynamic_network: networks_lib.FeedForwardNetwork
