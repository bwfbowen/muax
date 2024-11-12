import dm_env
import jumanji
import jumanji.wrappers
from jax import numpy as jnp 
from acme import wrappers
from acme import specs
from acme.wrappers import base


def binary_representation(obs):
    # Flatten the 4x4 board into a single vector of size 16
    board = obs.board
    # Convert each number to a binary representation of 31 bits
    board = jnp.unpackbits(board.astype(jnp.int32).view(jnp.uint8)).reshape(board.size, 32)[:, 1:]
    # Flatten to a total size of 496
    flattened_board = board.astype(jnp.float32).reshape(496,)
    invalid_actions = (~obs.action_mask).astype(jnp.float32).reshape(4,)
    return {'board': flattened_board, 'invalid_actions': invalid_actions}
    

class BinaryRepresentationWrapper(base.EnvironmentWrapper):
    def __init__(self, environment: dm_env.Environment):
        super().__init__(environment)

    def observation_spec(self):
        # base_obs_spec = self._environment.observation_spec()
        return {'board': specs.Array(shape=(496,), dtype=jnp.float32, name='observation'), 
                'invalid_actions': specs.Array(shape=(4,), dtype=jnp.float32)}

    def reset(self):
        timestep = self._environment.reset()
        timestep = timestep._replace(observation=binary_representation(timestep.observation))
        return timestep

    def step(self, action):
        timestep = self._environment.step(action)
        timestep = timestep._replace(observation=binary_representation(timestep.observation))
        return timestep


def make_jumanji_environment(level: str, seed: int = None) -> dm_env.Environment:
    env = jumanji.make(level)
    
    env_dm = jumanji.wrappers.JumanjiToDMEnvWrapper(env)
    
    wrapper_list = [
        BinaryRepresentationWrapper,
        wrappers.SinglePrecisionWrapper,
    ]
    
    return wrappers.wrap_all(env_dm, wrapper_list)