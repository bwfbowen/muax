from acme.adders.reverb import base as reverb_base
from acme.agents.jax import actor_core as actor_core_lib


def temperature_fn(max_training_steps, training_steps):
  r"""Determines the randomness for the action taken by the model"""
  if training_steps < 0.5 * max_training_steps:
      return 1.0
  elif training_steps < 0.75 * max_training_steps:
      return 0.5
  else:
      return 0.25


def get_priority_fn_with_reanalyse(policy: actor_core_lib.ActorCore) -> reverb_base.PriorityFn:
    
    def priority_fn(trajectory: reverb_base.PriorityFnInput):
        return 
    
    return  


def get_priority_fn_with_replaybuffer(policy: actor_core_lib.ActorCore) -> reverb_base.PriorityFn:
    
    def priority_fn(trajectory: reverb_base.PriorityFnInput):
        return 

    return  