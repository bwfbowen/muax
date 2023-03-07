import os
from functools import partial
import jax 
from jax import numpy as jnp
import optax 
# import coax
import gymnasium as gym 

from .wrappers import TrainMonitor
from .episode_tracer import PNStep
from .replay_buffer import Trajectory, TrajectoryReplayBuffer


def _temperature_fn(max_training_steps, training_steps):
  if training_steps < 0.5 * max_training_steps:
      return 1.0
  elif training_steps < 0.75 * max_training_steps:
      return 0.5
  else:
      return 0.25


def fit(model, env_id, 
          tracer=PNStep(50, 0.997, 0.5), 
          buffer=TrajectoryReplayBuffer(500),
          max_episodes: int = 1000, 
          max_training_steps: int = 10000,
          save_every_n_steps: int = 1000,
          num_simulations: int = 50,
          k_steps: int = 10,
          buffer_warm_up: int = 128,
          num_trajectory: int = 32,
          sample_per_trajectory: int = 10,
          name: str = None,
          tensorboard_dir=None, 
          save_path=None,
          random_seed: int = 42,
          temperature_fn=_temperature_fn,
          log_all_metrics=False,
          ):
  if name is None:
    name = env_id 
  if tensorboard_dir is None:
    tensorboard_dir = '.'
  if save_path is None:
    save_path = 'model_params'
  env = gym.make(env_id, render_mode='rgb_array')
  # env.seed(random_seed)
  env = TrainMonitor(env, name=name, 
    tensorboard_dir=os.path.join(tensorboard_dir, name),
    log_all_metrics=log_all_metrics)

  sample_input = jnp.expand_dims(jnp.zeros(env.observation_space.shape), axis=0)
  key = jax.random.PRNGKey(random_seed)
  key, subkey = jax.random.split(key)
  model.init(subkey, sample_input) 

  training_step = 0

  for ep in range(max_episodes):
    obs, info = env.reset(seed=random_seed)    
    trajectory = Trajectory()
    temperature = temperature_fn(max_training_steps=max_training_steps, training_steps=training_step)
    for t in range(env.spec.max_episode_steps):
      key, subkey = jax.random.split(key)
      a, pi, v = model.act(subkey, obs, 
                           with_pi=True, 
                           with_value=True, 
                           obs_from_batch=False,
                           num_simulations=num_simulations,
                           temperature=temperature)
      obs_next, r, done, truncated, info = env.step(a)
      if truncated:
        r = 1 / (1 - tracer.gamma)
      tracer.add(obs, a, r, done or truncated, v=v, pi=pi)
      while tracer:
        trans = tracer.pop()
        trajectory.add(trans)
        env.record_metrics({'v': trans.v, 'Rn': trans.Rn})
      if done or truncated:
        break 
      obs = obs_next 
    trajectory.finalize()
    buffer.add(trajectory, trajectory.batched_transitions.w.mean())
    if len(buffer) >= buffer_warm_up:
      for _ in range(t):
        transition_batch = buffer.sample(num_trajectory=num_trajectory,
                                         sample_per_trajectory=sample_per_trajectory,
                                         k_steps=k_steps)
        loss_metric = model.update(transition_batch)
        training_step += 1
        env.record_metrics(loss_metric)
        
        if training_step % save_every_n_steps == 0:
          model.save(save_path)
        if training_step >= max_training_steps:
          return model
    env.record_metrics({'training_step': training_step})

  return model
