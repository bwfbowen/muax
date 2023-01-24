import os
from functools import partial
import jax 
from jax import numpy as jnp
import optax 
import coax
import gym

from .episode_tracer import PNStep
from .replay_buffer import Trajectory, TrajectoryReplayBuffer

def fit(model, env_id, 
          tracer=PNStep(50, 0.997, 0.5), 
          buffer=TrajectoryReplayBuffer(500),
          max_episodes: int = 1000, 
          save_every_n_eps: int = 50,
          num_simulations: int = 50,
          k_steps: int = 10,
          buffer_warm_up: int = 128,
          num_trajectory: int = 32,
          sample_per_trajectory: int = 10,
          name: str = None,
          tensorboard_dir=None, 
          save_path=None,
          random_seed: int = 42):
  if name is None:
    name = env_id 
  if tensorboard_dir is None:
    tensorboard_dir = '.'
  if save_path is None:
    save_path = 'model_params'
  env = gym.make(env_id, render_mode='rgb_array')
  env = coax.wrappers.TrainMonitor(env, name=name, tensorboard_dir=os.path.join(tensorboard_dir, name))

  sample_input = jnp.expand_dims(jnp.zeros(env.observation_space.shape), axis=0)
  key = jax.random.PRNGKey(random_seed)
  key, subkey = jax.random.split(key)
  model.init(subkey, sample_input) 

  for ep in range(max_episodes):
    obs, info = env.reset()
    trajectory = Trajectory()
    for t in range(env.spec.max_episode_steps):
      key, subkey = jax.random.split(key)
      a, pi, v = model.act(subkey, obs, 
                           with_pi=True, 
                           with_value=True, 
                           obs_from_batch=False,
                           num_simulations=num_simulations)
      obs_next, r, done, truncated, info = env.step(a)
      if truncated:
        r = 1 / (1 - tracer.gamma)
      tracer.add(obs, a, r, done or truncated, v=v, pi=pi)
      while tracer:
        trajectory.add(tracer.pop())
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
        metrics = model.update(transition_batch)
        env.record_metrics(metrics)
    if ep % save_every_n_eps == 0:
      model.save(save_path)
  return model
    