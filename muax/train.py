import os
import time 
import jax 
from jax import numpy as jnp
# import coax
import gymnasium as gym 
from typing import Optional

from .wrappers import TrainMonitor
from .episode_tracer import PNStep
from .replay_buffer import Trajectory, TrajectoryReplayBuffer
from .test import test


def _temperature_fn(max_training_steps, training_steps):
  r"""Determines the randomness for the action taken by the model"""
  if training_steps < 0.5 * max_training_steps:
      return 1.0
  elif training_steps < 0.75 * max_training_steps:
      return 0.5
  else:
      return 0.25


def fit(model, 
        env_id: Optional[str] = None,
        env: Optional[gym.Env] = None,
        test_env: Optional[gym.Env] = None, 
        tracer=PNStep(50, 0.997, 0.5), 
        buffer=TrajectoryReplayBuffer(500),
        max_episodes: int = 1000, 
        test_interval: int = 10,
        num_test_episodes: int = 10,
        max_training_steps: int = 10000,
        save_every_n_epochs: int = 1,
        num_simulations: int = 50,
        k_steps: int = 10,
        buffer_warm_up: int = 128,
        num_trajectory: int = 32,
        sample_per_trajectory: int = 10,
        name: str = None,
        tensorboard_dir=None, 
        model_save_path=None,
        save_name=None,
        random_seed: int = 42,
        temperature_fn=_temperature_fn,
        log_all_metrics=False,
        ):
  r"""  Fits the model on the given `env_id` environment.
        
        Parameters
        ----------
        model: An instance of `MuZero`
        
        env_id: str, the environment id for `gym.make(env_id, render_mode='rgb_array')`

        env: gym.Env. The gym-style environment. Either `env_id` or `env` needs to be provided. 
        If `env` is provided, `test_env` must be provided as well.

        test_env: gym.Env. The gym-style environment for testing. If not provided, `gym.make(env_id, render_mode='rgb_array')` will be used.
        
        tracer: An instance of episode tracer that inherents `BaseTracer`. 
        
        buffer: An instance of replay buffer that inherents `BaseReplayBuffer`

        max_episodes: int, positive integer. Maximum training episodes

        test_interval: int, positive integer. Tests the model every `test_interval` episodes.

        num_test_episodes: int, positive integer. The number of episodes to test the model.

        max_training_steps: int, positive integer. Maximum training steps.

        save_every_n_epochs: int, positive integer. Save the model every `save_every_n_epochs` episodes.

        num_simulations: int, positive int. The number of simulations.

        k_steps: int, positive int. The number of unrolled steps.

        buffer_warm_up: int, positive int. Collected trajectories until at least `buffer_warm_up` 
        trajectories are stored in the buffer.

        num_trajectory: int, positive int. Number of trajectories to sample from the buffer.

        sample_per_trajectory: int, positive int. The number of training samples from each of the trajectories.

        name: str. The name of the `TrainMonitor`, and the tensorboard is stored at 
        `tensorboard_dir=os.path.join(tensorboard_dir, name)`

        tensorboard_dir: str. The directory to store the tensorboard.

        model_save_path: str. The path to save the model parameters during training.

        save_name: str. The name of the file to store the model parameters.

        random_seed: int. random seed

        temperature_fn: Callable. Determines the randomness for the action taken by the model

        log_all_metrics: bool. If True, all metrics will be displayed by the `TrainMonitor`; 
        else only `T`, `avg_r`, `G`, `avg_G`, `t`, `dt` will be displayed.
        
        Returns
        -------
        model_path: str. A path to the model parameter that has the best performance during testing.

  """
  if env_id is None and env is None:
    raise ValueError("You must provide either `env_id` or `env`.")
    
  if env_id is not None and env is not None:
    raise ValueError("You can only provide either `env_id` or `env`, not both.")
  
  if env is None:
    env = gym.make(env_id, render_mode='rgb_array')
  
  if test_env is None:
    if env_id is not None:
      test_env = gym.make(env_id, render_mode='rgb_array')
    else:
      raise ValueError("You must provide `test_env` when using a custom `env`.")

  if name is None:
    name = env_id 
  if tensorboard_dir is None:
    tensorboard_dir = '.'
  if save_name is None:
    save_name = 'model_params'
  if model_save_path is None:
    timestr = time.strftime("%Y-%m-%d_%H-%M-%S")
    model_dir = os.path.join('models', timestr) 
  else:
    model_dir = model_save_path 

  sample_input = jnp.expand_dims(jnp.zeros(env.observation_space.shape), axis=0)
  key = jax.random.PRNGKey(random_seed)
  key, test_key, subkey = jax.random.split(key, num=3)
  model.init(subkey, sample_input) 

  training_step = 0
  best_test_G = -float('inf')
  model_path = None

  # buffer warm up
  print('buffer warm up stage...')
  while len(buffer) < buffer_warm_up:
    obs, info = env.reset()    
    tracer.reset()
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
      if done or truncated:
        break 
      obs = obs_next 
    trajectory.finalize()
    if len(trajectory) >= k_steps:
      buffer.add(trajectory, trajectory.batched_transitions.w.mean())
  
  print('start training...')
  # Apply TrainMonitor wrapper if env_id is provided
  if env_id is not None:
    env = TrainMonitor(env, tensorboard_dir=os.path.join(tensorboard_dir, name), log_all_metrics=log_all_metrics)
  
  for ep in range(max_episodes):
    obs, info = env.reset(seed=random_seed)   
    tracer.reset() 
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
    if len(trajectory) >= k_steps:
      buffer.add(trajectory, trajectory.batched_transitions.w.mean())

    train_loss = 0
    for _ in range(t):
      transition_batch = buffer.sample(num_trajectory=num_trajectory,
                                        sample_per_trajectory=sample_per_trajectory,
                                        k_steps=k_steps)
      loss_metric = model.update(transition_batch)
      train_loss += loss_metric['loss']
      training_step += 1
      
    train_loss /= t 
    env.record_metrics({'loss': train_loss})
    if ep % save_every_n_epochs == 0:
      model_folder_name = f'epoch_{ep:04d}_loss_{train_loss:.8f}'
      if not os.path.exists(os.path.join(model_dir, model_folder_name)):
        os.makedirs(os.path.join(model_dir, model_folder_name))
      cur_path = os.path.join(model_dir, model_folder_name, save_name) 
      model.save(cur_path)
      if not model_path:
        model_path = cur_path
    if training_step >= max_training_steps:
      return model_path
    env.record_metrics({'training_step': training_step})

    # Periodically test the model
    if ep % test_interval == 0:
      test_G = test(model, test_env, test_key, num_simulations=num_simulations, num_test_episodes=num_test_episodes)
      # test_env.record_metrics({'test_G': test_G})
      env.record_metrics({'test_G': test_G})
      if test_G >= best_test_G:
        best_test_G = test_G
        model_folder_name = f'epoch_{ep:04d}_test_G_{test_G:.8f}'
        if not os.path.exists(os.path.join(model_dir, model_folder_name)):
          os.makedirs(os.path.join(model_dir, model_folder_name))
        model_path = os.path.join(model_dir, model_folder_name, save_name)
        model.save(model_path)

  return model_path
