import numpy as np 
import jax 


def test(model, env, key, num_simulations, num_test_episodes=10, random_seed=None):
    r"""Tests the model on the environment and calculates the average total reward.
    
    Parameters
    ----------
    model: An instance of `MuZero`

    env: A gym-like environment. The env should have `reset()`, `step()` methods and `spec.max_episode_steps` attribute.

    key: array. `jax.random.PRNGKey(random_seed)`

    num_simulations: int, positive integer. The number of simulations.

    num_test_episodes: int, positive integer. The number of episodes to test the model.

    random_seed: int. Set for environment

    Returns
    -------
    average_test_reward: float. The average total reward for `num_test_episodes` tests.
    """
    total_rewards = np.zeros(num_test_episodes)
    for episode in range(num_test_episodes):
        obs, info = env.reset(seed=random_seed)
        done = False
        episode_reward = 0
        for t in range(env.spec.max_episode_steps):
            key, subkey = jax.random.split(key)
            a = model.act(subkey, obs, 
                          with_pi=False, 
                          with_value=False, 
                          obs_from_batch=False,
                          num_simulations=num_simulations,
                          temperature=0.) # Use deterministic actions during testing
            obs_next, r, done, truncated, info = env.step(a)
            episode_reward += r
            if done or truncated:
                break 
            obs = obs_next 
        
        total_rewards[episode] = episode_reward

    average_test_reward = np.mean(total_rewards)
    return average_test_reward
