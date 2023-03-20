import numpy as np 
import jax 


def test(model, env, key, num_simulations, num_test_episodes=10, random_seed=None):
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
