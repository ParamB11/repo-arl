import sys

import numpy as np

def compute_avg_return(environment, policy, num_episodes=10):
    returns = []
    
    for _ in range(num_episodes):

        time_step = environment.reset()
        episode_return = 0.0

        while not time_step.is_last().numpy():
            if hasattr(policy, 'action'):
                action_step = policy.action(time_step)
            elif hasattr(policy, 'predict'):
                action = policy.predict(time_step.observation, deterministic=True)
                action_step = tuple_to_action_step(action)
            else:
                raise ValueError("Policy in argument policy is of invalid format.")
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        
        returns.append(episode_return.numpy()[0])
    
    avg_return = np.mean(returns)
    stddev = np.std(returns)
    return avg_return, stddev

def compute_avg_return_uposi(osi_env, up_policy, num_episodes=10):
    returns = []
    for _ in range(num_episodes):
        obs, _ = osi_env.reset()
        episode_return = 0.0
        done = False
        while not done:
            action, _ = up_policy.predict(obs, deterministic=True)
            obs, reward, done, _ = osi_env.step(action)
            episode_return += reward
        returns.append(episode_return)
    avg_return = np.mean(returns)
    stddev = np.std(returns)
    return avg_return, stddev

def compute_returns_uposi(osi_env, up_policy, num_episodes=10):
    returns = []
    for _ in range(num_episodes):
        obs, _ = osi_env.reset()
        episode_return = 0.0
        done = False
        while not done:
            action, _ = up_policy.predict(obs, deterministic=True)
            obs, reward, done, _ = osi_env.step(action)
            episode_return += reward
        returns.append(episode_return)
    return returns