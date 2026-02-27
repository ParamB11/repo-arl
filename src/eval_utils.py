import numpy as np

def compute_returns(environment, policy, num_episodes=10, deterministic=True):
    returns = []
    observation, _ = environment.reset()
    episode_return = 0.0
    episode_count = 0
    while episode_count < num_episodes:
        action = policy.predict(observation, deterministic=deterministic)[0]
        new_observation, reward, term, trunc, _ = environment.step(action)
        episode_return += reward
        if term or trunc:
            episode_count += 1
            new_observation, _ = environment.reset()
            returns.append(episode_return)
            episode_return = 0.0
        observation = new_observation
    return returns

def compute_avg_return(environment, policy, num_episodes=10, deterministic=True):
    returns = []
    observation, _ = environment.reset()
    episode_return = 0.0
    episode_count = 0
    while episode_count < num_episodes:
        action = policy.predict(observation, deterministic=deterministic)[0]
        new_observation, reward, term, trunc, _ = environment.step(action)
        episode_return += reward
        if term or trunc:
            episode_count += 1
            new_observation, _ = environment.reset()
            returns.append(episode_return)
            episode_return = 0.0
        observation = new_observation
    
    avg_return = np.mean(returns)
    # print("returns = ", returns)
    # print("np.mean(returns) = ", np.mean(returns))
    stddev = np.std(returns)
    # print("stddev = ", stddev)
    return avg_return, stddev