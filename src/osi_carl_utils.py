import gym, gymnasium
import numpy as np
import torch
from stable_baselines3 import PPO, DDPG
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.td3.policies import TD3Policy  # DDPG uses TD3Policy structure in SB3

def update_queues(obs_queue, act_queue, obs, act):
    # print('obs =', obs)
    obs_queue.appendleft(obs)
    act_queue.appendleft(act)
    obs_stacked = np.stack(obs_queue, axis=0)
    act_stacked = np.stack(act_queue, axis=0)
    updated_obs = np.concatenate((obs_stacked.flatten(), act_stacked.flatten()))
    return obs_queue, act_queue, updated_obs

def get_action_and_prob(policy, obs, env, noise_std):
    """
    Generates an action and calculates its probability/density mu_t.
    
    Args:
        policy: The SB3 policy (PPO or DDPG).
        obs: The current observation (numpy array).
        env: The gymnasium environment (for action space bounds).
        noise_std: Standard deviation for DDPG exploration noise.
        
    Returns:
        action (np.array): The action to take in the environment.
        mu (float): The probability pi(a|s).
    """
    # Check if the policy is DDPG (Deterministic)
    if isinstance(policy, DDPG):
        # --- DDPG Logic (Deterministic + Noise) ---
        # 1. Get deterministic action from the actor network
        # deterministic=True ensures we get the raw output of the actor
        mean_action, _ = policy.predict(obs, deterministic=True)
        
        # 2. Add Gaussian noise for exploration: a ~ N(mean_action, noise_std)
        noise = np.random.normal(0, noise_std, size=mean_action.shape)
        noisy_action = mean_action + noise
        
        # 3. Clip action to ensure it stays valid
        final_action = np.clip(noisy_action, env.action_space.low, env.action_space.high)
        
        # 4. Calculate mu_t (Probability Density)
        # PDF of Gaussian: f(x) = (1 / (sigma * sqrt(2pi))) * exp(-0.5 * ((x-mu)/sigma)^2)
        # We use the unclipped action (noisy_action) for the true density calculation 
        # to strictly follow the Gaussian distribution logic.
        density = (1 / (noise_std * np.sqrt(2 * np.pi))) * \
                  np.exp(-0.5 * ((noisy_action - mean_action) / noise_std)**2)
        
        # Product of densities across dimensions (assuming independent noise)
        mu = np.prod(density)
        
        return final_action, mu

    elif isinstance(policy, PPO):
        # --- PPO Logic (Stochastic) ---
        # 1. Convert observation to PyTorch tensor
        # obs_to_tensor handles the batch dimension automatically
        obs_tensor = policy.policy.obs_to_tensor(obs)[0]
        
        with torch.no_grad():
            # 2. Get the distribution from the policy
            distribution = policy.policy.get_distribution(obs_tensor)
            
            # 3. Sample an action
            action_tensor = distribution.get_actions(deterministic=False)
            
            # 4. Get the log probability of that specific action
            log_prob = distribution.log_prob(action_tensor)
            
            # 5. Convert to probability (mu) and numpy action
            mu = torch.exp(log_prob).item()
            final_action = action_tensor.cpu().numpy()
            
            # Clip for safety (though PPO usually outputs within bounds or tanh)
            if isinstance(env.action_space, (gym.spaces.box.Box, gymnasium.spaces.box.Box)):
                final_action = np.clip(final_action, env.action_space.low, env.action_space.high)
                
        return final_action, mu

    else:
        raise NotImplementedError("This function only supports PPO and DDPG policies.")


def get_action_probability(policy, obs, action, ddpg_noise_std=None):
    """
    Calculates the probability density of a specified action given an observation 
    and a Stable Baselines 3 policy.

    Args:
        policy: The SB3 policy model (e.g., model.policy).
        obs (np.ndarray): The observation/state (single or batch).
        action (np.ndarray): The action for which to calculate probability.
        ddpg_noise_std (float, optional): The standard deviation of the Gaussian noise 
                                          added to DDPG/TD3 actions. Required if policy is DDPG/TD3.

    Returns:
        np.ndarray: The probability density of the action(s).
    """
    # 1. Switch model to evaluation mode (prevents training updates/dropout)
    policy.set_training_mode(False)
    
    # 2. Preprocess observations using SB3's internal helper
    # This handles converting numpy arrays to PyTorch tensors and moving to the correct device (CPU/GPU)
    obs_tensor, _ = policy.obs_to_tensor(obs)
    
    # Handle action conversion to tensor
    if isinstance(action, np.ndarray):
        action_tensor = torch.as_tensor(action, device=policy.device, dtype=torch.float32)
    else:
        action_tensor = action

    # 3. Calculate Log Probability based on Policy Type
    with torch.no_grad():
        
        # --- CASE A: PPO (Stochastic Policy) ---
        # PPO uses ActorCriticPolicy which natively outputs a distribution
        if isinstance(policy, ActorCriticPolicy):
            # Get the probability distribution (e.g., DiagGaussian)
            distribution = policy.get_distribution(obs_tensor)
            
            # log_prob returns the log likelihood of the action
            log_prob = distribution.log_prob(action_tensor)
            
        # --- CASE B: DDPG / TD3 (Deterministic Policy) ---
        # DDPG typically uses TD3Policy or a custom Actor class.
        # We assume Action = Deterministic_Output + Gaussian_Noise
        elif isinstance(policy, TD3Policy) or hasattr(policy, "actor"):
            if ddpg_noise_std is None:
                raise ValueError("For DDPG/TD3 policies, 'ddpg_noise_std' must be provided to calculate probability.")
            
            # Get the deterministic action (mu) from the actor network
            # Note: DDPG policies scale the output to the action space limits (e.g., tanh)
            mu = policy.actor(obs_tensor)
            
            # Construct a Gaussian distribution centered at mu with the specified noise std
            # We assume independent noise for each action dimension (Isotropic Gaussian)
            distribution = torch.distributions.Normal(mu, ddpg_noise_std)
            
            # Calculate log probability
            # We sum across the last dimension (action_dim) to get the joint probability of the action vector
            log_prob = distribution.log_prob(action_tensor).sum(dim=-1)
            
        else:
            raise TypeError(f"Unsupported policy type: {type(policy)}")

    # 4. Convert Log Probability to Probability Density
    # PDF = exp(Log_Prob)
    prob_density = torch.exp(log_prob)
    
    return prob_density.cpu().numpy()
