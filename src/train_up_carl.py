import os
import sys
import time
import traceback
sys.path.append('./src')

from carl.context.selection import RoundRobinSelector, StaticSelector
from carl.envs import CARLAcrobot, CARLLunarLander, CARLMountainCar, CARLPendulum
import gym
import gymnasium
from gymnasium import spaces
from gymnasium.wrappers import FlattenObservation, FilterObservation, StepAPICompatibility
import numpy as np
from stable_baselines3 import DDPG, DQN, PPO, SAC, TD3
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.vec_env import SubprocVecEnv

from common_utils import SaveCallback
from custom_wrappers import ToGymActionSpace, ToGymObservationSpace, ToGymnasiumActionSpace
from envs.carl_brax import CARLBraxAnt, CARLBraxHalfcheetah
from yaml_functions import preprocess_hyperparams, read_hyperparams

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-carl_env_name', type=str, help='name of the CARL environment', default='CARLPendulum')
    parser.add_argument('-context_labels', '--context_labels', nargs='+', type=str, help='context labels', required=True)
    # parser.add_argument('-context_experts', '--context_experts', nargs='+', type=float, help='context of experts', required=True)
    # parser.add_argument('-rel_std', type=float, help='relative standard deviation for training', default=0.25)
    parser.add_argument('-rel_std', '--rel_std', nargs='+', type=float, help='relative standard deviation for training', default=[0.25])
    
    parser.add_argument('-alg', type=str, help='algorithm/agent type for the expert', default='ppo')
    # parser.add_argument('-num_iterations', type=int, help='number of training iterations', default=60)
    parser.add_argument('-savedir', type=str, help='directory where experts are saved', default='data/saved_models/') #change default dir
    parser.add_argument('-prefix', type=str, help='prefix of the saved models', required=True)
    # parser.add_argument('-multiplier', type=int, nargs='+', help='context multiplier for saving the models', default=0)
    parser.add_argument('-n_train_samples', type=int, help='number of contexts to sample for training', default=100)
    parser.add_argument('--load_contexts', action='store_true', help='bool var to load contexts') #default=False
    parser.add_argument('-contextdir', type=str, help='dir for storing/loading context data', default='context_data/')
    parser.add_argument('-exp_name', type=str, help='experiment name to be concatenated with model name', default='') # do not add '_' at the start
    
    parser.add_argument('-trainsteps', type=int, help='hp for training', default=0)
    parser.add_argument('-nenvs', type=int, help='hp for training', default=0)
    parser.add_argument('-device', type=str, help='device to use for training', default='cpu')
    parser.add_argument('-hp_env_id', type=str, help='env_id to be used for loading hyperparams', default='')

    args = parser.parse_args()
    carl_env_fn = eval(args.carl_env_name)
    context_labels = args.context_labels
    labels = np.array(context_labels)
    DEFAULT_CONTEXT = carl_env_fn.get_default_context()
    

    for key in context_labels: 
        if key not in DEFAULT_CONTEXT.keys():
            print(f"Warning: Ignoring invalid key {key}.")

    ordered_labels = []
    for key in DEFAULT_CONTEXT.keys():
        if key in context_labels:
            ordered_labels.append(key)

    if ordered_labels == context_labels:
        print('Input labels are in correct order.')
    else:
        print(
            'Input labels not in correct order.'
            'ordered_labels=', ordered_labels
             )

    context_mean = []
    for key in DEFAULT_CONTEXT.keys():
        if key in context_labels:
            context_mean.append(DEFAULT_CONTEXT[key])

    print(f"context_mean = {context_mean}")
    
    context_rel_std = args.rel_std
    if isinstance(context_rel_std, list):
        context_std = [abs(mean)*rel_std for mean, rel_std in zip(context_mean, context_rel_std)]
    elif isinstance(context_rel_std, float):
        context_std = [abs(mean)*context_rel_std for mean in context_mean]
    else:
        raise ValueError
    print("context_std = ", context_std)

    labels_str = ""
    for label in context_labels:
        labels_str = labels_str + "_" + label

    n_samples = args.n_train_samples
    if args.load_contexts:
        load_path = args.prefix + labels_str + "_contexts_up_" + str(n_samples) + ".npy"
        load_path = os.path.join(args.contextdir, load_path)
        print('contexts_up load_path={0}'.format(load_path))
        train_context_array = np.load(load_path)
        print('train_context_array.shape = ', train_context_array.shape)
    else:
        train_context_array = np.zeros((len(context_mean), n_samples))
        for i in range(len(context_mean)):
            train_context_array[i,:] = np.sign(context_mean[i])*abs(np.random.normal(context_mean[i], context_std[i], n_samples))
    
        save_path = args.prefix + labels_str + "_contexts_up_" + str(n_samples) + ".npy"
        save_path = os.path.join(args.contextdir, save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        print('contexts_up save_path={0}'.format(save_path))
        np.save(save_path, train_context_array)
        print('Saved context data points.')

    print("train_context_array = ", train_context_array[:, 0:5])
    train_context_dict = {}
    for i in range(n_samples):
        train_context_dict[i] = {key:value for key,value in zip(context_labels, train_context_array[:,i])}
        if i<5:
            print("train_context_dict[{0}] = {1}".format(i, train_context_dict[i]))

    print('len(train_context_dict) = ', len(train_context_dict))

    # if args.alg == "ppo":
    # Loading hyperparams from yml file
    config_path = 'hyperparams/' + args.alg + '.yml'
    if args.hp_env_id == '':
        gym_id = carl_env_fn.env_name
    else:
        gym_id = args.hp_env_id
    unprocessed_hyperparams = read_hyperparams(config_path, gym_id)
    print('unprocessed_hyperparams = ', unprocessed_hyperparams)
    
    if args.nenvs !=0:
        n_envs = args.nenvs
    else:
        n_envs = unprocessed_hyperparams.get("n_envs", 1)
    print('n_envs = ', n_envs)
    
    n_experts = 2*len(context_labels)+1 # No. of experts in CEP
    if args.trainsteps != 0:
        n_timesteps = args.trainsteps
    else:
        n_timesteps = int(n_experts*unprocessed_hyperparams["n_timesteps"])
    print('n_timesteps = ', n_timesteps)
    hyperparams, env_wrapper, callbacks, vec_env_wrapper = preprocess_hyperparams(unprocessed_hyperparams)
    print('hyperparams = ', hyperparams)
    print('env_wrapper = ', env_wrapper)
    print('callbacks = ', callbacks)
    print('vec_env_wrapper = ', vec_env_wrapper)
    policy_type = hyperparams['policy']
    del hyperparams['policy']
    noise_type = hyperparams.get('noise_type')

    if noise_type is not None:
        print(f'noise_type = {noise_type}')
        del hyperparams['noise_type']
        if noise_type == 'normal':
            noise_fn = NormalActionNoise
        elif noise_type == 'ornstein-uhlenbeck':
            noise_fn = OrnsteinUhlenbeckActionNoise
        else:
            raise ValueError(
                f'Found noise_type={noise_type} which is invalid'
            )
        n_actions = carl_env_fn().action_space.shape[-1]
        print('n_actions =', n_actions)
        noise_std = hyperparams.get('noise_std')
        del hyperparams['noise_std']
        action_noise = noise_fn(mean=np.zeros(n_actions), sigma=noise_std*np.ones(n_actions))
    else:
        action_noise = None
        print('No noise to be added.')
    def env_fn():
        supported_spaces = (spaces.box.Box, spaces.discrete.Discrete)
        tenv = carl_env_fn(contexts=train_context_dict, 
                           obs_context_features=list(train_context_dict[0].keys()), context_selector=RoundRobinSelector)
        if not isinstance(tenv.action_space, supported_spaces):
            tenv = ToGymnasiumActionSpace(tenv)
        return FlattenObservation(FilterObservation(tenv, filter_keys=["obs", "context"]))
    
    if n_envs != 1:
        if n_envs > 16:
            train_env = make_vec_env(env_fn, n_envs=n_envs)
        else:
            train_env = make_vec_env(env_fn, vec_env_cls=SubprocVecEnv, n_envs=n_envs)
    else:
        train_env = make_vec_env(env_fn, n_envs=n_envs)
    
    # eval_env = make_vec_env(env_fn, n_envs=1)
    eval_env = make_vec_env(env_fn, vec_env_cls=SubprocVecEnv, n_envs=10)
    
    policy_name = f"_{args.alg}"
    for label in context_labels:
        policy_name = policy_name + "_" + label
    # suffix1 = ""
    # if args.interval:
    #     policy_name = policy_name + "_inter"
    policy_name = args.prefix + policy_name + "_up"
    if args.exp_name != '':
        policy_name = f'{policy_name}_{args.exp_name}'
    print(f'policy_name={policy_name}')
    eval_freq = int(max(n_timesteps/100, 100)/n_envs)
    print(f'eval_freq={eval_freq}')
    save_callback = SaveCallback(save_path=args.savedir, name_prefix=policy_name, chckpt_bool=False, verbose=1)
    eval_callback = EvalCallback(eval_env, callback_on_new_best=save_callback, eval_freq=eval_freq,
                                 n_eval_episodes=n_samples, log_path="logs/")
    if args.alg == 'ddpg':
        up_agent = DDPG(policy_type, train_env, **hyperparams, action_noise=action_noise, verbose=1, device=args.device)
    elif args.alg == 'dqn':
        if action_noise is not None:
            print(f'Warning: Ignoring action_noise since this feature is not available. Found action_noise={action_noise}.')
        up_agent = DQN(policy_type, train_env, **hyperparams, verbose=1, device=args.device)
    elif args.alg == 'ppo':
        if action_noise is not None:
            print(f'Warning: Ignoring action_noise since this feature is not available. Found action_noise={action_noise}.')
        up_agent = PPO(policy_type, train_env, **hyperparams, verbose=1, device=args.device)
    elif args.alg == 'td3':
        up_agent = TD3(policy_type, train_env, **hyperparams, action_noise=action_noise, verbose=1, device=args.device)
    elif args.alg == 'sac':
        if action_noise is not None:
            print(f'Warning: Ignoring action_noise since this feature is not available. Found action_noise={action_noise}.')
        up_agent = SAC(policy_type, train_env, **hyperparams, verbose=1, device=args.device)
    up_agent.learn(total_timesteps=n_timesteps, callback=eval_callback)
    train_env.close()
    eval_env.close()

if __name__ == '__main__':
    try:
        global_start = time.time()
        main()
        global_end = time.time()
        print(f"Total execution time in minutes: {(global_end-global_start)/60}.")
    except:
        print(traceback.format_exc())

    print("Completed execution.")