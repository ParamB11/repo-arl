'''
Code for generating and labelling the evaluation set. Note that label refers to 
the reward of Universal Policy with true context i.e. UP(c).
The purpose of the code is similar to gen_evalset_th.py. The difference being 
that in gen_evalset_th.py has the best expert as the optimal policy while in
this code UP(c) is the optimal policy.
Base code: gen_evalset_th.py
'''

import argparse
import io
import joblib
import multiprocessing
# import pickle
import os
import sys
import time
import traceback
import zipfile

from carl.context.selection import StaticSelector
from carl.envs import CARLAcrobot, CARLCartPole, CARLLunarLander, CARLMountainCar, CARLPendulum
import gymnasium
from gymnasium.wrappers import FlattenObservation, FilterObservation
import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.save_util import json_to_data
# import tensorflow as tf
import torch

from common_utils import init_carl, load_policy, load_policy_from_params
from custom_wrappers import ToGymnasiumActionSpace
from envs.carl_brax import CARLBraxAnt, CARLBraxHalfcheetah
from yaml_functions import preprocess_hyperparams, read_hyperparams

def eval_helper(carl_env_name, contexti, policy_params_path, eval_episodes, device):
    # print(f'carl_env_name = {carl_env_name}')
    carl_env_fn = eval(carl_env_name)
    context_labels = list(contexti[0].keys())
    # print(f'Loading UP ...')
    context_temp = {0:{k:0.0 for k in context_labels}}
    env_temp = init_carl(carl_env_fn,
                         contexts=context_temp,
                         obs_context_features=context_labels,
                         hide_context=False,
                         context_selector=StaticSelector)
    if not isinstance(env_temp.action_space, (gymnasium.spaces.Discrete, gymnasium.spaces.Box)):
        # print(f'env_temp.action_space={env_temp.action_space} not of correct type.')
        # raise TypeError
        env_temp = ToGymnasiumActionSpace(env_temp)
        
    # print('env_temp initialized')
    policy_test = load_policy_from_params(policy_params_path, env_temp, device=device)
    # print(f'Loading UP: Successful.')
    # print(f'contexti = {contexti}')
    eval_env = init_carl(carl_env_fn, 
                         contexts=contexti, 
                         obs_context_features=context_labels,
                         hide_context=False,
                         context_selector=StaticSelector
                        )
    mean_rew, std_rew = evaluate_policy(policy_test, eval_env, n_eval_episodes=eval_episodes)
    # print(f'context: {contexti[0]}, Reward: mean: {mean_rew:.2f} +/- {std_rew:.2f}')
    return mean_rew

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-carl_env_name', type=str, help='name of the CARL environment', default='CARLPendulum')
    parser.add_argument('-context_labels', '--context_labels', nargs='+', type=str, help='context labels', required=True)
    parser.add_argument('-context_min', '--context_min', nargs='+', type=float, help='minimum value of context', required=True)
    parser.add_argument('-context_max', '--context_max', nargs='+', type=float, help='maximum value of context', required=True)
    parser.add_argument('-up_prefix', type=str, help='prefix used to load up policy', required=True)
    parser.add_argument('-savedir', type=str, help='directory where experts are saved', default='saved_models/')
    parser.add_argument('-datadir', type=str, help='directory where data is saved', default='eval_data/')
    parser.add_argument('-datasuffix', type=str, help='suffix for the saved data', default='') # include underscore ('_') at the start of modelsuffix for proper seperation
    parser.add_argument('-up_policy_path', type=str, help='path to dir where up policy is saved', default='data/saved_models/')
    parser.add_argument('-up_suffix', type=str, help='suffix used to load up policy', default='')
    parser.add_argument('--load_params', action='store_true', help='bool var to load params') #default=False
    parser.add_argument('-distribution', type=str, help='sample distribution', default='gaussian')
    parser.add_argument('-nrounds', type=int, help='number of rounds of evaluation', default=5)
    parser.add_argument('-n_eval_samples', type=int, help='number of contexts to sample for evaluation', default=100)
    parser.add_argument('-eval_episodes', type=int, help='number of times to evaluate on each context', default=10)
    parser.add_argument('--parallel_eval', action='store_true', help='bool var to parallelize evaluation') # default=False
    parser.add_argument('-n_parallel_processes', type=int, help='number of parallel processes to create while evaluation', default=1)
    parser.add_argument('-device', type=str, help='device to use for training', default='cpu')
    parser.add_argument('-hp_env_id', type=str, help='env_id to be used for loading hyperparams', default='')
    
    args = parser.parse_args()

    carl_env_fn = eval(args.carl_env_name)
    context_labels = args.context_labels
    DEFAULT_CONTEXT = carl_env_fn.get_default_context()
    current_dir = os.getcwd()

    context_min = np.array(args.context_min)
    context_max = np.array(args.context_max)
    context_mean = []
    for key in DEFAULT_CONTEXT.keys():
        if key in context_labels:
            context_mean.append(DEFAULT_CONTEXT[key])

    ## Loading hyperparams from yml file
    split_idx = args.up_prefix.find('_')
    alg_name = args.up_prefix[split_idx+1:]
    config_path = os.path.join(current_dir, 'hyperparams/' + alg_name + '.yml')
    if args.hp_env_id == '':
        gym_id = carl_env_fn.env_name
    else:
        gym_id = args.hp_env_id
    unprocessed_hyperparams = read_hyperparams(config_path, gym_id)
    print('unprocessed_hyperparams = ', unprocessed_hyperparams)

    if 'policy_kwargs' in unprocessed_hyperparams:
        print(f'unprocessed_hyperparams[policy_kwargs]={unprocessed_hyperparams["policy_kwargs"]}')
        print(f'type(unprocessed_hyperparams[policy_kwargs]) = {type(unprocessed_hyperparams["policy_kwargs"])}')
        if type(eval(unprocessed_hyperparams["policy_kwargs"])) is dict:
            print(f'unprocessed_hyperparams[policy_kwargs] is dict.')
            if 'net_arch' in eval(unprocessed_hyperparams["policy_kwargs"]):
                print(f'unprocessed_hyperparams[policy_kwargs][net_arch] = {eval(unprocessed_hyperparams["policy_kwargs"])["net_arch"]}')
                processed_hyperparams = {"policy_kwargs":{"net_arch":eval(unprocessed_hyperparams["policy_kwargs"])["net_arch"]}}
            else:
                processed_hyperparams = None
        else:
            processed_hyperparams = {"policy_kwargs":eval(unprocessed_hyperparams["policy_kwargs"])}
    else:
        processed_hyperparams = None
    print('processed_hyperparams = ', processed_hyperparams)

    ## Set UP policy load path
    label_str = ""
    for label in context_labels:
        label_str = label_str + "_" + label
    up_policy_name = args.up_prefix + label_str + "_up"
    if args.up_suffix != '':
        up_policy_name = f'{up_policy_name}_{args.up_suffix}'
    up_policy_path = os.path.join(current_dir, args.up_policy_path, up_policy_name)
    print('up_policy_path =', up_policy_path)
    policy_params_path = f'{up_policy_path}_params.pkl'
    print(f'policy_params_path = {policy_params_path}')

    ## Debuging code start 1/2
    ## For checking the size of the policy network in .zip file and the final policy
    # file = f"{up_policy_path}.zip"
    # with zipfile.ZipFile(file) as archive:
    #     namelist = archive.namelist()
    #     print(f"namelist = {namelist}")
    #     if "data" in namelist:
    #         # Load class parameters that are stored
    #         # with either JSON or pickle (not PyTorch variables).
    #         json_data = archive.read("data").decode()
    #         data = json_to_data(json_data, custom_objects=None)
    #         # print(f"data = {data}")

    #     pth_files = [file_name for file_name in namelist if os.path.splitext(file_name)[1] == ".pth"]
    #     for file_path in pth_files:
    #         with archive.open(file_path, mode="r") as param_file:
    #             # File has to be seekable, but param_file is not, so load in BytesIO first
    #             # fixed in python >= 3.7
    #             file_content = io.BytesIO()
    #             file_content.write(param_file.read())
    #             # go to start of file
    #             file_content.seek(0)
    #             # Load the parameters with the right ``map_location``.
    #             # Remove ".pth" ending with splitext
    #             # Note(antonin): we cannot use weights_only=True, as it breaks with PyTorch 1.13, see GH#1911
    #             th_object = torch.load(file_content, map_location=args.device, weights_only=False)
    #             print(f"file_path = {file_path}, th_object type = {type(th_object)}")
    #             if file_path == "policy.pth":
    #                 print(f"th_object.keys() = {th_object.keys()}")
    #                 for k, v in th_object.items():
    #                     print(f"{k}: {v.shape}")
    #             if file_path == "pytorch_variables.pth" or file_path == "tensors.pth":
    #                 # PyTorch variables (not state_dicts)
    #                 pytorch_variables = th_object
    #                 print(f"pytorch_variables = {pytorch_variables}")
    ## Debuging code end 1/2

    if not args.parallel_eval:
        up_policy = load_policy(up_policy_path, device=args.device)
        print(f'Successfully loaded policy using .zip')
    else:
        print(f'Evaluation will be parallelized and policy will be loaded using .pkl')

    # Print architecture of up_policy
    ## Debugging code start  2/2
    # print(f"up_policy.policy.net_arch = {up_policy.policy.net_arch}")
    ## Debugging code end 2/2

    ## Loop for generating evaluation set
    prefix_experts = f'{args.up_prefix}_sb3'
    initial_seed = 100
    for round in range(args.nrounds):
        seed = initial_seed + round
        print(f'Starting round {round+1} with seed {seed}.')
        np.random.seed(seed)
        eval_context_array = np.zeros((len(context_mean), args.n_eval_samples))

        if args.load_params:
            # load_path = args.prefix_experts + label_str + "_evalset_" + str(args.n_eval_samples) +  "_" + str(seed) + args.datasuffix + ".npy"
            load_path = f'{prefix_experts}{label_str}_evalset_{args.n_eval_samples}_{seed}{args.datasuffix}.npy'
            load_path = os.path.join(current_dir, args.datadir, load_path)
            print('evalset load_path={0}'.format(load_path))
            eval_context_array = np.load(load_path)
            print('eval_context_array.shape = ', eval_context_array.shape)
        else:
            for i in range(len(context_mean)):
                if args.distribution == 'uniform':
                    print(f'Drawing evaluation samples for {context_labels[i]} from uniform distribution.')
                    eval_context_array[i,:] = np.random.uniform(context_min[i], context_max[i], size=args.n_eval_samples)
                elif args.distribution == 'gaussian':
                    # (mean-2*std, mean+2*std) = (min, max) i.e. with approx. 95% probability it will lie in range (min, max).
                    print(f'Drawing evaluation samples for {context_labels[i]} from gaussian distribution.')
                    tmean_i = (context_min[i] + context_max[i])/2
                    tstd_i = abs(context_max[i] - tmean_i)/2
                    eval_context_array[i,:] = np.random.normal(tmean_i, tstd_i, size=args.n_eval_samples)
                else:
                    raise ValueError(
                        f"Got distribution {args.distribution} which is not supported currently.\n"
                        "Supported distributions are uniform, gaussian."
                                    )
            save_path = prefix_experts + label_str + "_evalset_" + str(args.n_eval_samples) + "_" + str(seed) + args.datasuffix
            save_path = os.path.join(current_dir, args.datadir, save_path)
            os.makedirs(os.path.join(current_dir, args.datadir), exist_ok=True)
            print('evalset save_path={0}'.format(save_path))
            np.save(save_path, eval_context_array)
            print('Saved context data points.')

        eval_context_dict = {}
        for i in range(args.n_eval_samples):
            eval_context_dict[i] = {0:{key:value for key,value in zip(context_labels, eval_context_array[:,i])}}
            # eval_context_dict[i] = {key:value for key,value in zip(context_labels, eval_context_array[:,i])}
            if i < 5:
                print(f'eval_context_dict[{i}]={eval_context_dict[i]}')
        
        ## Labelling the set by evaluating UP(c) 
        if args.parallel_eval:
            args_list = [(args.carl_env_name, eval_context_dict[i], policy_params_path, args.eval_episodes, args.device, processed_hyperparams) for i in range(len(eval_context_dict))]
            with multiprocessing.Pool(processes=args.n_parallel_processes) as pool:
                temp_result = pool.starmap(eval_helper, args_list)
            rewards_upc = np.array(temp_result)
        else:
            rewards_upc = np.zeros((len(eval_context_dict),))
            for i in range(len(eval_context_dict)):
                contexti = eval_context_dict[i]
                eval_env = init_carl(carl_env_fn, 
                                     contexts=contexti, 
                                     obs_context_features=context_labels,
                                     hide_context=False,
                                     context_selector=StaticSelector
                                    )
                if not isinstance(eval_env.action_space, (gymnasium.spaces.Discrete, gymnasium.spaces.Box)):
                    # print(f'env_temp.action_space={env_temp.action_space} not of correct type.')
                    # raise TypeError
                    eval_env = ToGymnasiumActionSpace(eval_env)
                # rewards_upc[i], _ = compute_avg_return(eval_env, up_policy, num_episodes=args.eval_episodes)
                rewards_upc[i], _ = evaluate_policy(up_policy, eval_env, n_eval_episodes=args.eval_episodes)
        print(f'rewards_upc.shape = {rewards_upc.shape}')
        save_path = prefix_experts + label_str + "_upcrewards_" + str(args.n_eval_samples) + "_" + str(seed) + args.datasuffix
        save_path = os.path.join(current_dir, args.datadir, save_path)
        print('rewards_upc save_path={0}'.format(save_path))
        np.save(save_path, rewards_upc)
        print('Saved evaluations rewards.')
        
if __name__ == "__main__":
    try:
        global_start = time.time()
        main()
        global_end = time.time()
        print(f"Total execution time in minutes: {(global_end-global_start)/60:.4f}.")
    except: 
        print(traceback.format_exc())

    print("Completed execution.")