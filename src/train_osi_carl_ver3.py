'''
Base code: train_osi_carl_ver2
Additions over ver2: The basic idea of creating ver3 is to have a training script of osi that is aligned with crl_torch/train_lstm_dagger.py
Specifically, the changes are as follows: 
1. Feature x_t = (s_t, s_{t-1}, s_{t-2}, ..., a_{t-1}, a_{t-2}, ...) instead of x_t = (s_{t-1}, s_{t-2}, ..., a_{t-1}, a_{t-2}, ...). 
2. For training usually use stack_size_obs=6, stack_size_act=5 which is close to OSI_hist=5 instead of OSI_hist=10. 
3. The label is scaled by mean instead of using raw label.
'''
from collections import deque
import os
import sys
import time
import traceback
sys.path.append('./src')

from baselines import logger
from baselines.common import tf_util as U
from baselines.common.mpi_adam import MpiAdam
from carl.context.selection import RoundRobinSelector, StaticSelector
from carl.envs import CARLAcrobot, CARLLunarLander, CARLMountainCar, CARLPendulum
import joblib
import gym
from gymnasium import spaces
from gymnasium.wrappers import FlattenObservation, FilterObservation, StepAPICompatibility
import numpy as np
import policy_transfer.envs
from policy_transfer.policies.mirror_policy import *
from policy_transfer.policies.mlp_policy import *
from policy_transfer.utils.mlp import *
from policy_transfer.utils.optimizer import *
import tensorflow as tf

from common_utils import init_carl, load_policy
from custom_wrappers import ToGymActionSpace, ToGymObservationSpace, ConcatObservationAction
from envs.carl_brax import CARLBraxAnt, CARLBraxHalfcheetah
from osi_carl_utils import update_queues
from osi_env_wrapper_carl import OSIEnvWrapper
from envs.carl_brax import CARLBraxAnt, CARLBraxHalfcheetah

def osi_train_callback(model, name, iter):
    params = model.get_variable_dict()
    save_path = os.path.join(logger.get_dir(), name+'_params.pkl')
    joblib.dump(params, save_path, compress=True)

def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-carl_env_name', type=str, help='name of the CARL environment', default='CARLPendulum')
    parser.add_argument('-context_labels', '--context_labels', nargs='+', type=str, help='context labels', required=True)
    parser.add_argument('-seed', help='RNG seed', type=int, default=0)
    parser.add_argument('-exp_name', type=str, help='experiment name to be concatenated with model name', default='') # do not add '_' at the start
    parser.add_argument('-OSI_hist', help='history step size', type=int, default=5)
    parser.add_argument('-dropout_prob', help='dropout probability for training pred_net', type=float, default=0.0)
    parser.add_argument('-policy_path', help='path to policy', type=str, default="data/saved_models/")
    parser.add_argument('-contextdir', type=str, help='dir for storing/loading context data', default='context_data/')
    parser.add_argument('-policy_prefix', type=str, help='prefix of the saved universal policy', required=True)
    parser.add_argument('-up_suffix', type=str, help='suffix used to load up policy', default='')
    parser.add_argument('-osi_layers', '--osi_layers', nargs='+', type=float, help='set the layer dimensions for osi', default=[256, 128, 64])
    parser.add_argument('-osi_iteration', help='number of iterations', type=int, default=6)
    parser.add_argument('-data_retain_frac', help='fraction of data to retain from previous rounds', type=float, default=1.0)
    parser.add_argument('-training_sample_num', help='number of training samples per iteration', type=int, default=20000)
    parser.add_argument('-action_noise', help='noise added to action', type=float, default=0.0)

    parser.add_argument('-rel_std', '--rel_std', nargs='+', type=float, help='relative standard deviation for training', default=[0.25])
    parser.add_argument('-n_train_contexts', type=int, help='number of contexts to sample for training', default=100)
    args = parser.parse_args()

    # extract information from args
    # name = args.name
    seed = args.seed
    OSI_hist = args.OSI_hist
    policy_path = args.policy_path
    osi_iteration = args.osi_iteration
    training_sample_num = args.training_sample_num
    current_dir = os.getcwd()

    context_rel_std = args.rel_std
    n_samples = args.n_train_contexts

    # setup the environments
    carl_env_fn = eval(args.carl_env_name)
    context_labels = args.context_labels #dyn_params
    DEFAULT_CONTEXT = carl_env_fn.get_default_context()

    for label in context_labels: 
        if label not in DEFAULT_CONTEXT.keys():
            print(f"Ignoring invalid label {label}.")

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

    label_scale_factor = np.ones((1, len(context_labels)))
    for idx in range(len(context_labels)):
        if context_mean[idx] != 0.0:
            label_scale_factor[0, idx] = context_mean[idx]
        else:
            label_scale_factor[0, idx] = 1.0
    
    if isinstance(args.rel_std, list):
        if len(args.rel_std) == 1:
            context_rel_std = [args.rel_std[0]]*len(context_mean)
        else:
            context_rel_std = args.rel_std
    else:
        raise TypeError(f"Found type(args.rel_std)={type(args.rel_std)}. Expected list.")
    
    context_std = np.zeros(len(context_mean))
    for i in range(len(context_mean)):
        mean = context_mean[i]
        if mean != 0.0:
            context_std[i] = abs(mean)*context_rel_std[i]
        else:
            context_std[i] = context_rel_std[i]
    print("context_std = ", context_std)

    train_context_array = np.zeros((len(context_mean), n_samples))

    split_idx = args.policy_prefix.find('_')
    env_prefix = args.policy_prefix[:split_idx]
    labels_str = ""
    for label in context_labels:
        labels_str = labels_str + "_" + label
    load_path = env_prefix + labels_str + "_contexts_up_" + str(n_samples) + ".npy"
    load_path = os.path.join(args.contextdir, load_path)
    print('contexts_up load_path={0}'.format(load_path))
    train_context_array = np.load(load_path)
    print('train_context_array.shape = ', train_context_array.shape)
    
    train_context_dict = {}
    for i in range(n_samples):
        train_context_dict[i] = {key:value for key,value in zip(context_labels, train_context_array[:,i])}
        if i<5:
            print("train_context_dict[{0}] = {1}".format(i, train_context_dict[i]))

    print('len(train_context_dict) = ', len(train_context_dict))

    def to_gym(env):
        supported_spaces = (gym.spaces.box.Box, gym.spaces.discrete.Discrete)
        if not isinstance(env.observation_space, supported_spaces):
            env = ToGymObservationSpace(env)
        if not isinstance(env.action_space, supported_spaces):
            env = ToGymActionSpace(env)
        env = StepAPICompatibility(env, output_truncation_bool=False)
        return env

    env_up = to_gym(init_carl(carl_env_fn, 
                              contexts=train_context_dict, 
                              obs_context_features=list(train_context_dict[0].keys()), 
                              hide_context=False, 
                              context_selector=RoundRobinSelector))
    env_hist = to_gym(ConcatObservationAction(
        init_carl(carl_env_fn, 
                  contexts=train_context_dict,
                  obs_context_features=list(train_context_dict[0].keys()),
                  hide_context=True,
                  context_selector=RoundRobinSelector
                 ),
        stack_size_obs=OSI_hist+1,
        stack_size_act=OSI_hist
    ))
    stack_size_obs, stack_size_act = OSI_hist+1, OSI_hist
    obs_queue = deque(maxlen=stack_size_obs)
    act_queue = deque(maxlen=stack_size_act)
    tenv = init_carl(carl_env_fn,
                     contexts=train_context_dict, 
                     obs_context_features=list(train_context_dict[0].keys()), 
                     hide_context=True, 
                     context_selector=RoundRobinSelector)
    padding_value_obs = np.zeros(tenv.observation_space.shape, dtype=tenv.observation_space.dtype) # create_zero_array(tenv.observation_space)
    padding_value_act = np.zeros(tenv.action_space.shape, dtype=tenv.action_space.dtype)
    
    def policy_fn(name, ob_space, ac_space):
        hid_size = 64
        return MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                    hid_size=hid_size, num_hid_layers=3)

    def policy_mirror_fn(name, ob_space, ac_space):
        return MirrorPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                            hid_size=64, num_hid_layers=3, observation_permutation=env_up.env.obs_perm,
                            action_permutation=env_up.env.act_perm, soft_mirror=False)


    labels_str = ""
    # for label in dyn_params:
    for label in context_labels:
        labels_str += "_" + label
    # configure things and load the learned universal policy
    config_name = 'data/osi_data/' + args.policy_prefix + labels_str

    logger.configure(config_name, ['json', 'stdout'])

    test_param_dict = locals()
    with open(logger.get_dir() + "/testinfo.txt", "w") as text_file:
        text_file.write(str(test_param_dict))

    if 'mirror' in policy_path:
        pol_fn = policy_mirror_fn
    else:
        pol_fn = policy_fn

    policy_name = args.policy_prefix+labels_str+"_up"
    if args.up_suffix != '':
        policy_name = f'{policy_name}_{args.up_suffix}'
    up_policy_path = os.path.join(current_dir, args.policy_path, policy_name)
    up_policy = load_policy(up_policy_path, 'cpu')

    print('env_hist.obs_space.shape[0] =', env_hist.observation_space.shape[0])
    osi_name = 'osi'+labels_str
    if args.up_suffix != '':
        osi_name = f'{osi_name}_{args.up_suffix}'
    if args.exp_name != '':
        osi_name = f'{osi_name}_{args.exp_name}'

    osi_layers = [int(e) for e in args.osi_layers]
    osi = MLP(name=osi_name, in_dim=env_hist.observation_space.shape[0], out_dim=len(context_labels), 
              layers=osi_layers, #[256, 128, 64],
              activation=tf.nn.relu, 
              last_activation=None, 
              dropout=args.dropout_prob) # 0.1)
    updater = MpiAdam(osi.get_trainable_variables())
    optimizer = RegressorOptimizer(osi, updater)
    for params_list in osi.params:
        for param in params_list:
            print(f'name: {param.name}, shape: {param.shape}')
    sess = tf.compat.v1.InteractiveSession()
    U.initialize()

    osi_env = OSIEnvWrapper(env_hist, osi, OSI_hist, len(context_labels), np.squeeze(label_scale_factor))

    updater.sync()

    seed_up = seed 
    seed_hist = seed 
    
    env_up.reset(seed=seed_up)
    env_hist.reset(seed=seed_hist)
    for _ in range(stack_size_obs - 1):
        obs_queue.appendleft(padding_value_obs)
    for _ in range(stack_size_act):
        act_queue.appendleft(padding_value_act)

    ## Define max_abs_action_space to add action noise
    if args.action_noise != 0.0:
        print(f'args.action_noise = {args.action_noise}')
    tenv = carl_env_fn()
    if isinstance(tenv.action_space, (gym.spaces.box.Box, gymnasium.spaces.box.Box)):
        print(f'action_space = {tenv.action_space}')
        if len(tenv.action_space.shape) == 1:
            space_low = np.expand_dims(tenv.action_space.low, axis=1)
            space_high = np.expand_dims(tenv.action_space.high, axis=1)
        else:
            space_low = tenv.action_space.low
            space_high = tenv.action_space.high
        abs_action_space = np.concatenate((space_low, space_high), axis=1)
        print(f'abs_action_space = {abs_action_space}')
        max_abs_action_space = np.max(abs_action_space, axis=1)
        print(f'max_abs_action_space = {max_abs_action_space}')

    input_data = []
    output_data = []
    for iter in range(osi_iteration):
        print('------------- Iter ', iter, ' ----------------')
        # collect samples
        env_to_use = env_up
        if iter > 0:
            env_to_use = osi_env

        lengths = []
        collected_data_size = 0
        new_input = []
        new_output = []
        while collected_data_size < training_sample_num:
            for _ in range(stack_size_obs - 1):
                obs_queue.appendleft(padding_value_obs)
            for _ in range(stack_size_act):
                act_queue.appendleft(padding_value_act)
            # collect one trajectory
            o, cid = env_to_use.reset()
            if iter == 0: osi_env.reset()
            one_obs_len = env_to_use.unwrapped.observation_space.shape[0]
            last_obs = o[-one_obs_len:]
            last_act = padding_value_act
            
            length = 0
            while True:
                if iter == 0:
                    true_dyn = np.array([value for value in env_to_use.get_wrapper_attr('context').values()])
                else:
                    true_dyn = np.array([value for value in env_to_use.wrapped_env.get_wrapper_attr('context').values()])
        
                if iter == 0:
                    obs_queue, act_queue, new_obs = update_queues(obs_queue, act_queue, obs=last_obs, act=last_act)
                    osi_input = new_obs
                else:
                    obs_queue, act_queue, new_obs = update_queues(obs_queue, act_queue, obs=last_obs, act=last_act)
                    osi_input = new_obs

                new_input.append(osi_input)
                true_dyn_scaled = true_dyn/np.squeeze(label_scale_factor)
                new_output.append(true_dyn_scaled)

                act, _ = up_policy.predict(o, deterministic=False)
                if isinstance(tenv.action_space, (gym.spaces.box.Box, gymnasium.spaces.box.Box)):
                    noise = np.random.normal(0, args.action_noise, size=act.shape[0])*max_abs_action_space
                    noisy_act = act + noise
                elif isinstance(tenv.action_space, (gym.spaces.discrete.Discrete, gymnasium.spaces.discrete.Discrete)):
                    ## choose a random action with probability args.action.noise
                    if np.random.uniform() < args.action_noise:
                        noisy_act = env_to_use.action_space.sample()
                        while noisy_act == act:
                            print(f'Noisy action matched action. Resampling. noisy_act = {noisy_act}, act = {act}.')
                            noisy_act = env_to_use.action_space.sample()
                        print(f'Random action = {act}, noisy action = {noisy_act}')
                    else:
                        noisy_act = act
                o, r, d, _ = env_to_use.step(noisy_act)
                last_obs = o[-one_obs_len:]
                last_act = noisy_act # act
                length += 1
                collected_data_size += 1

                if d:
                    lengths.append(length)
                    break
        print('Average rollout length: ', np.mean(lengths))
        # delete a fraction of input_data
        if iter > 0:
            split_idx = int(args.data_retain_frac*len(input_data))
            input_data = input_data[:split_idx]
            output_data = output_data[:split_idx]
            print('(after deletion) len(input_data)={0}, len(output_data)={1}'.format(len(input_data), len(output_data)))
        print(f'len(new_input) = {len(new_input)}, len(new_output) = {len(new_output)}')
        # permute and add new_input to input_data
        temp_input = input_data + new_input
        temp_output = output_data + new_output
        index_arr = np.arange(len(temp_input))
        index_arr = np.random.permutation(index_arr)
        input_data = [temp_input[i] for i in index_arr]
        output_data = [temp_output[i] for i in index_arr]
        print('len(input_data)={0}, len(output_data)={1}'.format(len(input_data), len(output_data)))
        # update osi model
        optimizer.fit_data(np.array(input_data), np.array(output_data), iter_num = 200, save_model_callback=osi_train_callback)
        params = osi.get_variable_dict()
        joblib.dump(params, logger.get_dir() + '/'+osi_name+'_params_'+str(iter)+'.pkl', compress=True)

if __name__ == '__main__':
    try:
        global_start = time.time()
        main()
        global_end = time.time()
        print(f"Total execution time in minutes: {(global_end-global_start)/60}.")
    except:
        print(traceback.format_exc())

    print("Completed execution.")