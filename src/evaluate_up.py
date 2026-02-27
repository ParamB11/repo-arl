'''
Code for evaluating UP policies.
Output: mean/max (best expert reward - policy reward). The rewards are calculated over each context value of an evaluation set.
'''

import argparse
import joblib
import os
import re
import sys
from tabulate import tabulate
import time
import traceback
sys.path.append('./crl_py')

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import tensorflow as tf

from baselines.common import tf_util as U
from carl.context.selection import StaticSelector, RoundRobinSelector
from carl.envs import CARLAcrobot, CARLCartPole, CARLLunarLander, CARLMountainCar, CARLPendulum
import gym
from gymnasium.wrappers import FlattenObservation, FilterObservation, StepAPICompatibility
from stable_baselines3.common.evaluation import evaluate_policy
from tf_agents.environments import tf_py_environment
import torch

# from carl_wrapper_tf_agents_py import CarlWrapper
from common_utils import init_carl, load_gru, load_policy
from config import get_config
from custom_wrappers import ConcatObservationAction, LstmObservationAction, ToGymActionSpace, ToGymObservationSpace, UPExpertWrapper
from envs.carl_brax import CARLBraxAnt, CARLBraxHalfcheetah
from eval_utils import compute_avg_return
# from evaluation_rel_th import env_fn
from lstm_context_pred import GRUContextPredictor, LSTMContextPredictor, RecurrentContextPredictor
from lstm_env_wrapper import InformerEnvWrapper, LstmEnvWrapper, PredEnvWrapper
from policy_transfer.utils.mlp import MLP
from osi_env_wrapper_carl import OSIEnvWrapper
# from ppo_sb3 import sb3PPO
# from test_functions import compute_avg_return_moe, compute_avg_return_predictor_moe, compute_avg_return_uposi
from test_functions import compute_avg_return_uposi

def to_gym(env):
    supported_spaces = (gym.spaces.box.Box, gym.spaces.discrete.Discrete)
    if not isinstance(env.observation_space, supported_spaces):
        env = ToGymObservationSpace(env)
    if not isinstance(env.action_space, supported_spaces):
        env = ToGymActionSpace(env)
    env = StepAPICompatibility(env, output_truncation_bool=False)
    return env
    
def env_fn(carl_env_fn, context, stack_size_obs, stack_size_act, show_context=False):
    tenv = init_carl(carl_env_fn, contexts=context, 
                     obs_context_features=list(context[0].keys()),
                     hide_context = not show_context,
                     context_selector=StaticSelector,
                    )
    env_hist = to_gym(
        ConcatObservationAction(tenv, stack_size_obs=stack_size_obs, stack_size_act=stack_size_act)
    )
    return env_hist

label_dict = {
    'up_c':'UP(c)',
    'uphist': 'UP(hist)',
    'uposi':'UP-OSI',
    'up_gru':'UP(GRU)',
    'up_informer': 'UP(informer)',
    'up_lstm':'UP(lstm)',
    'up_knn': 'UP(kNN)'
}
def label_fn(suffix):
    if suffix.find("dag")!=-1:
        split_idx = suffix.find("dag")
        assert suffix[:split_idx-1] in label_dict.keys(), f'Could not find suffix={suffix[:split_idx-1]} in label_dict.keys()={label_dict.keys()}.'
        label = label_dict[suffix[:split_idx-1]]
        label = label[:-1] + f'({suffix[split_idx:]}))'
        return label
    else:
        assert suffix in label_dict.keys(), f'Could not find suffix={suffix} in label_dict.keys()={label_dict.keys()}.'
        return label_dict[suffix]
def parse_args(args, parser):
    parser.add_argument('-context_min', '--context_min', nargs='+', type=float, help='minimum value of context', required=True)
    parser.add_argument('-context_max', '--context_max', nargs='+', type=float, help='maximum value of context', required=True)
    parser.add_argument('-context_experts', '--context_experts', nargs='+', type=float, help='context of experts', default = [0.0])
    parser.add_argument('-optimal_policy_name', type=str, help='policy whose rewards are considered optimal and metrics calculated wrt', default='expert')
    # parser.add_argument('-savedir', type=str, help='directory where experts are saved', default='/home/param/crl_notebooks/saved_models/')
    parser.add_argument('-transform_var', type=int, help='transform to use for CEP', default=0)
    parser.add_argument('-transform_suffix', type=str, help='suffix for loading transform', default='120')
    parser.add_argument('-cpredsuffix', type=str, help='suffix for the saved cpred model', default='') # include underscore ('_') at the start of cpredsuffix for proper seperation
    parser.add_argument('-mcsuffix', type=str, help='suffix for the saved mc model', default='') # include underscore ('_') at the start of mcsuffix for proper seperation
    parser.add_argument('-datadir', type=str, help='directory where data is saved', default='eval_data/')
    parser.add_argument('-datasuffix', type=str, help='suffix for the saved data', default='') # include underscore ('_') at the start of modelsuffix for proper seperation
    parser.add_argument('-all_suffix', '-all_suffix', nargs='+', type=str, help='suffix of agents to test', required=True)
    # parser.add_argument('-figdir', type=str, help='directory where figures are saved', default='/home/param/crl_notebooks/figs/')
    # parser.add_argument('--hide_context', action='store_false', help='bool var for hide_context') #default=True
    # parser.add_argument('-reward_threshold', type=float, help='mean reward should be close to this threshold', default=-200.0)
    # parser.add_argument('-n_points', type=int, help='number of testing points', default=40)
    # parser.add_argument('-eval_episodes', type=int, help='number of eval episodes', default=10)
    parser.add_argument('-nrounds', type=int, help='number of rounds of evaluation', default=5)
    parser.add_argument('-n_eval_samples', type=int, help='number of contexts to sample for evaluation', default=100)
    parser.add_argument('-nevals', type=int, help='number of times to evaluate on each context', default=1)

    # parser.add_argument('-up_policy_path', type=str, help='path to dir where up policy is saved', default='data/saved_models/')
    # parser.add_argument('-up_prefix', type=str, help='prefix used to load up policy', required=True)
    # parser.add_argument('-up_suffix', type=str, help='suffix used to load up policy', default='')
    parser.add_argument('-stack_height', type=int, help='stack_height to use for lstm predictor', default=4)
    parser.add_argument('-lookback', type=int, help='lookback to use for lstm/gru predictor', default=4) # new argument
    parser.add_argument('-up_hist', help='history step size', type=int, default=1)
    parser.add_argument('-OSI_hist', type=int, help='parameter for OSI', default=10)
    parser.add_argument('-knnepsuffix', type=str, help='suffix for lstm and dense model in cpred', default='') # include underscore ('_') at the start of cpredsuffix for proper seperation
    parser.add_argument('-osi_layers', '--osi_layers', nargs='+', type=float, help='set the layer dimensions for osi', default=[256, 128, 64])
    parser.add_argument('-recurrent_hidden_size', type=int, help='set hidden_size for pred_net', default=32)
    parser.add_argument('-hidden_layers', '--hidden_layers', nargs='*', type=float, help='set the hidden layer dimensions for pred_net', default=[])
    parser.add_argument('-osisuffix', type=str, help='suffix for osi in UP(osi)', default='') # include underscore ('_') at the start of cpredsuffix for proper seperation
    parser.add_argument('-lstmsuffix', type=str, help='suffix for lstm in UP(lstm)', default='') # do not include '_'
    parser.add_argument('-grusuffix', type=str, help='suffix for gru in UP(gru)', default='') # do not include '_'
    parser.add_argument('-uphistsuffix', type=str, help='suffix for gru in UP(hist)', default='') # do not include '_'
    parser.add_argument('-knnclfsuffix', type=str, help='suffix for loading transform', default='300')

    all_args = parser.parse_known_args(args)[0]

    return all_args

def main():
    args = sys.argv[1:]
    parser = get_config()
    args = parse_args(args, parser)
    current_dir = os.getcwd()

    # Local aliases to avoid rebinding the imported class names inside this function.
    # Use these aliases when we may swap in alternative implementations at runtime.
    RecurrentContextPredictor_cls = RecurrentContextPredictor
    PredEnvWrapper_cls = PredEnvWrapper
    
    carl_env_fn = eval(args.carl_env_name)
    DEFAULT_CONTEXT = carl_env_fn.get_default_context()
    
    context_labels = args.context_labels
    labels = np.array(context_labels)
    # print("labels = ", labels, type(labels))
    context_min = np.array(args.context_min)
    context_max = np.array(args.context_max)
    context_mean = []
    for key in DEFAULT_CONTEXT.keys():
        if key in context_labels:
            context_mean.append(DEFAULT_CONTEXT[key])
    
    for i in range(len(context_mean)):
        if context_mean[i] == 0.0:
            print(f'Found context_mean[{i}]={context_mean[i]}. Modifying it.')
            context_mean[i] = 1.0
    if not (labels.shape == context_min.shape == context_max.shape):
        print("labels.shape={0}, context_min.shape={1}, context_max.shape={2}".format(labels.shape,context_min.shape,context_max.shape))
        raise ValueError("The condition labels.shape == context_min.shape == context_max.shape is not satisfied.")

    if args.transform_var > 5:
        raise ValueError(
            'Found transform_var={0} which is invalid.'.format(args.transform_var)
        )
        
    labels_str = ""
    for label in context_labels:
        labels_str = labels_str + "_" + label

    label_scale_factor = np.ones((1, len(context_labels)))
    for idx in range(len(context_labels)):
        label_scale_factor[0, idx] = context_mean[idx]
        
    results = {}
    initial_seed = 100
    n_eval_samples = args.n_eval_samples
    expert_rewards = np.zeros((args.nrounds, n_eval_samples))
    eval_contexts = np.zeros((args.nrounds, n_eval_samples, len(context_mean)))
    
    for round in range(args.nrounds):
        seed = initial_seed + round
        print(f'Starting round {round+1} with seed {seed}.')
        np.random.seed(seed)
        # n_eval_samples = args.n_eval_samples
        eval_context_array = np.zeros((len(context_mean), n_eval_samples))
        load_path = args.prefix_experts + labels_str + "_evalset_" + str(n_eval_samples) +  "_" + str(seed) + args.datasuffix + ".npy"
        load_path = os.path.join(args.datadir, load_path)
        print('evalset load_path={0}'.format(load_path))
        eval_context_array = np.load(load_path)
        print('eval_context_array.shape = ', eval_context_array.shape)
        eval_contexts[round,:,:] = eval_context_array.T 
        
        eval_context_dict = {}
        for i in range(n_eval_samples):
            eval_context_dict[i] = {0:{key:value for key,value in zip(context_labels, eval_context_array[:,i])}}

        '''Loading optimal rewards'''
        # load_path = args.prefix_experts + labels_str + "_expertrewards_" + str(n_eval_samples) +  "_" + str(seed) + args.datasuffix + ".npy"
        load_path = f'{args.prefix_experts}{labels_str}_{args.optimal_policy_name}rewards_{n_eval_samples}_{seed}{args.datasuffix}.npy'
        load_path = os.path.join(args.datadir, load_path)
        print('expertrewards load_path={0}'.format(load_path))
        expert_rewards[round,:] = np.load(load_path)
        print('expert_rewards[{0}]: Mean = {1:.2f}, Std. dev = {2:.2f}'
              .format(round, np.mean(expert_rewards[round]), np.std(expert_rewards[round])))
        
        '''Testing up-osi agent'''
        uposi_suffix = []
        for suffix in args.all_suffix:
            if suffix.find("up")!=-1:
                uposi_suffix.append(suffix)

        if uposi_suffix != []:
            for suffix in uposi_suffix:
                if suffix.find("c")!=-1:
                    label = label_fn(suffix)
                    print(f'Testing {label} agent ...')
                    up_policy_name = args.up_prefix+labels_str+"_up"
                    if args.up_suffix != '':
                        up_policy_name = f'{up_policy_name}_{args.up_suffix}'
                    up_policy_path = os.path.join(current_dir, args.up_policy_path, up_policy_name)
                    print('up_policy_path =', up_policy_path)
                    # up_policy = PPO.load(up_policy_path, env=None)
                    up_policy = load_policy(up_policy_path, args.device)
                    rewards_uposi = np.zeros((len(eval_context_dict),))
                    for i in range(len(eval_context_dict)):
                        contexti = eval_context_dict[i]
                        eval_env = init_carl(carl_env_fn, 
                                             contexts=contexti, 
                                             obs_context_features=context_labels,
                                             hide_context=False,
                                             context_selector=RoundRobinSelector
                                            )
                        rewards_uposi, _ = compute_avg_return(eval_env, up_policy, num_episodes=args.nevals)
                    if round == 0:
                        results[label] = np.zeros((args.nrounds,len(eval_context_dict)))
                        results[label][0] = rewards_uposi
                    else:
                        results[label][round] = rewards_uposi
                    print("Agent: {0}. Average reward over evaluation set: {1:.2f}, Standard deviation: {2:.2f}"
                          .format(label, np.mean(rewards_uposi), np.std(rewards_uposi)))
                
                elif suffix.find("osi")!=-1:
                    label = label_fn(suffix)
                    print(f'Testing {label} agent ...')
                    up_policy_name = args.up_prefix+labels_str+"_up"
                    if args.up_suffix != '':
                        up_policy_name = f'{up_policy_name}_{args.up_suffix}'
                    up_policy_path = os.path.join(current_dir, args.up_policy_path, up_policy_name)
                    print('up_policy_path =', up_policy_path)
                    # up_policy = PPO.load(up_policy_path, env=None)
                    up_policy = load_policy(up_policy_path, 'cpu')

                    env_hist = env_fn(carl_env_fn, eval_context_dict[0], args.OSI_hist+1, args.OSI_hist, show_context=False)
                    osi_name = 'osi'+labels_str
                    if args.up_suffix != '':
                        osi_name = f'{osi_name}_{args.up_suffix}'
                    osi_name = osi_name + args.osisuffix
                    osi_layers = [int(e) for e in args.osi_layers]
                    osi = MLP(name=osi_name, in_dim=env_hist.observation_space.shape[0], out_dim=len(context_labels), layers=osi_layers,
                              activation=tf.nn.relu, last_activation=None, dropout=0.0)
                    osi_dir = os.path.join(current_dir, 'data/osi_data/'+args.up_prefix+labels_str)
                    osi_path = os.path.join(osi_dir, osi_name + '_params.pkl')
                    print('osi_path =', osi_path)
                    sess = tf.compat.v1.InteractiveSession()
                    U.initialize()
                    osi.set_variable_from_dict(joblib.load(osi_path))
                    rewards_uposi = np.zeros((len(eval_context_dict),))
                    for i in range(len(eval_context_dict)):
                        contexti = eval_context_dict[i]
                        # py_envi = CarlWrapper(carl_env_fn=carl_env_fn, contexts=contexti, 
                        #                    hide_context=True, context_selector=StaticSelector)
                        env_histi = env_fn(carl_env_fn, contexti, args.OSI_hist+1, args.OSI_hist, show_context=False)
                        py_envi = OSIEnvWrapper(env_histi, osi, args.OSI_hist, len(context_labels), label_scale_factor[0])
                        # envi = tf_py_environment.TFPyEnvironment(py_envi)
                        rewards_uposi[i], _ = compute_avg_return_uposi(py_envi, up_policy, num_episodes=args.nevals)
                    if round == 0:
                        results[label] = np.zeros((args.nrounds,len(eval_context_dict)))
                        results[label][0] = rewards_uposi
                    else:
                        results[label][round] = rewards_uposi
                    print("Agent: {0}. Average reward over evaluation set: {1:.2f}, Standard deviation: {2:.2f}"
                          .format(label, np.mean(rewards_uposi), np.std(rewards_uposi)))
                    tf.compat.v1.InteractiveSession.close(sess)
                elif suffix.find("lstm")!=-1:
                    label = label_fn(suffix)
                    print(f'Testing {label} agent ...')
                    up_policy_name = args.up_prefix+labels_str+"_up"
                    if args.up_suffix != '':
                        up_policy_name = f'{up_policy_name}_{args.up_suffix}'
                    up_policy_path = os.path.join(current_dir, args.up_policy_path, up_policy_name)
                    print('up_policy_path =', up_policy_path)
                    up_policy = load_policy(up_policy_path, args.device)
                    comb_prefix = args.up_prefix + labels_str #+ args.knnepsuffix
                    if suffix.find("dag")!=-1:
                        split_idx = suffix.find("dag")
                        # pred_name = comb_prefix + f"_up_lstm_predictor_{suffix[split_idx:]}.pt"
                        pred_name = comb_prefix + f"_up_lstm_predictor"
                        if args.lstmsuffix!='':
                            pred_name = f"{pred_name}_{args.lstmsuffix}_{suffix[split_idx:]}.pt"
                        else:
                            pred_name = f"{pred_name}_{suffix[split_idx:]}.pt"
                    else:
                        pred_name = comb_prefix + "_up_predictor.pt"

                    if round == 0:
                        if args.lstmsuffix.endswith("_re"):
                            print(f'Using lstm_env_wrapper_re.PredEnvWrapper since lstmsuffix={args.lstmsuffix} ends with "_re".')
                            import lstm_context_pred_re
                            import lstm_env_wrapper_re
                            RecurrentContextPredictor_cls = lstm_context_pred_re.RecurrentContextPredictor
                            PredEnvWrapper_cls = lstm_env_wrapper_re.PredEnvWrapper

                    tenv = init_carl(carl_env_fn)
                    eng_bool = False
                    stack_height = args.stack_height #4
                    obs_len = tenv.observation_space.shape[0]
                    if tenv.action_space.shape ==():
                        act_len = 1
                    else:
                        act_len = tenv.action_space.shape[0]
                    if eng_bool:
                        state_len = 6*obs_len + 2*act_len
                    else:
                        state_len = (stack_height+2)*obs_len + (stack_height+1)*act_len # for stack_height = 4
                    # pred_net = LSTMContextPredictor(state_len, 32, len(context_labels), device=args.device)
                    hidden_layers = [int(e) for e in args.hidden_layers]
                    pred_net = RecurrentContextPredictor_cls(
                        state_len, args.recurrent_hidden_size, hidden_layers, len(context_labels), 
                        model_arch='lstm', #dropout_prob=args.dropout_prob, 
                        device=args.device
                    )
                    pred_net_path = os.path.join(current_dir, args.savedir, pred_name)
                    print(f'pred_net_path = {pred_net_path}')
                    if args.device == 'cpu' or not torch.cuda.is_available():
                        pred_net.load_state_dict(torch.load(pred_net_path, map_location=torch.device('cpu')))
                    else: 
                        pred_net.load_state_dict(torch.load(pred_net_path))
                    # print(f'pred_net={pred_net}')
                    
                    rewards_uplstm = np.zeros((len(eval_context_dict),))
                    for i in range(len(eval_context_dict)):
                        contexti = eval_context_dict[i]
                        eval_env = init_carl(carl_env_fn, 
                                             contexts=contexti, 
                                             obs_context_features=context_labels,
                                             hide_context=True,
                                             context_selector=StaticSelector
                                            )
                        eval_env_wrapped = LstmObservationAction(eval_env, stack_size=stack_height)
                        if args.lstmsuffix.endswith("_re"):
                            eval_env_wrapped = PredEnvWrapper_cls(eval_env_wrapped, pred_net, net_arch='lstm', lookback=args.lookback, 
                                                              context_dim=len(context_labels), context_scale=label_scale_factor)
                        else:
                            eval_env_wrapped = LstmEnvWrapper(eval_env_wrapped, pred_net, stack_size=stack_height, 
                                                            context_dim=len(context_labels), context_scale=label_scale_factor)
                        rewards_uplstm[i], _ = compute_avg_return(eval_env_wrapped, up_policy, num_episodes=args.nevals)
                    # print(f'rewards_uplstm.shape={rewards_uplstm.shape}')
                    if round == 0:
                        results[label] = np.zeros((args.nrounds,len(eval_context_dict)))
                        results[label][0] = rewards_uplstm
                    else:
                        results[label][round] = rewards_uplstm
                    print("Agent: {0}. Average reward over evaluation set: {1:.2f}, Standard deviation: {2:.2f}"
                          .format(label, np.mean(rewards_uplstm), np.std(rewards_uplstm)))
                elif suffix.find("gru")!=-1:
                    label = label_fn(suffix)
                    print(f'Testing {label} agent ...')
                    up_policy_name = args.up_prefix+labels_str+"_up"
                    if args.up_suffix != '':
                        up_policy_name = f'{up_policy_name}_{args.up_suffix}'
                    up_policy_path = os.path.join(current_dir, args.up_policy_path, up_policy_name)
                    print('up_policy_path =', up_policy_path)
                    up_policy = load_policy(up_policy_path, args.device)
                    comb_prefix = args.up_prefix + labels_str #+ args.knnepsuffix
                    if suffix.find("dag")!=-1:
                        split_idx = suffix.find("dag")
                        # pred_name = comb_prefix + f"_up_gru_predictor_{suffix[split_idx:]}.pt"
                        pred_name = comb_prefix + f"_up_gru_predictor"
                        if args.grusuffix != '':
                            pred_name = f"{pred_name}_{args.grusuffix}_{suffix[split_idx:]}.pt"
                        else:
                            pred_name = f"{pred_name}_{suffix[split_idx:]}.pt"
                    else:
                        pred_name = comb_prefix + "_gru_up_predictor.pt"
                    # pred_name = comb_prefix + "_gru_up_predictor.pt"

                    if round == 0:
                        if args.grusuffix.endswith("_re"):
                            print(f'Using lstm_env_wrapper_re.PredEnvWrapper since grusuffix={args.grusuffix} ends with "_re".')
                            import lstm_context_pred_re
                            import lstm_env_wrapper_re
                            RecurrentContextPredictor_cls = lstm_context_pred_re.RecurrentContextPredictor
                            PredEnvWrapper_cls = lstm_env_wrapper_re.PredEnvWrapper

                    tenv = init_carl(carl_env_fn)
                    eng_bool = False
                    stack_height = args.stack_height #4
                    obs_len = tenv.observation_space.shape[0]
                    if tenv.action_space.shape ==():
                        act_len = 1
                    else:
                        act_len = tenv.action_space.shape[0]
                    if eng_bool:
                        state_len = 6*obs_len + 2*act_len
                    else:
                        state_len = (stack_height+2)*obs_len + (stack_height+1)*act_len # for stack_height = 4
                    # pred_net = LSTMContextPredictor(state_len, 32, len(context_labels), device=args.device)
                    hidden_layers = [int(e) for e in args.hidden_layers]
                    pred_net = RecurrentContextPredictor_cls(
                        state_len, args.recurrent_hidden_size, hidden_layers, len(context_labels), 
                        model_arch='gru', #dropout_prob=args.dropout_prob, 
                        device=args.device
                    )
                    pred_net_path = os.path.join(current_dir, args.savedir, pred_name)
                    print(f'pred_net_path = {pred_net_path}')
                    if args.device == 'cpu' or not torch.cuda.is_available():
                        pred_net.load_state_dict(torch.load(pred_net_path, map_location=torch.device('cpu')))
                    else: 
                        pred_net.load_state_dict(torch.load(pred_net_path))
                    # pred_net = load_gru(carl_env_fn, context_labels, pred_net_path, stack_height=4, eng_bool=False, device=args.device)
                    rewards_upgru = np.zeros((len(eval_context_dict),))
                    for i in range(len(eval_context_dict)):
                        contexti = eval_context_dict[i]
                        eval_env = init_carl(carl_env_fn, 
                                             contexts=contexti, 
                                             obs_context_features=context_labels,
                                             hide_context=True,
                                             context_selector=RoundRobinSelector
                                            )
                        eval_env_wrapped = LstmObservationAction(eval_env, stack_size=stack_height)
                        if args.grusuffix.endswith("_re"):
                            eval_env_wrapped = PredEnvWrapper_cls(eval_env_wrapped, pred_net, net_arch='gru', lookback=args.lookback, 
                                                              context_dim=len(context_labels), context_scale=label_scale_factor)
                        else:
                            eval_env_wrapped = PredEnvWrapper_cls(eval_env_wrapped, pred_net, net_arch='gru', stack_size=stack_height, 
                                                            context_dim=len(context_labels), context_scale=label_scale_factor)
                        rewards_upgru[i], _ = compute_avg_return(eval_env_wrapped, up_policy, num_episodes=args.nevals)
                    if round == 0:
                        results[label] = np.zeros((args.nrounds,len(eval_context_dict)))
                        results[label][0] = rewards_upgru
                    else:
                        results[label][round] = rewards_upgru
                    print("Agent: {0}. Average reward over evaluation set: {1:.2f}, Standard deviation: {2:.2f}"
                          .format(label, np.mean(rewards_upgru), np.std(rewards_upgru)))
                else:
                    print(f'Found suffix {suffix} which is not implemented yet.')
    '''Plotting all the results'''
    print("Final Results:")
    print('expert_rewards: Mean = ', np.mean(expert_rewards, axis=1))
    print('expert_rewards: Mean of means: {0:.2f}, Std. of means: {1:.2f}'
          .format(np.mean(np.mean(expert_rewards, axis=1)), np.std(np.mean(expert_rewards, axis=1))))

    robustness_gap_data = {}
    mean_gap_data = {}
    for key,value in results.items():
        # print(f'key={key}, value.shape={value.shape}')
        diff = expert_rewards - value
        # print('diff =', diff)
        sign_factor = stats.mode(np.sign(expert_rewards.flatten())).mode
        # diff_scaled = sign_factor*diff/expert_rewards
        diff_scaled = diff/abs(expert_rewards)
        mean_diff = np.mean(diff, axis=1)
        max_diff = np.max(diff, axis=1)
        mean_diff_scaled = np.mean(diff_scaled, axis=1)
        max_diff_scaled = np.max(diff_scaled, axis=1)
        diff_scaled_sort_idx = np.argsort(diff_scaled, axis=1)
        mean_gap_data[key] = f'{np.mean(mean_diff_scaled):.2f} +/- {np.std(mean_diff_scaled):.2f}'
        robustness_gap_data[key] = f'{np.mean(max_diff_scaled):.2f} +/- {np.std(max_diff_scaled):.2f}'
        value_mean = np.mean(value, axis=1)
        value_min = np.min(value, axis=1)
        # print(f'{key}: Mean Rewards: Mean: {np.mean(value_mean):.2f} Std.: {np.std(value_mean):.2f}')
        # print(f'{key}: Worst case Rewards: Mean: {np.mean(value_min):.2f} Std.: {np.std(value_min):.2f}')
        
        # print('Absolute values:')
        # print('{0}: Mean difference:{1}, Max difference:{2}'.format(key, mean_diff, max_diff))
        # print('{0}: Mean of means: {1:.2f}, Std. of means: {2:.2f}'
        #       .format(key, np.mean(mean_diff), np.std(mean_diff)))
        # print('{0}: Mean of maxs: {1:.2f}, Std. of maxs: {2:.2f}'
        #       .format(key, np.mean(max_diff), np.std(max_diff)))
        # print('Percentages:')
        # print('{0}: Mean difference:{1}, Max difference:{2}'.format(key, mean_diff_scaled, max_diff_scaled))
        # print('{0}: Mean of means: {1:.2f}, Std. of means: {2:.2f}'
        #       .format(key, np.mean(mean_diff_scaled), np.std(mean_diff_scaled)))
        # print('{0}: Mean of maxs: {1:.2f}, Std. of maxs: {2:.2f}'
        #       .format(key, np.mean(max_diff_scaled), np.std(max_diff_scaled)))

    # print(f'mean_gap_data (type={type(mean_gap_data)}) = {mean_gap_data}')
    mean_gap_table = [list(mean_gap_data.keys()), list(mean_gap_data.values())]
    robustness_gap_table = [list(robustness_gap_data.keys()), list(robustness_gap_data.values())]
    print('Mean performance gap:')
    print(tabulate(mean_gap_table, tablefmt="grid"))
    print('Robustness gap:')
    print(tabulate(robustness_gap_table, tablefmt="grid"))

if __name__ == '__main__':
    try:
        global_start = time.time()
        main()
        global_end = time.time()
        print(f"Total execution time in minutes: {(global_end-global_start)/60}.")
    except:
        print(traceback.format_exc())
        
    # print('Releasing GPU memory ...')
    # tf.keras.backend.clear_session()
    # print('Released GPU memory.')
    print("Completed execution.")