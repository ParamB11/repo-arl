import argparse
import os
import sys
import time
import traceback

from carl.context.selection import RoundRobinSelector, StaticSelector
from carl.envs import CARLAcrobot, CARLLunarLander, CARLMountainCar, CARLPendulum
import numpy as np
import torch
from torch.optim import Adam

from collect_utils import collect_timesteps_up, collect_timesteps_uppred
from common_utils import init_carl, load_policy
from custom_wrappers import LstmObservationAction
from envs.carl_brax import CARLBraxAnt, CARLBraxHalfcheetah
from lstm_context_pred import GRUContextPredictor, LSTMContextPredictor, RecurrentContextPredictor
from lstm_env_wrapper import LstmEnvWrapper, PredEnvWrapper
from lstm_utils import LstmOptimizer

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-carl_env_name', type=str, help='name of the CARL environment', default='CARLPendulum')
    parser.add_argument('-context_labels', '--context_labels', nargs='+', type=str, help='context labels', required=True)
    parser.add_argument('-stack_height', type=int, help='stack_height to use for lstm predictor', default=4)
    parser.add_argument('-model_arch', type=str, help='specify pred_net architecture. options=[lstm, gru]', default='lstm')
    parser.add_argument('-recurrent_hidden_size', type=int, help='set hidden_size for pred_net', default=32)
    parser.add_argument('-hidden_layers', '--hidden_layers', nargs='*', type=float, help='set the hidden layer dimensions for pred_net', default=[])
    parser.add_argument('-dropout_prob', help='dropout probability for training pred_net', type=float, default=0.0)
    parser.add_argument('-exp_name', type=str, help='experiment name to be concatenated with model name', default='') # do not add '_' at the start
    parser.add_argument('-save_path', type=str, help='path to dir to save pred_net parameters', default='saved_models/')
    parser.add_argument('-up_policy_path', type=str, help='path to dir where up policy is saved', default='data/saved_models/')
    parser.add_argument('-up_prefix', type=str, help='prefix used to load up policy', default='')
    parser.add_argument('-up_suffix', type=str, help='suffix used to load up policy', default='')
    parser.add_argument('-contextdir', type=str, help='dir for storing/loading context data', default='context_data/')
    parser.add_argument('-max_epochs', type=int, help='max number of epochs in each dagger round', default=50)
    parser.add_argument('-dagger_rounds', help='number of dagger rounds', type=int, default=6)
    parser.add_argument('-data_retain_frac', help='fraction of data to retain from previous rounds', type=float, default=1.0)
    parser.add_argument('-n_train_samples', type=int, help='number of contexts to sample for training', default=100)
    parser.add_argument('-n_train_steps', help='number of environment steps per round', type=int, default=20000)
    parser.add_argument('-action_noise', help='noise added to action', type=float, default=0.0)
    parser.add_argument('-device', type=str, help='device to load agents on', default='cpu')
    args = parser.parse_args()

    # extract information from args
    # setup the environments and initialize basic variables
    carl_env_fn = eval(args.carl_env_name)
    context_labels = args.context_labels
    DEFAULT_CONTEXT = carl_env_fn.get_default_context()
    current_dir = os.getcwd()

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

    labels_str = ""
    for label in context_labels:
        labels_str = labels_str + "_" + label

    label_scale_factor = np.ones((1, len(context_labels)))
    for idx in range(len(context_labels)):
        if context_mean[idx] != 0.0:
            label_scale_factor[0, idx] = context_mean[idx]
        else:
            label_scale_factor[0, idx] = 1.0

    ## Load training contexts
    train_context_array = np.zeros((len(context_mean), args.n_train_samples))

    split_idx = args.up_prefix.find('_')
    env_prefix = args.up_prefix[:split_idx]
    labels_str = ""
    for label in context_labels:
        labels_str = labels_str + "_" + label
    load_path = env_prefix + labels_str + "_contexts_up_" + str(args.n_train_samples) + ".npy"
    load_path = os.path.join(current_dir, args.contextdir, load_path)
    print('contexts_up load_path={0}'.format(load_path))
    train_context_array = np.load(load_path)
    print('train_context_array.shape = ', train_context_array.shape)
    
    train_context_dict = {}
    for i in range(args.n_train_samples):
        train_context_dict[i] = {key:value for key,value in zip(context_labels, train_context_array[:,i])}
        if i<5:
            print("train_context_dict[{0}] = {1}".format(i, train_context_dict[i]))

    print('len(train_context_dict) = ', len(train_context_dict))

    ## Load UP policy
    policy_name = args.up_prefix+labels_str+"_up"
    if args.up_suffix != '':
        policy_name = f'{policy_name}_{args.up_suffix}'

    policy_path = os.path.join(current_dir, args.up_policy_path, policy_name)
    print(f'policy_path={policy_path}')
    up_policy = load_policy(policy_path, args.device)

    ## Defining the LSTM context predictor
    tenv = init_carl(carl_env_fn)
    obs_len = tenv.observation_space.shape[0]
    if tenv.action_space.shape ==():
        act_len = 1
    else:
        act_len = tenv.action_space.shape[0]
    state_len = (args.stack_height+2)*obs_len + (args.stack_height+1)*act_len # for stack_height = 4
    
    if args.device=='cuda' and torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    hidden_layers = [int(e) for e in args.hidden_layers]
    pred_net = RecurrentContextPredictor(
        state_len, args.recurrent_hidden_size, hidden_layers, len(context_labels), 
        model_arch=args.model_arch, dropout_prob=args.dropout_prob, device=args.device
    )
    print(f'pred_net={pred_net}')
    for name, param in pred_net.named_parameters():
        if param.requires_grad:
            print(f'name={name},data={param.data.shape}')

    ## Defining optimizer
    optimizer = LstmOptimizer()

    ## Defining environments for training
    env_upc = init_carl(carl_env_fn, 
                        contexts=train_context_dict, 
                        obs_context_features=context_labels, 
                        hide_context=False, 
                        context_selector=RoundRobinSelector)
    env_hist = LstmObservationAction(
        init_carl(carl_env_fn, 
                  contexts=train_context_dict, 
                  obs_context_features=context_labels, 
                  hide_context=True, 
                  context_selector=RoundRobinSelector), 
        stack_size=args.stack_height)
    env_uppred = PredEnvWrapper(env_hist, pred_net, net_arch=args.model_arch, stack_size=args.stack_height, 
                                context_dim=len(context_labels), context_scale=label_scale_factor)
    
    ## Training loop
    features_list_train = []
    labels_list_train = []
    features_list_val = []
    labels_list_val = []
    pred_name = args.up_prefix + labels_str + f"_up_{args.model_arch}_predictor"
    if args.exp_name != '':
        pred_name = f'{pred_name}_{args.exp_name}'
    for round in range(args.dagger_rounds):
        print(f'------------- Round {round} ----------------')
        # load last round's pred_net params
        if round > 0:
            pred_net_path = os.path.join(current_dir, args.save_path, f'{pred_name}_dag_{round-1}.pt')
            print(f'pred_net_path = {pred_net_path}')
            if args.device == 'cpu' or not torch.cuda.is_available():
                pred_net.load_state_dict(torch.load(pred_net_path, map_location=torch.device('cpu')))
            else: 
                pred_net.load_state_dict(torch.load(pred_net_path))
        updater = Adam(pred_net.parameters(), lr=0.001, betas=(0.9,0.999))
        assert pred_net.state_dict().__str__() == env_uppred.pred_net.state_dict().__str__(), 'pred_net and env_uppred.pred_net are not synchronised.'
        
        # collect data
        new_features_list = []
        new_labels_list = []
        env_to_use = env_upc
        collect_fn = collect_timesteps_up
        if round > 0:
            env_to_use = env_uppred
            collect_fn = collect_timesteps_uppred
        # collect_fn(env_to_use, up_policy, new_features_list, new_labels_list, nsteps=args.n_train_steps)
        collect_fn(env_to_use, up_policy, new_features_list, new_labels_list, nsteps=args.n_train_steps, action_noise=args.action_noise)

        # delete a fraction train_list and val_list
        if round > 0:
            split_idx = int(args.data_retain_frac*len(features_list_train))
            features_list_train = features_list_train[:split_idx]
            labels_list_train = labels_list_train[:split_idx]
            split_idx = int(args.data_retain_frac*len(features_list_val))
            features_list_val = features_list_val[:split_idx]
            labels_list_val = labels_list_val[:split_idx]
            print(f'(after deletion) len(features_list_train)={len(features_list_train)}, len(features_list_val)={len(features_list_val)}')
        # permute and add data to train_list and val_list
        print(f'len(new_features_list)={len(new_features_list)}')
        split_idx = int(0.8*len(new_features_list))
        features_list = features_list_train + new_features_list[:split_idx]
        labels_list = labels_list_train + new_labels_list[:split_idx]
        index_arr = np.arange(len(features_list))
        index_arr = np.random.permutation(index_arr)
        features_list_train = [features_list[i] for i in index_arr]
        labels_list_train = [labels_list[i] for i in index_arr]
        features_list = features_list_val + new_features_list[split_idx:]
        labels_list = labels_list_val + new_labels_list[split_idx:]
        index_arr = np.arange(len(features_list))
        index_arr = np.random.permutation(index_arr)
        features_list_val = [features_list[i] for i in index_arr]
        labels_list_val = [labels_list[i] for i in index_arr]
        print(f'len(features_list_train)={len(features_list_train)}, len(features_list_val)={len(features_list_val)}')
        print(f'features_list_train[0].shape={features_list_train[0].shape}')
        
        # update pred_net model
        optimizer.fit_data(
            pred_net,
            features_list_train,
            labels_list_train,
            features_list_val,
            labels_list_val,
            updater,
            label_scale_factor,
            args.stack_height,
            obs_len,
            batchsize=2,
            num_epochs=args.max_epochs,
            max_patience=3
        )
        best_pred_net = optimizer.pred_net
        # save pred_net model
        pred_path = os.path.join(current_dir, args.save_path, f'{pred_name}_dag_{round}.pt')
        print(f"save path: pred_net: {pred_path}")
        os.makedirs(os.path.dirname(pred_path), exist_ok=True)
        torch.save(best_pred_net.state_dict(), pred_path)
        print("Saved best_pred_net")
        
if __name__ == '__main__':
    try:
        global_start = time.time()
        main()
        global_end = time.time()
        print(f"Total execution time in minutes: {(global_end-global_start)/60}.")
    except:
        print(traceback.format_exc())

    print("Completed execution.")