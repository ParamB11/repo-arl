from itertools import combinations
import joblib
import os
import sys
import time
sys.path.append('./src')

from baselines.common import tf_util as U
from carl.context.selection import StaticSelector
# from carl.envs import CARLAcrobot, CARLBipedalWalker, CARLBraxAnt, CARLBraxHalfcheetah, CARLCartPole, CARLLunarLander, CARLMountainCar, CARLMountainCarContinuous, CARLPendulum
import gym
from gymnasium.wrappers import FlattenObservation, FilterObservation, StepAPICompatibility
import numpy as np
from stable_baselines3 import DDPG, DQN, PPO, SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback
import tensorflow as tf
import torch

from custom_wrappers import ConcatObservationAction, ToGymActionSpace, ToGymObservationSpace
from lstm_context_pred import GRUContextPredictor, LSTMContextPredictor
# from lstm_utils import eng_feature, stack_observations
from policy_transfer.utils.mlp import MLP
# from save_policy_params import save_policy_params

def gen_context_experts(context_mean, rel_std=0.25):
    context_l = len(context_mean)
    context_std = rel_std*np.array(context_mean)
    nominal_expert = np.array(context_mean).reshape(context_l,-1)
    expert1f = np.tile(nominal_expert, (1,2*context_l))
    comb2 = list(combinations(range(3), 2)) 
    expert2f = np.tile(nominal_expert, (1,4*len(comb2)))
    
    for i in range(expert1f.shape[1]):
        f_idx = int(i/2)
        expert1f[f_idx, i] = context_mean[f_idx] - pow(-1,i)*2*context_std[f_idx]
    # print('expert1f =', expert1f)
    for c_idx,i in enumerate(comb2):
        i1, i2 = i
        for j in range(4):
            expert2f[i1, 4*c_idx+j] = context_mean[i1] - pow(-1,int(j/2))*context_std[i1]
            expert2f[i2, 4*c_idx+j] = context_mean[i2] - pow(-1,j)*context_std[i2]
        # print('expert2f[:,{0}:{1}] = \n{2}\n'.format(4*c_idx, 4*(c_idx+1), expert2f[:,4*c_idx:4*(c_idx+1)]))
    context_experts = np.concatenate((nominal_expert, expert1f, expert2f), axis=1)
    return context_experts

def init_carl(carl_env_fn, contexts=None, obs_context_features=None, hide_context=True, context_selector=None):
    env = carl_env_fn(contexts=contexts, 
                      obs_context_features=obs_context_features, 
                      context_selector=context_selector)
    if hide_context:
        env = FlattenObservation(FilterObservation(env, filter_keys=["obs"]))
    else:
        env = FlattenObservation(FilterObservation(env, filter_keys=["obs", "context"]))
    return env

def load_lstm(carl_env_fn, context_labels, load_path, stack_height, eng_bool, device='cpu'):
    tenv = init_carl(carl_env_fn)
    obs_len = tenv.observation_space.shape[0]
    if tenv.action_space.shape ==():
        act_len = 1
    else:
        act_len = tenv.action_space.shape[0]
    if eng_bool:
        state_len = 6*obs_len + 2*act_len
    else:
        state_len = (stack_height+2)*obs_len + (stack_height+1)*act_len # for stack_height = 4
    pred_net = LSTMContextPredictor(state_len, 32, len(context_labels), device=device)
    # pred_net_path = os.path.join(current_dir, savedir, pred_name)
    # print(f'pred_net_path = {pred_net_path}')
    if device == 'cpu' or not torch.cuda.is_available():
        pred_net.load_state_dict(torch.load(load_path, map_location=torch.device('cpu')))
    else: 
        pred_net.load_state_dict(torch.load(load_path))
    return pred_net

def load_gru(carl_env_fn, context_labels, load_path, stack_height, eng_bool, device='cpu'):
    tenv = init_carl(carl_env_fn)
    obs_len = tenv.observation_space.shape[0]
    if tenv.action_space.shape ==():
        act_len = 1
    else:
        act_len = tenv.action_space.shape[0]
    if eng_bool:
        state_len = 6*obs_len + 2*act_len
    else:
        state_len = (stack_height+2)*obs_len + (stack_height+1)*act_len # for stack_height = 4
    pred_net = GRUContextPredictor(state_len, 32, len(context_labels), device=device)
    # pred_net_path = os.path.join(current_dir, savedir, pred_name)
    # print(f'pred_net_path = {pred_net_path}')
    if device == 'cpu' or not torch.cuda.is_available():
        pred_net.load_state_dict(torch.load(load_path, map_location=torch.device('cpu')))
    else: 
        pred_net.load_state_dict(torch.load(load_path))
    return pred_net
    

def to_gym(env):
    supported_spaces = (gym.spaces.box.Box, gym.spaces.discrete.Discrete)
    if not isinstance(env.observation_space, supported_spaces):
        env = ToGymObservationSpace(env)
    if not isinstance(env.action_space, supported_spaces):
        env = ToGymActionSpace(env)
    env = StepAPICompatibility(env, output_truncation_bool=False)
    return env

# def env_fn(carl_env_fn, context, OSI_hist, show_context=False):
#     tenv = init_carl(carl_env_fn, contexts=context, 
#                      obs_context_features=list(context[0].keys()),
#                      hide_context = not show_context,
#                      context_selector=StaticSelector,
#                     )
#     env_hist = to_gym(
#         ConcatObservationAction(tenv, stack_size_obs=OSI_hist, stack_size_act=OSI_hist)
#     )
#     return env_hist

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

def load_osi(carl_env_fn, context_labels, OSI_hist, osi_name, osi_path):
    context_zeros = np.zeros(len(context_labels))
    context_temp = {0:{k:v for k,v in zip(context_labels, context_zeros)}}
    env_hist = env_fn(carl_env_fn, context_temp, OSI_hist+1, OSI_hist, show_context=False)
    # print('load_osi: Initializing OSI ...')
    osi = MLP(name=osi_name, in_dim=env_hist.observation_space.shape[0], out_dim=len(context_labels), layers=[256, 128, 64],
              activation=tf.nn.relu, last_activation=None, dropout=0.1)
    sess = tf.compat.v1.InteractiveSession()
    U.initialize()
    # print('load_osi: Loading pretrained weights ... ')
    osi.set_variable_from_dict(joblib.load(osi_path))
    # print('load_osi: Returning pretrained OSI ...')
    # tf.compat.v1.InteractiveSession.close(sess)
    return osi, sess

def load_policy(path, device='cpu'):
    if path.find("ppo")!=-1:
        expert = PPO.load(path, env=None, device=device)
    if path.find("sac")!=-1:
        expert = SAC.load(path, env=None, device=device)
    elif path.find('dqn')!=-1:
        expert = DQN.load(path, env=None, device=device)
    elif path.find('td3')!=-1:
        expert = TD3.load(path, env=None, device=device)
    elif path.find('ddpg')!=-1:
        # if path.find('sb3')!=-1:
        expert = DDPG.load(path, env=None, device=device)
    return expert

def save_policy_params(load_path, device):
    print(f'load_path: {load_path}')
    policy = load_policy(load_path, device)
    save_path = f'{load_path}_params.pkl'
    policy.save(save_path)
    print(f'params save_path: {save_path}')

def load_policy_from_params(path, env, device='cpu'):
    if not os.path.exists(path):
        # Code to run if the file doesn't exist
        # print("File doesn't exist. Waiting for 2 mins.")
        # time.sleep(120)
        # if not os.path.exists(path):
        print("File doesn't exist, creating it now. Calling save_policy_params()")
        load_path = path.replace('_params.pkl', '')
        save_policy_params(load_path, device)
    if path.find("ppo")!=-1:
        # expert = PPO.load(path, env=None, device=device)
        policy = PPO("MlpPolicy", env, device=device)
        policy.set_parameters(path)
    if path.find("sac")!=-1:
        # expert = SAC.load(path, env=None, device=device)
        policy = SAC("MlpPolicy", env, device=device)
        policy.set_parameters(path)
    elif path.find('dqn')!=-1:
        # expert = DQN.load(path, env=None, device=device)
        policy = DQN("MlpPolicy", env, device=device)
        policy.set_parameters(path)
    elif path.find('td3')!=-1:
        # expert = TD3.load(path, env=None, device=device)
        policy = TD3("MlpPolicy", env, device=device)
        policy.set_parameters(path)
    elif path.find('ddpg')!=-1:
        # if path.find('sb3')!=-1:
        policy = DDPG("MlpPolicy", env, device=device)
        policy.set_parameters(path)
    return policy
    
# def make_carl_fn(env_name:str):
#     return eval(env_name)

class SaveCallback(BaseCallback):
    def __init__(
        self,
        save_path: str,
        name_prefix:str = "rl_model",
        chckpt_bool:bool = False,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.chckpt_bool = chckpt_bool
    
    def _on_step(self) -> bool:
        model_path = os.path.join(self.save_path, f"{self.name_prefix}.zip")
        
        self.model.save(model_path)
        if self.verbose >= 1:
            print(f"Saving model to {model_path}.")
        if self.chckpt_bool:
            checkpoint_path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps.zip")
            self.model.save(checkpoint_path)
            if self.verbose >= 1:
                print(f"Checkpointing model at {checkpoint_path}.")
        return True