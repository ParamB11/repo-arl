import gymnasium
from gymnasium import spaces
import numpy as np
import torch

class LstmEnvWrapper(gymnasium.ObservationWrapper):
    def __init__(self, env, lstm, stack_size, context_dim, context_scale):
        self.wrapped_env = env
        self.env = env.unwrapped # skip a wrapper for retaining other apis
        self.lstm = lstm
        self.ht = None
        self.ct = None
        self.stack_size = stack_size
        self.context_dim = context_dim
        if context_scale.ndim > 1:
            self.context_scale = context_scale.flatten()
        else:
            self.context_scale = context_scale

        high = np.inf * np.ones(int(context_dim + self.env.observation_space.shape[0]))
        low = -high
        self.observation_space = spaces.Box(low, high)

    def process_raw_obs(self, raw_o):
        if len(self.env.action_space.shape) == 0:
            action_space_len = 1
        else:
            action_space_len = self.env.action_space.shape[0]
        one_obs_len = self.env.observation_space.shape[0]
        raw_o_tensor = torch.from_numpy(raw_o).to(dtype=torch.float32)
        raw_o_tensor = raw_o_tensor.expand(1,1,raw_o.shape[0])
        pred_mu, (new_ht, new_ct) = self.lstm(raw_o_tensor, self.ht, self.ct)
        pred_mu = pred_mu.cpu().detach().numpy()[0,0]*self.context_scale
        self.ht, self.ct = new_ht, new_ct
        return np.concatenate([pred_mu, raw_o[0:one_obs_len]])

    def step(self, a):
        raw_o, r, tm, tu, info = self.wrapped_env.step(a)

        return self.process_raw_obs(raw_o), r, tm, tu, info

    def reset(self):
        self.ht, self_ct = None, None
        raw_o, info = self.wrapped_env.reset()
        return self.process_raw_obs(raw_o), info

valid_archs = np.array(['gru', 'lstm'])
class PredEnvWrapper(gymnasium.ObservationWrapper):
    def __init__(self, env, pred_net, net_arch, stack_size, context_dim, context_scale):
        self.wrapped_env = env
        self.env = env.unwrapped # skip a wrapper for retaining other apis
        self.pred_net = pred_net # self.lstm = lstm
        assert np.any(valid_archs == net_arch), f'net_arch={net_arch} not in valid_archs={valid_archs}.'
        self.net_arch = net_arch
        self.ht = None
        self.ct = None
        self.stack_size = stack_size
        self.context_dim = context_dim
        if context_scale.ndim > 1:
            self.context_scale = context_scale.flatten()
        else:
            self.context_scale = context_scale

        high = np.inf * np.ones(int(context_dim + self.env.observation_space.shape[0]))
        low = -high
        self.observation_space = spaces.Box(low, high)

    def process_raw_obs(self, raw_o):
        if len(self.env.action_space.shape) == 0:
            action_space_len = 1
        else:
            action_space_len = self.env.action_space.shape[0]
        one_obs_len = self.env.observation_space.shape[0]
        raw_o_tensor = torch.from_numpy(raw_o).to(dtype=torch.float32)
        raw_o_tensor = raw_o_tensor.expand(1,1,raw_o.shape[0])
        if self.net_arch == 'lstm':
            pred_mu, (new_ht, new_ct) = self.pred_net(raw_o_tensor, self.ht, self.ct)
            self.ht, self.ct = new_ht, new_ct
        elif self.net_arch == 'gru':
            pred_mu, new_ht = self.pred_net(raw_o_tensor, self.ht)
            self.ht = new_ht
        pred_mu = pred_mu.cpu().detach().numpy()[0,0]*self.context_scale
        return np.concatenate([pred_mu, raw_o[0:one_obs_len]])

    def step(self, a):
        raw_o, r, tm, tu, info = self.wrapped_env.step(a)

        return self.process_raw_obs(raw_o), r, tm, tu, info

    def reset(self, **kwargs):
        self.ht, self.ct = None, None
        raw_o, info = self.wrapped_env.reset(**kwargs)
        return self.process_raw_obs(raw_o), info

class InformerEnvWrapper(gymnasium.ObservationWrapper):
    def __init__(self, env, pred_net, stack_size, context_dim, context_scale, pred_len, label_len):
        self.wrapped_env = env
        self.env = env.unwrapped # skip a wrapper for retaining other apis
        self.pred_net = pred_net
        self.label_len = label_len
        self.pred_len = pred_len
        self.dec_start_token = torch.zeros([1, self.label_len+self.pred_len, context_dim]).float()
        self.stack_size = stack_size
        self.context_dim = context_dim
        if context_scale.ndim > 1:
            self.context_scale = context_scale.flatten()
        else:
            self.context_scale = context_scale

        high = np.inf * np.ones(int(context_dim + self.env.observation_space.shape[0]))
        low = -high
        self.observation_space = spaces.Box(low, high)

    def process_raw_obs(self, raw_o):
        if len(self.env.action_space.shape) == 0:
            action_space_len = 1
        else:
            action_space_len = self.env.action_space.shape[0]
        obs_stack = np.vstack(self.wrapped_env.obs_queue)
        act_stack = np.vstack(self.wrapped_env.act_queue)
        feature_stack = np.expand_dims(np.hstack((obs_stack, act_stack)), axis=0)
        one_obs_len = self.env.observation_space.shape[0]
        feature_tensor = torch.from_numpy(feature_stack).to(dtype=torch.float32)
        y_pred = self.pred_net(feature_tensor, self.dec_start_token)
        self.dec_start_token = torch.cat((y_pred, self.dec_start_token[:,:-2,:], self.dec_start_token[:,-1:,:]), dim=1)
        y_pred = y_pred.cpu().detach().numpy()[0,0]*self.context_scale
        return np.concatenate([y_pred, raw_o[0:one_obs_len]])

    def step(self, a):
        raw_o, r, tm, tu, info = self.wrapped_env.step(a)

        return self.process_raw_obs(raw_o), r, tm, tu, info

    def reset(self):
        self.dec_start_token = torch.zeros([1, self.label_len+self.pred_len, self.context_dim]).float()
        raw_o, info = self.wrapped_env.reset()
        return self.process_raw_obs(raw_o), info