import gym
import numpy as np
from gym import error, spaces


class OSIEnvWrapper(gym.ObservationWrapper):
    def __init__(self, env, osi, osi_hist, up_dim, label_scale_factor=None):
        self.wrapped_env = env
        self.env = env.unwrapped # skip a wrapper for retaining other apis
        self.osi = osi
        self.osi_hist = osi_hist
        self.up_dim = up_dim
        self.label_scale_factor = label_scale_factor

        high = np.inf * np.ones(int(self.wrapped_env.observation_space.shape[0] / osi_hist + up_dim))
        low = -high
        self.observation_space = spaces.Box(low, high)

    def process_raw_obs(self, raw_o):
        if len(self.env.action_space.shape) == 0:
            action_space_len = 1
        else:
            action_space_len = self.env.action_space.shape[0]
        one_obs_len = int((len(raw_o) - action_space_len * self.wrapped_env.stack_size_act) / self.wrapped_env.stack_size_obs)
        pred_mu = self.osi.predict(raw_o)[0]
        if self.label_scale_factor is not None:
            pred_mu = self.label_scale_factor*pred_mu
        return np.concatenate([pred_mu, raw_o[0:one_obs_len]])

    def step(self, a):
        raw_o, r, d, info = self.wrapped_env.step(a)

        return self.process_raw_obs(raw_o), r, d, info

    def reset(self):
        raw_o, info = self.wrapped_env.reset()
        return self.process_raw_obs(raw_o), info