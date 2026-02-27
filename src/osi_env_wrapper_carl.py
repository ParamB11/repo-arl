import gym
import numpy as np
from gym import error, spaces


class OSIEnvWrapper(gym.ObservationWrapper):
    def __init__(self, env, osi, osi_hist, up_dim, label_scale_factor=None):
        self.wrapped_env = env
        # print('wrapped_env.stack_size_obs = {0}, wrapped_env.stack_size_act = {1}'
        #      .format(self.wrapped_env.stack_size_obs, self.wrapped_env.stack_size_act))
        self.env = env.unwrapped # skip a wrapper for retaining other apis
        self.osi = osi
        self.osi_hist = osi_hist
        self.up_dim = up_dim
        self.label_scale_factor = label_scale_factor

        high = np.inf * np.ones(int(self.wrapped_env.observation_space.shape[0] / osi_hist + up_dim)) ## 03/01/2025: Maybe needs to be updated
        low = -high
        self.observation_space = spaces.Box(low, high)

        # self.action_space = env.action_space

    def process_raw_obs(self, raw_o):
        if len(self.env.action_space.shape) == 0:
            action_space_len = 1
        else:
            action_space_len = self.env.action_space.shape[0]
        one_obs_len = int((len(raw_o) - action_space_len * self.wrapped_env.stack_size_act) / self.wrapped_env.stack_size_obs)
        pred_mu = self.osi.predict(raw_o)[0]
        if self.label_scale_factor is not None:
            # print(f'OSIEnvWrapper.process_raw_obs: (before) pred_mu={pred_mu}')
            pred_mu = self.label_scale_factor*pred_mu
            # print(f'OSIEnvWrapper.process_raw_obs: (after) pred_mu={pred_mu}')
        return np.concatenate([pred_mu, raw_o[0:one_obs_len]])

    def step(self, a):
        raw_o, r, d, info = self.wrapped_env.step(a)

        return self.process_raw_obs(raw_o), r, d, info

    def reset(self):
        raw_o, info = self.wrapped_env.reset()
        return self.process_raw_obs(raw_o), info

# class OSIEnvWrapper:
#     def __init__(self, env, osi, osi_hist, up_dim):
#         self.wrapped_env = env
#         self.env = env.unwrapped # skip a wrapper for retaining other apis
#         self.osi = osi
#         self.osi_hist = osi_hist
#         self.up_dim = up_dim

#         high = np.inf * np.ones(int(self.env.obs_dim / osi_hist + up_dim))
#         low = -high
#         self.observation_space = spaces.Box(low, high)

#         self.action_space = env.action_space

#     def process_raw_obs(self, raw_o):
#         one_obs_len = int((len(raw_o) - self.env.action_space.shape[0] * self.osi_hist) / self.osi_hist)
#         pred_mu = self.osi.predict(raw_o)[0]
#         return np.concatenate([pred_mu, raw_o[0:one_obs_len]])

#     def step(self, a):
#         raw_o, r, d, dict = self.wrapped_env.step(a)

#         return self.process_raw_obs(raw_o), r, d, dict

#     def reset(self):
#         raw_o = self.wrapped_env.reset()
#         return self.process_raw_obs(raw_o)

#     def render(self):
#         return self.wrapped_env.render()

#     @property
#     def seed(self):
#         return self.env.seed

#     def pad_action(self, a):
#         return self.env.pad_action(a)

#     def unpad_action(self, a):
#         return self.env.unpad_action(a)

#     def about_to_contact(self):
#         return self.env.about_to_contact()

#     def state_vector(self):
#         return self.env.state_vector()

#     def set_state_vector(self, s):
#         self.env.set_state_vector(s)

#     def set_sim_parameters(self, pm):
#         self.env.set_sim_parameters(pm)

#     def get_sim_parameters(self):
#         return self.env.get_sim_parameters()