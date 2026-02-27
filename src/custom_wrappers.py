from collections import deque
from copy import deepcopy
from typing import Any, Iterable

import gym
import gymnasium
from gymnasium.core import ActType, ObsType, WrapperActType, WrapperObsType
# from gymnasium.wrappers.utils import create_zero_array
import numpy as np

# from lstm_functions import eng_feature, stack_observations

class ToGymnasiumActionSpace(gymnasium.ActionWrapper):
    '''Convert gym action space to gymnasium action space'''
    def __init__(self, env):
        if not isinstance(env.action_space, (gym.spaces.box.Box)):
            raise ValueError(f"Found action_space {env.action_space} which is unsupported.")

        gymnasium.ActionWrapper.__init__(self, env)
        if isinstance(env.action_space, gym.spaces.box.Box):
            self.action_space = gymnasium.spaces.box.Box(
                low=env.action_space.low,
                high=env.action_space.high,
                shape=env.action_space.shape,
                dtype=env.action_space.dtype,
            )

    def action(self, action):
        return action

class ToGymActionSpace(gym.ActionWrapper):
    '''Convert gymnasium action space to gym action space'''
    def __init__(self, env):
        # if not isinstance(env.action_space, (gymnasium.spaces.box.Box)):
        #     raise ValueError(f"Found action_space {env.action_space} which is unsupported.")
        # print('isinstance(env.act_space, (gymn.spaces.box.Box)) =', 
        #       isinstance(env.action_space, (gymnasium.spaces.box.Box)))
        supported_spaces = (gymnasium.spaces.box.Box, gymnasium.spaces.discrete.Discrete)
        assert isinstance(env.action_space, supported_spaces)

        gym.ActionWrapper.__init__(self, env)
        if isinstance(env.action_space, gymnasium.spaces.box.Box):
            self.action_space = gym.spaces.box.Box(
                low=env.action_space.low,
                high=env.action_space.high,
                shape=env.action_space.shape,
                dtype=env.action_space.dtype,
            )
        elif isinstance(env.action_space, gymnasium.spaces.discrete.Discrete):
            maximum = env.action_space.n
            self.action_space = gym.spaces.discrete.Discrete(maximum)

    def action(self, action):
        return action

class ToGymObservationSpace(gym.ObservationWrapper):
    '''Convert gymnasium observation space to gym observation space'''
    def __init__(self, env):
        # print('isinstance(env.obs_space, (gymn.spaces.box.Box)) =', 
        #       isinstance(env.observation_space, (gymnasium.spaces.box.Box)))
        assert isinstance(env.observation_space, gymnasium.spaces.box.Box)

        gym.ActionWrapper.__init__(self, env)
        if isinstance(env.observation_space, gymnasium.spaces.box.Box):
            self.observation_space = gym.spaces.box.Box(
                low=env.observation_space.low,
                high=env.observation_space.high,
                shape=env.observation_space.shape,
                dtype=env.observation_space.dtype,
            )
    
    def observation(self, observation):
        return observation

def concat_space(space: gymnasium.spaces.Space, n: int = 1):
    supported_spaces = (gym.spaces.box.Box, gymnasium.spaces.box.Box, gymnasium.spaces.discrete.Discrete)
    assert isinstance(space, supported_spaces)
    if isinstance(space, (gym.spaces.box.Box, gymnasium.spaces.box.Box)):
        low, high = np.tile(space.low, n), np.tile(space.high, n)
        return gymnasium.spaces.box.Box(low=low, high=high, dtype=space.dtype, seed=deepcopy(space.np_random))
    elif isinstance(space, gymnasium.spaces.discrete.Discrete):
        # maximum = np.tile(space.n, n)
        low, high = np.tile(0, n), np.tile(space.n-1, n)
        return gymnasium.spaces.box.Box(low=low, high=high, dtype=np.float32, seed=deepcopy(space.np_random))
    else:
        raise ValueError(f'Found unsupported space, space={space}.')
        # return gymnasium.spaces.MultiDiscrete(maximum, seed=deepcopy(space.np_random))

def space_specs(space):
    box_spaces = (gym.spaces.box.Box, gymnasium.spaces.box.Box)
    discrete_spaces = (gym.spaces.discrete.Discrete, gymnasium.spaces.discrete.Discrete)
    if isinstance(space, box_spaces):
        space_low = space.low
        space_high = space.high
        space_dtype = space.dtype
    elif isinstance(space, discrete_spaces):
        space_low = [0]
        space_high = [space.n]
        space_dtype = np.float32
    else:
        raise TypeError
    return space_low, space_high, space_dtype

def joint_space(space1: gymnasium.spaces.Space, space2: gymnasium.spaces.Space):
    '''Returns the joint space [space1, space2]'''
    supported_spaces = (gym.spaces.box.Box, gymnasium.spaces.box.Box, 
                        gym.spaces.discrete.Discrete, gymnasium.spaces.discrete.Discrete)
    assert isinstance(space1, supported_spaces)
    assert isinstance(space2, supported_spaces)
    space1_low, space1_high, space1_dtype = space_specs(space1)
    space2_low, space2_high, space2_dtype = space_specs(space2)
    joint_low = np.concatenate((space1_low, space2_low))
    joint_high = np.concatenate((space1_high, space2_high))
    if space1_dtype == space2_dtype == np.float32:
        joint_dtype = np.float32
    else:
        raise ValueError(f'Unable to define joint_dtype. Found space1_dtype={space1_dtype}, space2_dtype={space2_dtype}.')
    joint_space = gymnasium.spaces.box.Box(low=joint_low, high=joint_high, dtype=joint_dtype)
    return joint_space

def create_zero_array(space: gymnasium.spaces.Space):
    box_spaces = (gym.spaces.box.Box, gymnasium.spaces.box.Box)
    discrete_spaces = (gym.spaces.discrete.Discrete, gymnasium.spaces.discrete.Discrete)
    if isinstance(space, box_spaces):
        space_shape = space.shape
    elif isinstance(space, discrete_spaces):
        # print(f'Found discrete space. space.shape={space.shape}')
        space_shape = (1,)
    else:
        raise TypeError
    zero_array = np.zeros(space_shape, dtype=space.dtype)
    # zero_array = np.where(space.low > 0, space.low, zero_array)
    # zero_array = np.where(space.high < 0, space.high, zero_array)
    return zero_array

def create_empty_array(space: gymnasium.spaces.Space, n: int = 1, fn=np.zeros) -> np.ndarray:
    box_spaces = (gym.spaces.box.Box, gymnasium.spaces.box.Box)
    discrete_spaces = (gym.spaces.discrete.Discrete, gymnasium.spaces.discrete.Discrete)
    if isinstance(space, box_spaces):
        space_shape = space.shape
        space_dtype = space.dtype
    elif isinstance(space, discrete_spaces):
        space_shape = (1,)
        space_dtype = np.float32
    else:
        raise TypeError
    return fn((n,) + space_shape, dtype=space_dtype)

def concatenate(
    space: gymnasium.spaces.box.Box,
    items: Iterable,
    out: np.ndarray,
) -> np.ndarray:
    return np.stack(items, axis=0, out=out)
    
class ConcatObservationAction(gymnasium.ObservationWrapper, gymnasium.utils.RecordConstructorArgs):
    '''
    Concatenate observation from the last ``n`` time steps and action from the last ``m`` time steps
    Based on gymnasium.wrappers.FrameStack
    '''
    def __init__(
        self,
        env,
        stack_size_obs:int,
        stack_size_act:int,
        # lz4_compress: bool = False,
    ):
        gymnasium.utils.RecordConstructorArgs.__init__(
            self, num_stack=stack_size_obs, lz4_compress=False
        )
        gymnasium.ObservationWrapper.__init__(self, env)

        self.stack_size_obs = stack_size_obs
        self.stack_size_act = stack_size_act
        # self.lz4_compress = lz4_compress
        # print(f'env.action_space={env.action_space}')
        self.padding_value_obs = create_zero_array(env.observation_space)
        self.padding_value_act = create_zero_array(env.action_space)
        self.obs_queue = deque(maxlen=stack_size_obs)
        self.act_queue = deque(maxlen=stack_size_act)

        # low = np.repeat(self.observation_space.low[np.newaxis, ...], num_stack, axis=0)
        # high = np.repeat(
        #     self.observation_space.high[np.newaxis, ...], num_stack, axis=0
        # )
        # self.observation_space = Box(
        #     low=low, high=high, dtype=self.observation_space.dtype
        # )
        obs_space = concat_space(env.observation_space, stack_size_obs)
        act_space = concat_space(env.action_space, stack_size_act)
        # print(f'obs_space = {obs_space}')
        # print(f'act_space = {act_space}')
        joint_low = np.concatenate((obs_space.low, act_space.low))
        joint_high = np.concatenate((obs_space.high, act_space.high))
        if obs_space.dtype == act_space.dtype == np.float32:
            joint_dtype = np.float32
        else:
            raise ValueError(f'Unable to define joint_dtype. Found obs_space.dtype={obs_space.dtype}, act_space.dtype={act_space.dtype}.')
        # print('joint_low =', joint_low)
        joint_space = gymnasium.spaces.box.Box(low=joint_low, high=joint_high, dtype=joint_dtype)
        self.observation_space = joint_space
        self.stacked_obs = create_empty_array(env.observation_space, n=self.stack_size_obs)
        self.stacked_act = create_empty_array(env.action_space, n=self.stack_size_act)
        # print('init: self.stacked_obs =', self.stacked_obs)
        # print('init: self.stacked_act =', self.stacked_act)
    # def observation(self, observation):
    #     """Converts the wrappers current frames to lazy frames.

    #     Args:
    #         observation: Ignored

    #     Returns:
    #         :class:`LazyFrames` object for the wrapper's frame buffer,  :attr:`self.frames`
    #     """
    #     assert len(self.frames) == self.num_stack, (len(self.frames), self.num_stack)
    #     return LazyFrames(list(self.frames), self.lz4_compress)

    def step(self, action):
        """Steps through the environment, appending the observation to the frame buffer.

        Args:
            action: The action to step through the environment with

        Returns:
            Stacked observations, reward, terminated, truncated, and information from the environment
        """
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.obs_queue.appendleft(observation)
        if isinstance(self.env.action_space, (gym.spaces.discrete.Discrete, gymnasium.spaces.discrete.Discrete)):
            # print(f'action = {action}')
            action = [action]
        self.act_queue.appendleft(action)
        concat_obs = deepcopy(
            concatenate(self.env.observation_space, self.obs_queue, self.stacked_obs)
        )
        # print(f'obs_queue = {self.obs_queue}, concat_obs = {concat_obs.flatten()}')
        # print(f'concat_obs = {concat_obs.flatten()}')
        concat_act = deepcopy(
            concatenate(self.env.action_space, self.act_queue, self.stacked_act)
        )
        # print(f'act_queue = {self.act_queue}, concat_act = {concat_act.flatten()}')
        # print(f'concat_act = {concat_act.flatten()}')
        updated_obs = np.concatenate((concat_obs.flatten(), concat_act.flatten()))
        return updated_obs, reward, terminated, truncated, info
        # return self.observation(None), reward, terminated, truncated, info

    def reset(self, **kwargs):
        """Reset the environment with kwargs.

        Args:
            **kwargs: The kwargs for the environment reset

        Returns:
            The stacked observations
        """
        obs, info = self.env.reset(**kwargs)
        # print('(Before) self.obs_queue =', self.obs_queue)
        # print('(Before) self.act_queue =', self.act_queue)
        for _ in range(self.stack_size_obs - 1):
            self.obs_queue.appendleft(self.padding_value_obs)
        for _ in range(self.stack_size_act):
            self.act_queue.appendleft(self.padding_value_act)
        self.obs_queue.appendleft(obs)
        # print('(After) self.obs_queue =', self.obs_queue)
        # print('(After) self.act_queue =', self.act_queue)
        
        concat_obs = deepcopy(
            concatenate(self.env.observation_space, self.obs_queue, self.stacked_obs)
        )
        
        concat_act = deepcopy(
            concatenate(self.env.action_space, self.act_queue, self.stacked_act)
        )
        updated_obs = np.concatenate((concat_obs.flatten(), concat_act.flatten()))
        return updated_obs, info

class LstmObservationAction(gymnasium.ObservationWrapper, gymnasium.utils.RecordConstructorArgs):
    '''
    Similar to ConcatObservationAction. This function interleaves the observations and actions i.e.
    the final observation will be something like [s_t, s_{t-1}, a_{t-1}, s_{t-2}, ...].
    '''
    def __init__(
        self,
        env,
        stack_size:int,
        # lz4_compress: bool = False,
    ):
        gymnasium.utils.RecordConstructorArgs.__init__(
            self, num_stack=stack_size, lz4_compress=False
        )
        gymnasium.ObservationWrapper.__init__(self, env)

        self.stack_size = stack_size
        # self.stack_size_act = stack_size_act
        # self.lz4_compress = lz4_compress

        self.padding_value_obs = create_zero_array(env.observation_space)
        self.padding_value_act = create_zero_array(env.action_space)
        self.obs_queue = deque(maxlen=stack_size+2)
        self.act_queue = deque(maxlen=stack_size+1)

        # print(f'obs_space={env.observation_space}, act_space={env.action_space}')
        obs_act_space = joint_space(env.observation_space, env.action_space)
        obs_act_stack = concat_space(obs_act_space, stack_size+1)
        feature_space = joint_space(env.observation_space, obs_act_stack)
        self.observation_space = feature_space
        self.stacked_obs = create_empty_array(env.observation_space, n=self.stack_size+2)
        self.stacked_act = create_empty_array(env.action_space, n=self.stack_size+1)

    def step(self, action):
        """Steps through the environment, appending the observation to the frame buffer.

        Args:
            action: The action to step through the environment with

        Returns:
            Stacked observations, reward, terminated, truncated, and information from the environment
        """
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.obs_queue.appendleft(observation)
        if isinstance(self.env.action_space, (gym.spaces.discrete.Discrete, gymnasium.spaces.discrete.Discrete)):
            action = [action]
        # print(f'action={action}')
        self.act_queue.appendleft(action)
        # self.act_queue.appendleft(self.padding_value_act)
        concat_obs = deepcopy(
            concatenate(self.env.observation_space, self.obs_queue, self.stacked_obs)
        )
        
        concat_act = deepcopy(
            concatenate(self.env.action_space, self.act_queue, self.stacked_act)
        )
        features = np.concatenate((concat_obs[1:,:], concat_act), axis=1)
        # if features.ndim < 3:
        #     features = np.expand_dims(features, axis=0)
        updated_obs = np.concatenate((concat_obs[0,:], features.flatten()))
        # updated_obs = stack_observations(features, obs_len=observation.shape[0], stack_height=self.stack_size)
        # updated_obs = updated_obs[0, -1, :]
        # updated_obs = np.concatenate((concat_obs.flatten(), concat_act.flatten()))
        return updated_obs, reward, terminated, truncated, info
        # return self.observation(None), reward, terminated, truncated, info

    def reset(self, **kwargs):
        """Reset the environment with kwargs.

        Args:
            **kwargs: The kwargs for the environment reset

        Returns:
            The stacked observations
        """
        # print(f'LstmObservationAction.reset: kwargs = {kwargs}')
        obs, info = self.env.reset(**kwargs)
        # print('(Before) self.obs_queue =', self.obs_queue)
        # print('(Before) self.act_queue =', self.act_queue)
        for _ in range(self.stack_size + 2):
            self.obs_queue.appendleft(self.padding_value_obs)
        for _ in range(self.stack_size + 1):
            self.act_queue.appendleft(self.padding_value_act)
        self.obs_queue.appendleft(obs)
        # print('(After) self.obs_queue =', self.obs_queue)
        # print('(After) self.act_queue =', self.act_queue)
        # print('self.env.obs_space =', self.env.observation_space)
        # print('self.stacked_obs.shape =', self.stacked_obs.shape)
        concat_obs = deepcopy(
            concatenate(self.env.observation_space, self.obs_queue, self.stacked_obs)
        )
        
        concat_act = deepcopy(
            concatenate(self.env.action_space, self.act_queue, self.stacked_act)
        )
        # print(f'concat_obs[1:,:]={concat_obs[1:,:]}, concat_act={concat_act}')
        features = np.concatenate((concat_obs[1:,:], concat_act), axis=1)
        # if features.ndim < 3:
        #     features = np.expand_dims(features, axis=0)
        # updated_obs = stack_observations(features, obs_len=obs.shape[0], stack_height=self.stack_size)
        updated_obs = np.concatenate((concat_obs[0], features.flatten()))
        # updated_obs = updated_obs[0, -1, :]
        # updated_obs = np.concatenate((concat_obs.flatten(), concat_act.flatten()))
        return updated_obs, info

class StartStateWrapper(gymnasium.Wrapper):
    def __init__(self, env: gymnasium.Env, start_state):
        super().__init__(env)
        self.start_state = start_state

    # def reset(
    #     self, *, seed: int | None = None, options: dict[str, Any] | None = None
    # ) -> tuple[WrapperObsType, dict[str, Any]]:
    def reset(
        self, *, seed = None, options = None
    ) -> tuple[WrapperObsType, dict[str, Any]]:
        """Resets the environment and sets autoreset to False preventing."""
        # print(f'StartStateWrapper.reset: self.start_state = {self.start_state}')
        # try:
        #     return self.env.reset(options=self.start_state)
        # except:
        #     return self.wrapped_env.reset(options=self.start_state)
        return self.env.reset(options=self.start_state)

    def set_start_state(self, new_start_state):
        self.start_state = new_start_state