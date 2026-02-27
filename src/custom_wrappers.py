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


## ConcatObservationAction2 is not needed
# class ConcatObservationAction2(gymnasium.ObservationWrapper, gymnasium.utils.RecordConstructorArgs):
#     '''
#     Concatenate observation from the last ``n+1`` time steps and action from the last ``n`` time steps
#     Based on gymnasium.wrappers.FrameStack
#     '''
#     def __init__(
#         self,
#         env,
#         stack_size_obs:int,
#         # stack_size_act:int,
#         # lz4_compress: bool = False,
#     ):
#         gymnasium.utils.RecordConstructorArgs.__init__(
#             self, num_stack=stack_size_obs, lz4_compress=False
#         )
#         gymnasium.ObservationWrapper.__init__(self, env)

#         self.stack_size_obs = stack_size_obs
#         self.stack_size_act = stack_size_obs
#         # self.lz4_compress = lz4_compress
#         # print(f'env.action_space={env.action_space}')
#         self.padding_value_obs = create_zero_array(env.observation_space)
#         self.padding_value_act = create_zero_array(env.action_space)
#         self.obs_queue = deque(maxlen=stack_size_obs+1)
#         self.act_queue = deque(maxlen=stack_size_obs)
        
#         obs_space = concat_space(env.observation_space, stack_size_obs+1)
#         act_space = concat_space(env.action_space, stack_size_obs)
#         # print(f'obs_space = {obs_space}')
#         # print(f'act_space = {act_space}')
#         joint_low = np.concatenate((obs_space.low, act_space.low))
#         joint_high = np.concatenate((obs_space.high, act_space.high))
#         if obs_space.dtype == act_space.dtype == np.float32:
#             joint_dtype = np.float32
#         else:
#             raise ValueError(f'Unable to define joint_dtype. Found obs_space.dtype={obs_space.dtype}, act_space.dtype={act_space.dtype}.')
#         # print('joint_low =', joint_low)
#         joint_space = gymnasium.spaces.box.Box(low=joint_low, high=joint_high, dtype=joint_dtype)
#         self.observation_space = joint_space
#         self.stacked_obs = create_empty_array(env.observation_space, n=self.stack_size_obs+1)
#         self.stacked_act = create_empty_array(env.action_space, n=self.stack_size_act)
#         # print('init: self.stacked_obs =', self.stacked_obs)
#         # print('init: self.stacked_act =', self.stacked_act)

#     def step(self, action):
#         """Steps through the environment, appending the observation to the frame buffer.

#         Args:
#             action: The action to step through the environment with

#         Returns:
#             Stacked observations, reward, terminated, truncated, and information from the environment
#         """
#         observation, reward, terminated, truncated, info = self.env.step(action)
#         self.obs_queue.appendleft(observation)
#         if isinstance(self.env.action_space, (gym.spaces.discrete.Discrete, gymnasium.spaces.discrete.Discrete)):
#             # print(f'action = {action}')
#             action = [action]
#         self.act_queue.appendleft(action)
#         concat_obs = deepcopy(
#             concatenate(self.env.observation_space, self.obs_queue, self.stacked_obs)
#         )
#         # print(f'obs_queue = {self.obs_queue}, concat_obs = {concat_obs.flatten()}')
#         # print(f'concat_obs = {concat_obs.flatten()}')
#         concat_act = deepcopy(
#             concatenate(self.env.action_space, self.act_queue, self.stacked_act)
#         )
#         # print(f'act_queue = {self.act_queue}, concat_act = {concat_act.flatten()}')
#         # print(f'concat_act = {concat_act.flatten()}')
#         updated_obs = np.concatenate((concat_obs.flatten(), concat_act.flatten()))
#         return updated_obs, reward, terminated, truncated, info
#         # return self.observation(None), reward, terminated, truncated, info

#     def reset(self, **kwargs):
#         """Reset the environment with kwargs.

#         Args:
#             **kwargs: The kwargs for the environment reset

#         Returns:
#             The stacked observations
#         """
#         obs, info = self.env.reset(**kwargs)
#         # print('(Before) self.obs_queue =', self.obs_queue)
#         # print('(Before) self.act_queue =', self.act_queue)
#         for _ in range(self.stack_size_obs):
#             self.obs_queue.appendleft(self.padding_value_obs)
#         for _ in range(self.stack_size_act):
#             self.act_queue.appendleft(self.padding_value_act)
#         self.obs_queue.appendleft(obs)
#         # print('(After) self.obs_queue =', self.obs_queue)
#         # print('(After) self.act_queue =', self.act_queue)
        
#         concat_obs = deepcopy(
#             concatenate(self.env.observation_space, self.obs_queue, self.stacked_obs)
#         )
        
#         concat_act = deepcopy(
#             concatenate(self.env.action_space, self.act_queue, self.stacked_act)
#         )
#         updated_obs = np.concatenate((concat_obs.flatten(), concat_act.flatten()))
#         return updated_obs, info

# class ConcatObservationAction(
#     gymnasium.Wrapper[WrapperObsType, ActType, ObsType, ActType],
#     gymnasium.utils.RecordConstructorArgs,
# ):
#     '''
#     Concatenate observation from the last ``n`` time steps and action from the last ``m`` time steps
#     Based on gymnasium.wrappers.FrameStackObservation
#     '''
#     def __init__(
#         self,
#         env,
#         stack_size_obs:int,
#         stack_size_act:int,
#     ):
#         gymnasium.Wrapper.__init__(self, env)

#         if not np.issubdtype(type(stack_size_obs), np.integer):
#             raise TypeError(
#                 f"The stack_size_obs is expected to be an integer, actual type: {type(stack_size_obs)}"
#             )
#         if not 1 < stack_size_obs:
#             raise ValueError(
#                 f"The stack_size_obs needs to be greater than one, actual value: {stack_size_obs}"
#             )
#         if not np.issubdtype(type(stack_size_act), np.integer):
#             raise TypeError(
#                 f"The stack_size_act is expected to be an integer, actual type: {type(stack_size_act)}"
#             )
#         if not 1 < stack_size_act:
#             raise ValueError(
#                 f"The stack_size_act needs to be greater than one, actual value: {stack_size_act}"
#             )

#         self.padding_value_obs: ObsType = create_zero_array(env.observation_space)
#         self.padding_value_act: ActType = create_zero_array(env.action_space)
#         self.observation_space = concat_space(env.observation_space, stack_size_obs)
#         # self.action_space = concat_space(env.action_space, stack_size_act)
#         self.stack_size_obs: Final[int] = stack_size_obs
#         self.stack_size_act: Final[int] = stack_size_act
#         self.obs_queue = deque(
#             [self.padding_value_obs for _ in range(self.stack_size_obs)], maxlen=self.stack_size_obs
#         )
#         self.act_queue = deque(
#             [self.padding_value_act for _ in range(self.stack_size_act)], maxlen=self.stack_size_act
#         )
#         self.stacked_obs = np.flatten(create_empty_array(env.observation_space, n=self.stack_size_obs))
#         self.stacked_act = np.flatten(create_empty_array(env.action_space, n=self.stack_size_act))

#     def step(
#         self, action: WrapperActType
#     ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
#         obs, reward, terminated, truncated, info = self.env.step(action)
#         self.obs_queue.append(obs)
#         self.act_queue.append(action)

#         concat_obs = deepcopy(
#             concatenate(self.env.observation_space, self.obs_queue, self.stacked_obs)
#         )
        
#         concat_act = deepcopy(
#             concatenate(self.env.action_space, self.act_queue, self.stacked_act)
#         )
#         updated_obs = np.concat(np.flatten(concat_obs), np.flatten(concat_act))
#         return updated_obs, reward, terminated, truncated, info

#     def reset(
#         self, *, seed: int | None = None, options: dict[str, Any] | None = None
#     ) -> tuple[WrapperObsType, dict[str, Any]]:
#         obs, info = self.env.reset(seed=seed, options=options)
#         # The below for loop should not be required
#         for _ in range(self.stack_size_obs - 1):
#             self.obs_queue.append(self.padding_value)
#         self.obs_queue.append(obs)
#         concat_obs = deepcopy(
#             concatenate(self.env.observation_space, self.obs_queue, self.stacked_obs)
#         )
        
#         concat_act = deepcopy(
#             concatenate(self.env.action_space, self.act_queue, self.stacked_act)
#         )
#         updated_obs = np.concat(np.flatten(concat_obs), np.flatten(concat_act))
#         return updated_obs, info

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

# """Wrapper around a Brax GymWrapper, that converts outputs to PyTorch tensors.

# This conversion happens directly on-device, without moving values to the CPU.
# """
# from typing import Optional

# # NOTE: The following line will emit a warning and raise ImportError if `torch`
# # isn't available.
# from brax.io import torch as btorch
# # import gym


# class TorchWrapper(gymnasium.Wrapper):
#     """Wrapper that converts Jax tensors to PyTorch tensors."""
    
#     def __init__(self, env: gymnasium.Env, device: Optional[btorch.Device] = None):
#         """Creates a gym Env to one that outputs PyTorch tensors."""
#         super().__init__(env)
#         self.device = device
    
#     def reset(self, seed: Optional[int] = None, options: dict = {}):
#         obs = super().reset()
#         return btorch.jax_to_torch(obs, device=self.device), {}
    
#     def step(self, action):
#         # print(f'(TorchWrapper) Before: action = {action}, type = {type(action)}')
#         # action = btorch.torch_to_jax(action)
#         # print(f'(TorchWrapper) torch_to_jax(action) = {action}')
#         obs, reward, term, trunc, info = super().step(action)
#         print(f'(TorchWrapper) obs = {obs}, type = {type(obs)}')
#         obs = btorch.jax_to_torch(obs, device=self.device)
#         print(f'(TorchWrapper) jax_to_torch(obs) = {obs}')
#         reward = btorch.jax_to_torch(reward, device=self.device)
#         # done = btorch.jax_to_torch(done, device=self.device)
#         term = btorch.jax_to_torch(term, device=self.device)
#         trunc = btorch.jax_to_torch(trunc, device=self.device)
#         info = btorch.jax_to_torch(info, device=self.device)
#         # return obs, reward, done, info
#         return obs, reward, term, trunc, info

# """Implementation of StepAPICompatibility wrapper class for transforming envs between new and old step API."""
# # import gymnasium as gym
# from gymnasium.logger import deprecation
# from gymnasium.utils.step_api_compatibility import step_api_compatibility

# class GymStepAPICompatibility(gymnasium.Wrapper, gymnasium.utils.RecordConstructorArgs):
#     r"""A wrapper which can transform an environment from new step API to old and vice-versa.

#     Old step API refers to step() method returning (observation, reward, done, info)
#     New step API refers to step() method returning (observation, reward, terminated, truncated, info)
#     (Refer to docs for details on the API change)

#     Example:
#         >>> import gymnasium as gym
#         >>> from gymnasium.wrappers import StepAPICompatibility
#         >>> env = gym.make("CartPole-v1")
#         >>> env # wrapper not applied by default, set to new API
#         <TimeLimit<OrderEnforcing<PassiveEnvChecker<CartPoleEnv<CartPole-v1>>>>>
#         >>> env = StepAPICompatibility(gym.make("CartPole-v1"))
#         >>> env
#         <StepAPICompatibility<TimeLimit<OrderEnforcing<PassiveEnvChecker<CartPoleEnv<CartPole-v1>>>>>>
#     """

#     def __init__(self, env: gymnasium.Env, output_truncation_bool: bool = True):
#         """A wrapper which can transform an environment from new step API to old and vice-versa.

#         Args:
#             env (gym.Env): the env to wrap. Can be in old or new API
#             output_truncation_bool (bool): Whether the wrapper's step method outputs two booleans (new API) or one boolean (old API)
#         """
#         gymnasium.utils.RecordConstructorArgs.__init__(
#             self, output_truncation_bool=output_truncation_bool
#         )
#         gymnasium.Wrapper.__init__(self, env)

#         self.is_vector_env = isinstance(env.unwrapped, (gymnasium.vector.VectorEnv, gym.vector.VectorEnv))
#         self.output_truncation_bool = output_truncation_bool
#         if not self.output_truncation_bool:
#             deprecation(
#                 "Initializing environment in (old) done step API which returns one bool instead of two."
#             )

#     def step(self, action):
#         """Steps through the environment, returning 5 or 4 items depending on `output_truncation_bool`.

#         Args:
#             action: action to step through the environment with

#         Returns:
#             (observation, reward, terminated, truncated, info) or (observation, reward, done, info)
#         """
#         step_returns = self.env.step(action)
#         return step_api_compatibility(
#             step_returns, self.output_truncation_bool, self.is_vector_env
#         )

# from brax.envs.base import PipelineEnv
# from brax.io import image
# import jax
# from typing import ClassVar, Optional

# class GymnasiumWrapper(gymnasium.Env):
#     """A wrapper that converts Brax Env to one that follows Gymnasium API."""
    
#     # Flag that prevents `gym.register` from misinterpreting the `_step` and
#     # `_reset` as signs of a deprecated gym Env API.
#     _gym_disable_underscore_compat: ClassVar[bool] = True

#     def __init__(self,
#                  env: PipelineEnv,
#                  seed: int = 0,
#                  backend: Optional[str] = None):
#         self._env = env
#         self.metadata = {
#             'render.modes': ['human', 'rgb_array'],
#             'video.frames_per_second': 1 / self._env.dt
#         }
#         self.seed(seed)
#         self.backend = backend
#         self._state = None
        
#         obs = np.inf * np.ones(self._env.observation_size, dtype='float32')
#         self.observation_space = gymnasium.spaces.Box(-obs, obs, dtype='float32')
        
#         action = jax.tree.map(np.array, self._env.sys.actuator.ctrl_range)
#         self.action_space = gymnasium.spaces.Box(action[:, 0], action[:, 1], dtype='float32')
    
#         def reset(key):
#             key1, key2 = jax.random.split(key)
#             state = self._env.reset(key2)
#             return state, state.obs, key1
        
#         self._reset = jax.jit(reset, backend=self.backend)
        
#         def step(state, action):
#             state = self._env.step(state, action)
#             info = {**state.metrics, **state.info}
#             trunc = info.pop("truncation", False)
#             return state, state.obs, state.reward, state.done, trunc, info
        
#         self._step = jax.jit(step, backend=self.backend)
    
#     def reset(self, seed: Optional[int] = None, options: dict = {}):
#         self._state, obs, self._key = self._reset(self._key)
#         info = {**self._state.metrics, **self._state.info}
#         return obs, info
    
#     def step(self, action):
#         # print(f'state.pipeline_state.q={self._state.pipeline_state.q}, action={action}')
#         action_ = np.expand_dims(action, axis=0)
#         self._state, obs, reward, done, trunc, info = self._step(self._state, action_)
#         # print(f'reward={reward}, shape={reward.shape}')
#         # print(f'trunc={trunc}, shape={trunc.shape}')
#         return obs, reward[0], done, trunc[0], info
    
#     def seed(self, seed: int = 0):
#         self._key = jax.random.PRNGKey(seed)

#     def render(self, mode='human'):
#         if mode == 'rgb_array':
#             sys, state = self._env.sys, self._state
#             if state is None:
#                 raise RuntimeError('must call reset or step before rendering')
#             return image.render_array(sys, state.pipeline_state, 256, 256)
#         else:
#             return super().render(mode=mode)  # just raise an exception

# class VectorGymnasiumWrapper(gymnasium.vector.VectorEnv):
#     """A wrapper that converts batched Brax Env to one that follows Gymnasium VectorEnv API."""
    
#     # Flag that prevents `gym.register` from misinterpreting the `_step` and
#     # `_reset` as signs of a deprecated gym Env API.
#     _gym_disable_underscore_compat: ClassVar[bool] = True
    
#     def __init__(self, 
#                  env: PipelineEnv,
#                  seed: int = 0,
#                  backend: Optional[str] = None):
#         self._env = env
#         self.metadata = {
#             'render.modes': ['human', 'rgb_array'],
#             'video.frames_per_second': 1 / self._env.dt
#         }
#         if not hasattr(self._env, 'batch_size'):
#             raise ValueError('underlying env must be batched')
        
#         self.num_envs = self._env.batch_size
#         self.seed(seed)
#         self.backend = backend
#         self._state = None
        
#         obs = np.inf * np.ones(self._env.observation_size, dtype='float32')
#         obs_space = gymnasium.spaces.Box(-obs, obs, dtype='float32')
#         self.observation_space = gymnasium.vector.utils.batch_space(obs_space, self.num_envs)
        
#         action = jax.tree.map(np.array, self._env.sys.actuator.ctrl_range)
#         action_space = gymnasium.spaces.Box(action[:, 0], action[:, 1], dtype='float32')
#         self.action_space = gymnasium.vector.utils.batch_space(action_space, self.num_envs)
        
#         def reset(key):
#             key1, key2 = jax.random.split(key)
#             state = self._env.reset(key2)
#             return state, state.obs, key1
        
#         self._reset = jax.jit(reset, backend=self.backend)
        
#         def step(state, action):
#             state = self._env.step(state, action)
#             info = {**state.metrics, **state.info}
#             print('(VectorGymnWrapper) info.truncation =', info.truncation)
#             truncated = info.pop("truncation", False)
#             return state, state.obs, state.reward, state.done, truncated, info
        
#         self._step = jax.jit(step, backend=self.backend)
    
#     def reset(self, seed: Optional[int] = None, options: dict = {}):
#         self._state, obs, self._key = self._reset(self._key)
#         info = {**self._state.metrics, **self._state.info}
#         return obs, info
    
#     def step(self, action):
#         self._state, obs, reward, done, truncated, info = self._step(self._state, action)
#         return obs, reward, done, truncated, info
    
#     def seed(self, seed: int = 0):
#         self._key = jax.random.PRNGKey(seed)
    
#     def render(self, mode='human'):
#         if mode == 'rgb_array':
#             sys, state = self._env.sys, self._state
#             if state is None:
#                 raise RuntimeError('must call reset or step before rendering')
#             return image.render_array(sys, state.pipeline_state.take(0), 256, 256)
#         else:
#             return super().render(mode=mode)  # just raise an exception

# from gymnasium.core import ActType, ObsType, WrapperObsType
# from gymnasium.wrappers import TransformObservation

class UPExpertWrapper(gymnasium.ObservationWrapper):
    def __init__(self, env, experts_context, context_mean, fixed_idx=-1, knnclf=None):
        assert isinstance(env.observation_space, (gym.spaces.Box, gymnasium.spaces.Box))
        self.env = env
        new_low = np.concatenate((np.array([-np.inf]*len(context_mean)), env.observation_space.low))
        new_high = np.concatenate((np.array([np.inf]*len(context_mean)), env.observation_space.high))
        new_observation_space = gymnasium.spaces.Box(
            low=new_low,
            high=new_high,
            shape=new_low.shape,
            dtype=env.observation_space.dtype
        )
        # self.shape = new_low.shape
        # gymnasium.utils.RecordConstructorArgs.__init__(self, shape=new_low.shape)
        assert fixed_idx==-1 or knnclf is None # Both cannot be absent 
        assert not (fixed_idx!=-1 and knnclf is not None) # Both cannot be given
        # print(f'Base env context: {self.env.context}')
        # print(f'experts_context = {experts_context}, fixed_idx={fixed_idx}')
        if fixed_idx!=-1:
            self.transform_func = lambda obs: np.concatenate((experts_context[fixed_idx], obs))
        elif knnclf is not None:
            # print(f'Using knnclf.')
            self.transform_func = lambda obs: self.knn_context(obs, experts_context, context_mean, knnclf)
        # gymnasium.utils.RecordConstructorArgs.__init__(self, f=self.transform_func)
        self.observation_space = new_observation_space

    def knn_context(self, obs, experts_context, context_mean, knnclf):
        obs_len = obs.shape[0] - len(context_mean)
        obs_wo_context = obs[-obs_len:]
        context_t = obs[:-obs_len].reshape(-1,1)
        # print(f'context_t = {context_t}')
        assert context_t.size != 0
        context_mean_copy = np.array(context_mean).reshape(-1, 1)
        # print(f'context_mean_copy = {context_mean_copy}')
        context_t = context_t/context_mean_copy
        expert_index = knnclf.predict(context_t.T)[0]
        new_obs = np.concatenate((experts_context[expert_index], obs_wo_context))
        return new_obs
        
    # def observation(self, observation):
    #     return self.transform_func(observation)
    
    def step(self, a):
        raw_o, r, tm, tu, info = self.env.step(a)
        
        return self.transform_func(raw_o), r, tm, tu, info
    
    def reset(self):
        raw_o, info = self.env.reset()
        return self.transform_func(raw_o), info

# class UPExpertWrapper(
#     TransformObservation,
#     gymnasium.utils.RecordConstructorArgs,):
#     def __init__(self, env, experts_context, context_mean, fixed_idx=-1, knnclf=None):
#         assert isinstance(env.observation_space, (gym.spaces.Box, gymnasium.spaces.Box))
#         new_low = np.concatenate((np.array([-np.inf]*len(context_mean)), env.observation_space.low))
#         new_high = np.concatenate((np.array([np.inf]*len(context_mean)), env.observation_space.high))
#         new_observation_space = gymnasium.spaces.Box(
#             low=new_low,
#             high=new_high,
#             shape=new_low.shape,
#             dtype=env.observation_space.dtype
#         )
#         self.shape = new_low.shape
#         gymnasium.utils.RecordConstructorArgs.__init__(self, shape=new_low.shape)
#         assert fixed_idx==-1 or knnclf is None # Both cannot be absent 
#         assert not (fixed_idx!=-1 and knnclf is not None) # Both cannot be given
#         if fixed_idx!=-1:
#             transform_func = lambda obs: np.concatenate((experts_context[fixed_idx], obs))
#         elif knnclf is not None:
#             transform_func = lambda obs: knn_context(obs, experts_context, context_mean, knnclf)
#         TransformObservation.__init__(
#             self,
#             env=env,
#             f=transform_func,
#             observation_space=new_observation_space,
#         )

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