# import sys

import gym
import gymnasium
import numpy as np
import torch

supported_box_spaces = (gym.spaces.box.Box, gymnasium.spaces.box.Box)
supported_discrete_spaces = (gym.spaces.discrete.Discrete, gymnasium.spaces.discrete.Discrete)

def collect_datapoint(environment, obs, done, policy):
    if done:
        obs = environment.reset()
    obs_len = environment.unwrapped.observation_space.shape[0]
    observation_context =  obs
    # print(f'obs={obs},type={type(obs)}')
    observation = observation_context[-obs_len:]
    label = observation_context[:-obs_len]
    # print("observation_context = ", observation_context)
    # print("label = ", label)
    action = policy.predict(observation)[0]
    # next_time_step_context = environment.step(action)
    # print(f'action={action}')
    next_obs_context, next_reward, next_term, next_trunc, _ = environment.step(action)
    next_done = next_term or next_trunc
    # if observation.numpy().ndim != action.numpy().ndim:
    if observation.ndim != action.ndim:
        # print(f"obs.ndim = {observation.ndim}, act.ndim = {action.ndim}")
        temp_action = np.expand_dims(action, axis=-1)
        feature = np.concatenate((observation, temp_action), 0)
    else:
        feature = np.concatenate((observation, action), 0)
    return next_obs_context, next_done, feature, label

def collect_timesteps(environment, policy, features_list, labels_list, nsteps=None, nepisodes=None):
    tfeatures = np.array([])
    tlabels = np.array([])
    done = False
    obs, _ = environment.reset()
    if nsteps is not None and nepisodes is not None:
        raise ValueError(f"nsteps={nsteps}, nepisodes={nepisodes}. Both cannot be nonzero.")
    if nsteps is not None:
        for step in range(nsteps):
            if done:
                features_list.append(tfeatures)
                labels_list.append(tlabels)
                tfeatures = np.array([])
                tlabels = np.array([])
                # done = False
    
            # print(f"step {step}: tfeatures = {tfeatures}, tlabels = {tlabels}")
            obs, done, feature, label = collect_datapoint(environment, obs, done, policy)
            if type(feature) != type(None):
                # tfeatures = np.append(tfeatures, feature)
                tfeatures = np.vstack(tfeatures, feature)
            # Add label to labels_ds
            if type(label) != type(None): 
                tlabels = np.append(tlabels, label.numpy())
    if nepisodes is not None:
        episode = 0
        while episode < nepisodes:
            # print(f"step {step}: tfeatures = {tfeatures}, tlabels = {tlabels}")
            obs, done, feature, label = collect_datapoint(environment, obs, done, policy)
            if done:
                features_list.append(tfeatures)
                labels_list.append(tlabels)
                tfeatures = np.array([])
                tlabels = np.array([])
                # done = False
                episode += 1
            
            if type(feature) != type(None):
                # print("tfeatures = ", tfeatures)
                if tfeatures.size > 0:
                    tfeatures = np.vstack((tfeatures, feature))
                else:
                    tfeatures = np.append(tfeatures, feature)
            # Add label to labels_ds
            if type(label) != type(None):
                if tlabels.size > 0:
                    tlabels = np.vstack((tlabels, label))
                else:
                    tlabels = np.append(tlabels, label)

    if nsteps is None and nepisodes is None:
        raise ValueError("nsteps and nepisodes both are zero.")

# def collect_datapoint_up(environment, obs, done, policy):
def collect_datapoint_up(environment, obs, done, policy, action_noise=0.0, max_abs_action_space=None):
    if done:
        obs, _ = environment.reset()
    obs_len = environment.unwrapped.observation_space.shape[0]
    observation_context =  obs
    # print(f'obs={obs},type={type(obs)}')
    observation = observation_context[-obs_len:]
    label = observation_context[:-obs_len]
    # print("observation_context = ", observation_context)
    # print("label = ", label)
    action = policy.predict(observation_context)[0]
    # next_time_step_context = environment.step(action)
    # print(f'action={action}')
    # next_obs_context, next_reward, next_term, next_trunc, _ = environment.step(action)
    if action_noise > 0.0:
        if isinstance(environment.action_space, supported_box_spaces):
            # print('Has attribute len')
            # noisy_act = act + np.random.normal(0, args.action_noise, len(act))
            noise = np.random.normal(0, action_noise, size=action.shape[0])*max_abs_action_space
            # print(f'action = {action}, noise = {noise}')
            noisy_act = action + noise
        # elif isinstance(env_to_use.action_space, gym.spaces.discrete.Discrete):
        elif isinstance(environment.action_space, supported_discrete_spaces):
            # noisy_act = act
            ## choose a random action with probability args.action.noise
            if np.random.uniform() < action_noise:
                noisy_act = environment.action_space.sample()
                while noisy_act == action:
                    # print(f'Noisy action matched action. Resampling. noisy_act = {noisy_act}, action = {action}.')
                    noisy_act = environment.action_space.sample()
                # print(f'Random action = {action}, noisy action = {noisy_act}')
            else:
                noisy_act = action
    else:
        noisy_act = action
    # print(f"action={action}, noisy_act={noisy_act}")
    step_data = environment.step(noisy_act) # environment.step(action)
    if len(step_data) == 5:
        # assert False, f'Forced error. len(step_data) = {len(step_data)}'
        next_obs_context, next_reward, next_term, next_trunc, _ = step_data
        next_done = next_term or next_trunc
    else:
        # assert False, f'Forced error. len(step_data) = {len(step_data)}'
        next_obs_context, next_reward, next_done, _ = step_data
    # if observation.numpy().ndim != action.numpy().ndim:
    if observation.ndim != noisy_act.ndim: #action.ndim:
        # print(f"obs.ndim = {observation.ndim}, act.ndim = {action.ndim}")
        temp_action = np.expand_dims(noisy_act, axis=-1) #np.expand_dims(action, axis=-1)
        feature = np.concatenate((observation, temp_action), 0)
    else:
        feature = np.concatenate((observation, noisy_act), 0) # np.concatenate((observation, action), 0)
    # print(f'feature={feature}')
    # features_list.append(feature[0])
    # Add label to labels_ds
    # labels_list.append(label.numpy()[0])
    # return (next_lstm_state, next_features_prev, done, feature, label)
    return next_obs_context, next_done, feature, label

# def collect_timesteps_up(environment, policy, features_list, labels_list, nsteps=None, nepisodes=None):
def collect_timesteps_up(environment, policy, features_list, labels_list, action_noise=0.0, nsteps=None, nepisodes=None):
    tfeatures = np.array([])
    tlabels = np.array([])
    done = False

    if isinstance(environment.action_space, supported_box_spaces):
        # print(f'action_space = {environment.action_space}')
        if len(environment.action_space.shape) == 1:
            space_low = np.expand_dims(environment.action_space.low, axis=1)
            space_high = np.expand_dims(environment.action_space.high, axis=1)
        else:
            space_low = environment.action_space.low
            space_high = environment.action_space.high
        abs_action_space = np.concatenate((space_low, space_high), axis=1)
        # print(f'abs_action_space = {abs_action_space}')
        max_abs_action_space = np.max(abs_action_space, axis=1)
        # print(f'max_abs_action_space = {max_abs_action_space}')
    else:
        max_abs_action_space = None

    obs, _ = environment.reset()
    if nsteps is not None and nepisodes is not None:
        raise ValueError(f"nsteps={nsteps}, nepisodes={nepisodes}. Both cannot be nonzero.")
    if nsteps is not None:
        for step in range(nsteps):
            # obs, done, feature, label = collect_datapoint_up(environment, obs, done, policy)
            obs, done, feature, label = collect_datapoint_up(
                environment, obs, done, policy, action_noise=action_noise, max_abs_action_space=max_abs_action_space)
            if done:
                features_list.append(tfeatures)
                labels_list.append(tlabels)
                # print(f'step {step}: done={done} len(features_list)={len(features_list)}, len(labels_list)={len(labels_list)}')
                tfeatures = np.array([])
                tlabels = np.array([])
                # done = False
            if type(feature) != type(None):
                # print("tfeatures = ", tfeatures)
                if tfeatures.size > 0:
                    tfeatures = np.vstack((tfeatures, feature))
                else:
                    tfeatures = np.append(tfeatures, feature)
            else:
                raise ValueError(f'Invalid feature type. type(feature)={type(feature)}')
            # Add label to labels_ds
            if type(label) != type(None):
                if tlabels.size > 0:
                    tlabels = np.vstack((tlabels, label))
                else:
                    tlabels = np.append(tlabels, label)
            else:
                raise ValueError(f'Invalid label type. type(label)={type(label)}')
    if nepisodes is not None:
        episode = 0
        while episode < nepisodes:
            # print(f"step {step}: tfeatures = {tfeatures}, tlabels = {tlabels}")
            # obs, done, feature, label = collect_datapoint_up(environment, obs, done, policy)
            obs, done, feature, label = collect_datapoint_up(
                environment, obs, done, policy, action_noise=action_noise, max_abs_action_space=max_abs_action_space)
            if done:
                features_list.append(tfeatures)
                labels_list.append(tlabels)
                tfeatures = np.array([])
                tlabels = np.array([])
                # done = False
                episode += 1
            
            if type(feature) != type(None):
                # print("tfeatures = ", tfeatures)
                if tfeatures.size > 0:
                    tfeatures = np.vstack((tfeatures, feature))
                else:
                    tfeatures = np.append(tfeatures, feature)
            # Add label to labels_ds
            if type(label) != type(None):
                if tlabels.size > 0:
                    tlabels = np.vstack((tlabels, label))
                else:
                    tlabels = np.append(tlabels, label)

    if nsteps is None and nepisodes is None:
        raise ValueError("nsteps and nepisodes both are zero.")

def collect_datapoint_up_pendulum(environment, obs, done, policy, max_noise=0.1):
    '''
    Base function: collect_datapoint_up
    Additions: Noise can be added to the action
    '''
    if done:
        obs, _ = environment.reset()
    obs_len = environment.unwrapped.observation_space.shape[0]
    observation_context =  obs
    # print(f'obs={obs},type={type(obs)}')
    observation = observation_context[-obs_len:]
    label = observation_context[:-obs_len]
    # print("observation_context = ", observation_context)
    # print("label = ", label)
    action = policy.predict(observation_context)[0]
    if isinstance(environment.action_space, gymnasium.spaces.box.Box):
        # print('Has attribute len')
        noisy_act = action + np.random.normal(0, max_noise, len(action))
        # print(f'action = {action}, noisy_act = {noisy_act}')
    elif isinstance(environment.action_space, gymnasium.spaces.discrete.Discrete):
        noisy_act = action
    action = noisy_act
    # next_time_step_context = environment.step(action)
    # print(f'action={action}')
    # next_obs_context, next_reward, next_term, next_trunc, _ = environment.step(action)
    step_data = environment.step(action)
    if len(step_data) == 5:
        # assert False, f'Forced error. len(step_data) = {len(step_data)}'
        next_obs_context, next_reward, next_term, next_trunc, _ = step_data
        next_done = next_term or next_trunc
    else:
        # assert False, f'Forced error. len(step_data) = {len(step_data)}'
        next_obs_context, next_reward, next_done, _ = step_data
    # if observation.numpy().ndim != action.numpy().ndim:
    if observation.ndim != action.ndim:
        # print(f"obs.ndim = {observation.ndim}, act.ndim = {action.ndim}")
        temp_action = np.expand_dims(action, axis=-1)
        feature = np.concatenate((observation, temp_action), 0)
    else:
        feature = np.concatenate((observation, action), 0)
    # print(f'feature={feature}')
    # features_list.append(feature[0])
    # Add label to labels_ds
    # labels_list.append(label.numpy()[0])
    # return (next_lstm_state, next_features_prev, done, feature, label)
    return next_obs_context, next_done, feature, label

def collect_timesteps_up_pendulum(environment, policy, features_list, labels_list, nsteps=None, nepisodes=None, max_noise=0.05):
    '''
    Base function: collect_timesteps_up
    Additions: Noise can be added to the action
    '''
    tfeatures = np.array([])
    tlabels = np.array([])
    done = False
    obs, _ = environment.reset()
    if nsteps is not None and nepisodes is not None:
        raise ValueError(f"nsteps={nsteps}, nepisodes={nepisodes}. Both cannot be nonzero.")
    if nsteps is not None:
        for step in range(nsteps):
            obs, done, feature, label = collect_datapoint_up_pendulum(environment, obs, done, policy, max_noise=max_noise)
            if done:
                features_list.append(tfeatures)
                labels_list.append(tlabels)
                # print(f'step {step}: done={done} len(features_list)={len(features_list)}, len(labels_list)={len(labels_list)}')
                tfeatures = np.array([])
                tlabels = np.array([])
                # done = False
            if type(feature) != type(None):
                # print("tfeatures = ", tfeatures)
                if tfeatures.size > 0:
                    tfeatures = np.vstack((tfeatures, feature))
                else:
                    tfeatures = np.append(tfeatures, feature)
            else:
                raise ValueError(f'Invalid feature type. type(feature)={type(feature)}')
            # Add label to labels_ds
            if type(label) != type(None):
                if tlabels.size > 0:
                    tlabels = np.vstack((tlabels, label))
                else:
                    tlabels = np.append(tlabels, label)
            else:
                raise ValueError(f'Invalid label type. type(label)={type(label)}')
    if nepisodes is not None:
        episode = 0
        while episode < nepisodes:
            # print(f"step {step}: tfeatures = {tfeatures}, tlabels = {tlabels}")
            obs, done, feature, label = collect_datapoint_up_pendulum(environment, obs, done, policy, max_noise=max_noise)
            if done:
                features_list.append(tfeatures)
                labels_list.append(tlabels)
                tfeatures = np.array([])
                tlabels = np.array([])
                # done = False
                episode += 1
            
            if type(feature) != type(None):
                # print("tfeatures = ", tfeatures)
                if tfeatures.size > 0:
                    tfeatures = np.vstack((tfeatures, feature))
                else:
                    tfeatures = np.append(tfeatures, feature)
            # Add label to labels_ds
            if type(label) != type(None):
                if tlabels.size > 0:
                    tlabels = np.vstack((tlabels, label))
                else:
                    tlabels = np.append(tlabels, label)

    if nsteps is None and nepisodes is None:
        raise ValueError("nsteps and nepisodes both are zero.")

def collect_datapoint_upex(environment, obs, done, policy, expert_context):
    if done:
        obs = environment.reset()
    obs_len = environment.unwrapped.observation_space.shape[0]
    observation_context =  obs
    # print(f'obs={obs},type={type(obs)}')
    observation = observation_context[-obs_len:]
    label = observation_context[:-obs_len]
    # print("observation_context = ", observation_context)
    # print("label = ", label)
    observation_excontext = np.concatenate((expert_context, observation))
    # print(f'obs_excontext = {observation_excontext}')
    action = policy.predict(observation_excontext)[0]
    # next_time_step_context = environment.step(action)
    # print(f'action={action}')
    next_obs_context, next_reward, next_term, next_trunc, _ = environment.step(action)
    next_done = next_term or next_trunc
    # if observation.numpy().ndim != action.numpy().ndim:
    if observation.ndim != action.ndim:
        # print(f"obs.ndim = {observation.ndim}, act.ndim = {action.ndim}")
        temp_action = np.expand_dims(action, axis=-1)
        feature = np.concatenate((observation, temp_action), 0)
    else:
        feature = np.concatenate((observation, action), 0)
    # print(f'feature={feature}')
    # features_list.append(feature[0])
    # Add label to labels_ds
    # labels_list.append(label.numpy()[0])
    # return (next_lstm_state, next_features_prev, done, feature, label)
    return next_obs_context, next_done, feature, label

def collect_timesteps_upex(environment, policy, expert_context, features_list, labels_list, nsteps=None, nepisodes=None):
    tfeatures = np.array([])
    tlabels = np.array([])
    done = False
    obs, _ = environment.reset()
    if nsteps is not None and nepisodes is not None:
        raise ValueError(f"nsteps={nsteps}, nepisodes={nepisodes}. Both cannot be nonzero.")
    if nsteps is not None:
        for step in range(nsteps):
            if done:
                features_list.append(tfeatures)
                labels_list.append(tlabels)
                tfeatures = np.array([])
                tlabels = np.array([])
                # done = False
    
            # print(f"step {step}: tfeatures = {tfeatures}, tlabels = {tlabels}")
            obs, done, feature, label = collect_datapoint_upex(environment, obs, done, policy, expert_context)
            if type(feature) != type(None):
                # tfeatures = np.append(tfeatures, feature)
                tfeatures = np.vstack(tfeatures, feature)
            # Add label to labels_ds
            if type(label) != type(None): 
                tlabels = np.append(tlabels, label.numpy())
    if nepisodes is not None:
        episode = 0
        while episode < nepisodes:
            # print(f"step {step}: tfeatures = {tfeatures}, tlabels = {tlabels}")
            obs, done, feature, label = collect_datapoint_upex(environment, obs, done, policy, expert_context)
            if done:
                features_list.append(tfeatures)
                labels_list.append(tlabels)
                tfeatures = np.array([])
                tlabels = np.array([])
                # done = False
                episode += 1
            
            if type(feature) != type(None):
                # print("tfeatures = ", tfeatures)
                if tfeatures.size > 0:
                    tfeatures = np.vstack((tfeatures, feature))
                else:
                    tfeatures = np.append(tfeatures, feature)
            # Add label to labels_ds
            if type(label) != type(None):
                if tlabels.size > 0:
                    tlabels = np.vstack((tlabels, label))
                else:
                    tlabels = np.append(tlabels, label)

    if nsteps is None and nepisodes is None:
        raise ValueError("nsteps and nepisodes both are zero.")

# def collect_datapoint_uppred(environment, obs, done, policy):
def collect_datapoint_uppred(environment, obs, done, policy, action_noise=0.0, max_abs_action_space=None):
    if done:
        obs, _ = environment.reset()
    obs_len = environment.unwrapped.observation_space.shape[0]
    observation_context =  obs
    # print(f'obs={obs},type={type(obs)}')
    # print(f'obs_len = {obs_len}')
    observation = observation_context[-obs_len:]
    # label = observation_context[:-obs_len]
    label = list(environment.wrapped_env.context.values())
    # print("observation_context = ", observation_context)
    # print("label = ", label)
    action = policy.predict(observation_context)[0]
    # next_time_step_context = environment.step(action)
    # print(f'action={action}')
    # next_obs_context, next_reward, next_term, next_trunc, _ = environment.step(action)
    if action_noise > 0.0:
        if isinstance(environment.action_space, supported_box_spaces):
            # print('Has attribute len')
            # noisy_act = act + np.random.normal(0, args.action_noise, len(act))
            noise = np.random.normal(0, action_noise, size=action.shape[0])*max_abs_action_space
            # print(f'action = {action}, noise = {noise}')
            noisy_act = action + noise
        # elif isinstance(env_to_use.action_space, gym.spaces.discrete.Discrete):
        elif isinstance(environment.action_space, supported_discrete_spaces):
            # noisy_act = act
            ## choose a random action with probability args.action.noise
            if np.random.uniform() < action_noise:
                noisy_act = environment.action_space.sample()
                while noisy_act == action:
                    # print(f'Noisy action matched action. Resampling. noisy_act = {noisy_act}, action = {action}.')
                    noisy_act = environment.action_space.sample()
                # print(f'Random action = {action}, noisy action = {noisy_act}')
            else:
                noisy_act = action
    else:
        noisy_act = action
    step_data = environment.step(noisy_act) #environment.step(action)
    if len(step_data) == 5:
        # assert False, f'Forced error. len(step_data) = {len(step_data)}'
        next_obs_context, next_reward, next_term, next_trunc, _ = step_data
        next_done = next_term or next_trunc
    else:
        # assert False, f'Forced error. len(step_data) = {len(step_data)}'
        next_obs_context, next_reward, next_done, _ = step_data
    # if observation.numpy().ndim != action.numpy().ndim:
    if observation.ndim != noisy_act.ndim: #action.ndim:
        # print(f"obs.ndim = {observation.ndim}, act.ndim = {action.ndim}")
        temp_action = np.expand_dims(noisy_act, axis=-1) #np.expand_dims(action, axis=-1)
        feature = np.concatenate((observation, temp_action), 0)
    else:
        feature = np.concatenate((observation, noisy_act), 0) #np.concatenate((observation, action), 0)
    # print(f'feature={feature}')
    # features_list.append(feature[0])
    # Add label to labels_ds
    # labels_list.append(label.numpy()[0])
    # return (next_lstm_state, next_features_prev, done, feature, label)
    return next_obs_context, next_done, feature, label

def collect_timesteps_uppred(environment, policy, features_list, labels_list, action_noise=0.0, nsteps=None, nepisodes=None):
    tfeatures = np.array([])
    tlabels = np.array([])
    done = False

    if isinstance(environment.action_space, supported_box_spaces):
        # print(f'action_space = {environment.action_space}')
        if len(environment.action_space.shape) == 1:
            space_low = np.expand_dims(environment.action_space.low, axis=1)
            space_high = np.expand_dims(environment.action_space.high, axis=1)
        else:
            space_low = environment.action_space.low
            space_high = environment.action_space.high
        abs_action_space = np.concatenate((space_low, space_high), axis=1)
        # print(f'abs_action_space = {abs_action_space}')
        max_abs_action_space = np.max(abs_action_space, axis=1)
        # print(f'max_abs_action_space = {max_abs_action_space}')
    else:
        max_abs_action_space = None

    obs, _ = environment.reset()
    # print(f'collect_timesteps_uppred: environment.observation_space.shape = {environment.observation_space.shape}')
    # print(f'collect_timesteps_uppred: environment.reset(): obs = {obs}')
    if nsteps is not None and nepisodes is not None:
        raise ValueError(f"nsteps={nsteps}, nepisodes={nepisodes}. Both cannot be nonzero.")
    if nsteps is not None:
        for step in range(nsteps):
            # obs, done, feature, label = collect_datapoint_uppred(environment, obs, done, policy)
            obs, done, feature, label = collect_datapoint_uppred(environment, obs, done, policy, action_noise=action_noise, max_abs_action_space=max_abs_action_space)
            if done:
                features_list.append(tfeatures)
                labels_list.append(tlabels)
                tfeatures = np.array([])
                tlabels = np.array([])
                # done = False
            # print(f"step {step}: tfeatures = {tfeatures}, tlabels = {tlabels}")
            
            # if type(feature) != type(None):
            #     # tfeatures = np.append(tfeatures, feature)
            #     tfeatures = np.vstack(tfeatures, feature)
            # # Add label to labels_ds
            # if type(label) != type(None): 
            #     tlabels = np.append(tlabels, label.numpy())
            if type(feature) != type(None):
                # print("tfeatures = ", tfeatures)
                if tfeatures.size > 0:
                    tfeatures = np.vstack((tfeatures, feature))
                else:
                    tfeatures = np.append(tfeatures, feature)
            # Add label to labels_ds
            if type(label) != type(None):
                if tlabels.size > 0:
                    tlabels = np.vstack((tlabels, label))
                else:
                    tlabels = np.append(tlabels, label)
    if nepisodes is not None:
        episode = 0
        while episode < nepisodes:
            # print(f"step {step}: tfeatures = {tfeatures}, tlabels = {tlabels}")
            # obs, done, feature, label = collect_datapoint_uppred(environment, obs, done, policy)
            obs, done, feature, label = collect_datapoint_uppred(environment, obs, done, policy, action_noise=action_noise, max_abs_action_space=max_abs_action_space)
            if done:
                features_list.append(tfeatures)
                labels_list.append(tlabels)
                tfeatures = np.array([])
                tlabels = np.array([])
                # done = False
                episode += 1
            
            if type(feature) != type(None):
                # print("tfeatures = ", tfeatures)
                if tfeatures.size > 0:
                    tfeatures = np.vstack((tfeatures, feature))
                else:
                    tfeatures = np.append(tfeatures, feature)
            # Add label to labels_ds
            if type(label) != type(None):
                if tlabels.size > 0:
                    tlabels = np.vstack((tlabels, label))
                else:
                    tlabels = np.append(tlabels, label)

    if nsteps is None and nepisodes is None:
        raise ValueError("nsteps and nepisodes both are zero.")

def collect_datapoint_up_wr(environment, obs, done, policy):
    if done:
        obs = environment.reset()
    obs_len = environment.unwrapped.observation_space.shape[0]
    observation_context =  obs
    # print(f'obs={obs},type={type(obs)}')
    observation = observation_context[-obs_len:]
    label = observation_context[:-obs_len]
    # print("observation_context = ", observation_context)
    # print("label = ", label)
    action = policy.predict(observation_context)[0]
    # next_time_step_context = environment.step(action)
    # print(f'action={action}')
    next_obs_context, next_reward, next_term, next_trunc, _ = environment.step(action)
    next_done = next_term or next_trunc
    # if observation.numpy().ndim != action.numpy().ndim:
    if observation.ndim != action.ndim:
        # print(f"obs.ndim = {observation.ndim}, act.ndim = {action.ndim}")
        temp_action = np.expand_dims(action, axis=-1)
        feature = np.concatenate((observation, temp_action), 0)
    else:
        feature = np.concatenate((observation, action), 0)
    # print(f'feature={feature}')
    # features_list.append(feature[0])
    # Add label to labels_ds
    # labels_list.append(label.numpy()[0])
    # return (next_lstm_state, next_features_prev, done, feature, label)
    return next_obs_context, next_reward, next_done, feature, label

def collect_timesteps_up_wr(environment, policy, features_list, labels_list, nsteps=None, nepisodes=None):
    tfeatures = np.array([])
    tlabels = np.array([])
    done = False
    obs, _ = environment.reset()
    if nsteps is not None and nepisodes is not None:
        raise ValueError(f"nsteps={nsteps}, nepisodes={nepisodes}. Both cannot be nonzero.")
    if nsteps is not None:
        for step in range(nsteps):
            if done:
                features_list.append(tfeatures)
                labels_list.append(tlabels)
                tfeatures = np.array([])
                tlabels = np.array([])
                # done = False
    
            # print(f"step {step}: tfeatures = {tfeatures}, tlabels = {tlabels}")
            obs, done, feature, label = collect_datapoint_up(environment, obs, done, policy)
            if type(feature) != type(None):
                # tfeatures = np.append(tfeatures, feature)
                tfeatures = np.vstack(tfeatures, feature)
            # Add label to labels_ds
            if type(label) != type(None):
                tlabels = np.append(tlabels, label.numpy())
    if nepisodes is not None:
        episode = 0
        episode_reward = 0.0
        reward_all = np.zeros(nepisodes)
        while episode < nepisodes:
            # print(f"step {step}: tfeatures = {tfeatures}, tlabels = {tlabels}")
            obs, reward, done, feature, label = collect_datapoint_up_wr(environment, obs, done, policy)
            episode_reward += reward
            if done:
                features_list.append(tfeatures)
                labels_list.append(tlabels)
                tfeatures = np.array([])
                tlabels = np.array([])
                # done = False
                reward_all[episode] = episode_reward
                episode_reward = 0.0
                episode += 1
            
            if type(feature) != type(None):
                # print("tfeatures = ", tfeatures)
                if tfeatures.size > 0:
                    tfeatures = np.vstack((tfeatures, feature))
                else:
                    tfeatures = np.append(tfeatures, feature)
            # Add label to labels_ds
            if type(label) != type(None):
                if tlabels.size > 0:
                    tlabels = np.vstack((tlabels, label))
                else:
                    tlabels = np.append(tlabels, label)

    if nsteps is None and nepisodes is None:
        raise ValueError("nsteps and nepisodes both are zero.")
    return reward_all