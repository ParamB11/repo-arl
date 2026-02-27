# import sys

import gym
import gymnasium
import numpy as np
import torch

# import tensorflow as tf
# from tf_agents.replay_buffers import reverb_utils
# from tf_agents.trajectories import TimeStep
# from tf_agents.trajectories import trajectory

# def collect_step(environment, policy, replay_buffer_fn):
#     time_step = environment.current_time_step()
#     #print("time_step = ", time_step)
    
#     if time_step.is_last():
#         # reset policy_state at the end of the episode
#         #policy_state = policy.get_initial_state(environment.batch_size)
#         done = True
#         #return (policy_state, done)
#         return done
#     else:
#         done = False
        
#     action_step = policy.action(time_step)
#     #print("action_step = ", action_step)
#     next_time_step = environment.step(action_step.action)
#     #print("next_time_step = ", next_time_step)
    
#     #policy_state = next_policy_state
#     traj = trajectory.from_transition(time_step, action_step, next_time_step)

#     # Add trajectory to the replay buffer
#     # if hide_context:
#     #     replay_buffer_ddpg.add_batch(traj)
#     # else:
#     #     replay_buffer_ddpg_context.add_batch(traj)
#     replay_buffer_fn.add_batch(traj)
#     return done

# def collect_step_moe(environment, policy, options_dict, replay_buffer_fn):
#     # print("Collecting one step of data.")
#     time_step = environment.current_time_step()
    
#     if time_step.is_last():
#         done = True
#         environment.reset()
#         time_step = environment.current_time_step()
#         # return done
#     else:
#         done = False
        
#     action, choice_step = MoeFunction(time_step, policy, options_dict)
#     next_time_step = environment.step(action)
    
#     traj = trajectory.from_transition(time_step, choice_step, next_time_step)

#     replay_buffer_fn.add_batch(traj)
#     return done

# def collect_step_moe_predictor(environment, policy, options_dict, 
#                                lstm_model, dense_model, features_prev, predictor_state, replay_buffer,
#                                label_scale_factor, eng=False, dt=0.05):
#     time_step = environment.current_time_step()
#     if time_step.is_last():
#         done = True
#         next_predictor_state = None
#         next_features_prev = tf.zeros(features_prev.shape)
#         environment.reset()
#         time_step = environment.current_time_step()
#         # return (next_predictor_state, next_features_prev, done)
#     else:
#         done = False
        
#     ## calculating time_step_hat
#     observation = time_step.observation
#     obs_len = observation.shape[1]
#     if options_dict[0].action_spec.shape ==():
#         act_len = 1
#     else:
#         act_len = options_dict[0].action_spec.shape[0]
#     feat_len = observation.shape[1] + act_len
    
#     if eng:
# # #         print("observation = ", observation)
# # #         print("features_prev = ", features_prev)
# #         features_temp = tf.concat([observation, features_prev[0]], 1)
# #         v_t = tf.math.subtract(features_temp[:,0:obs_len], features_temp[:,obs_len:2*obs_len])/dt
# # #         v_t_prev = tf.math.subtract(features_temp[:,obs_len:2*obs_len], features_temp[:,obs_len+feat_len:2*obs_len+feat_len])/dt
# # #         accel = tf.math.subtract(v_t, v_t_prev)/dt
# #         max_v_t = tf.math.reduce_max(tf.math.abs(v_t), axis=1)
# #         v_t_prev = tf.math.subtract(features_temp[:,obs_len:2*obs_len], features_temp[:,obs_len+feat_len:2*obs_len+feat_len])/dt
# #         max_v_t_prev = tf.math.reduce_max(tf.math.abs(v_t_prev), axis=1)
# #         max_v = tf.math.maximum(max_v_t, max_v_t_prev)
# #         v_t = v_t/max_v
# #         v_t_prev = v_t_prev/max_v
# #         accel = tf.math.subtract(v_t, v_t_prev)/dt
# #         features_eng = tf.expand_dims(tf.concat([features_temp, v_t, v_t_prev, accel], 1), axis=0)
#         features_eng = eng_feature_tf(observation, features_prev, options_dict[0].action_spec.shape[0], dt=dt)
#         # context_, memory_state, cell_state = lstm_model(features_eng, initial_state = predictor_state)
#         lstm_layers = [layer.name for layer in lstm_model.layers]
#         context_, memory_state, cell_state = lstm_model.get_layer(lstm_layers[1])(features_eng, initial_state = predictor_state)
#     else:
#         features_stack = tf.expand_dims(tf.concat([observation, features_prev[0]], 1), axis=0)
#         lstm_layers = [layer.name for layer in lstm_model.layers]
#         context_, memory_state, cell_state = lstm_model.get_layer(lstm_layers[1])(features_stack, initial_state = predictor_state)
# #     context_, memory_state, cell_state = lstm_model(features_prev, initial_state = predictor_state)
#     # context_hat = tf.stop_gradient(dense_model(context_))*label_scale_factor
# #     context_hat = dense_model(context_)*label_scale_factor
#     if type(dense_model) is dict:
#         context_mid = dense_model[0](context_, training=False)
#         for model_index in range(1, len(dense_model)):
#             context_scalar = dense_model[model_index](context_, training=False)
#             context_mid = tf.concat([context_mid, context_scalar], axis=2)
#         context_hat = context_mid
#     else:
#         dense_layers = [layer.name for layer in dense_model.layers]
#         context_hat = dense_model.get_layer(dense_layers[0])(context_)
        
#     context_hat = context_hat*label_scale_factor
#     # observation_hat = tf.concat([observation, context_hat[0]], 1)
#     observation_hat = tf.concat([context_hat[0], observation], 1)
#     time_step_hat = TimeStep(time_step.step_type, time_step.reward, time_step.discount, observation_hat)
#     ##
        
#     action, next_predictor_state, next_features_prev, choice_step = MoeFunctionPredictor(time_step, policy, options_dict, 
#                                                lstm_model, dense_model, features_prev, predictor_state, label_scale_factor,
#                                                eng=eng, dt=dt)
#     next_time_step = environment.step(action)
    
#     ## calculating next_time_step_hat
#     next_observation = next_time_step.observation
#     if eng:
#         next_features_temp = tf.concat([next_observation, next_features_prev[0]], 1)
# #         print("next_features_eng = ", next_features_eng)
#         next_v_t = tf.math.subtract(next_features_temp[:,0:obs_len], next_features_temp[:,obs_len:2*obs_len])/dt
# #         v_t = (x_eng[:,:,0:obs_len] - x_eng[:,:,obs_len:2*obs_len])/dt
# #         print("next_v_t = ", next_v_t)
# #         next_v_t_prev = tf.math.subtract(next_features_eng[:,obs_len:2*obs_len], next_features_eng[:,obs_len+feat_len:2*obs_len+feat_len])/dt
# #         v_t_prev = (x_eng[:,:,obs_len:2*obs_len] -x_eng[:,:,obs_len+feat_len:2*obs_len+feat_len])/dt
# #         next_accel = tf.math.subtract(next_v_t, next_v_t_prev)/dt
#         max_next_v_t = tf.math.reduce_max(tf.math.abs(next_v_t), axis=1)
#         next_v_t_prev = tf.math.subtract(next_features_temp[:,obs_len:2*obs_len], next_features_temp[:,obs_len+feat_len:2*obs_len+feat_len])/dt
#         max_next_v_t_prev = tf.math.reduce_max(tf.math.abs(next_v_t_prev), axis=1)
#         max_next_v = tf.math.maximum(max_next_v_t, max_next_v_t_prev)
#         next_v_t = next_v_t/max_next_v
#         next_v_t_prev = next_v_t_prev/max_next_v
#         next_accel = tf.math.subtract(next_v_t, next_v_t_prev)/dt
#         next_features_eng = tf.expand_dims(tf.concat([next_features_temp, next_v_t, next_v_t_prev, next_accel], 1), axis=0)
#         # next_context_, next_memory_state, next_cell_state = lstm_model(next_features_eng, initial_state = next_predictor_state)
#         lstm_layers = [layer.name for layer in lstm_model.layers]
#         next_context_, next_memory_state, next_cell_state = lstm_model.get_layer(lstm_layers[1])(next_features_eng, initial_state = next_predictor_state)
#     else:
#         next_features_stack = tf.expand_dims(tf.concat([next_observation, next_features_prev[0]], 1), axis=0)
#         # next_context_, next_memory_state, next_cell_state = lstm_model(next_features_stack, initial_state = next_predictor_state)
#         lstm_layers = [layer.name for layer in lstm_model.layers]
#         next_context_, next_memory_state, next_cell_state = lstm_model.get_layer(lstm_layers[1])(next_features_stack, initial_state = next_predictor_state)
#     # next_context_hat = tf.stop_gradient(dense_model(next_context_))*label_scale_factor
# #     next_context_hat = dense_model(next_context_)*label_scale_factor
#     if type(dense_model) is dict:
#         next_context_mid = dense_model[0](next_context_, training=False)
#         for model_index in range(1, len(dense_model)):
#             next_context_scalar = dense_model[model_index](next_context_, training=False)
#             next_context_mid = tf.concat([next_context_mid, next_context_scalar], axis=2)
#         next_context_hat = next_context_mid
#     else:
#         # next_context_hat = dense_model(context_)
#         dense_layers = [layer.name for layer in dense_model.layers]
#         next_context_hat = dense_model.get_layer(dense_layers[0])(context_)
    
#     next_context_hat = next_context_hat*label_scale_factor
#     # next_observation_hat = tf.concat([next_observation, next_context_hat[0]], 1)
#     next_observation_hat = tf.concat([next_context_hat[0], next_observation], 1)
#     next_time_step_hat = TimeStep(next_time_step.step_type, next_time_step.reward, next_time_step.discount, next_observation_hat)
#     ##
    
#     traj = trajectory.from_transition(time_step_hat, choice_step, next_time_step_hat)

#     # Add trajectory to the replay buffer
#     replay_buffer.add_batch(traj)
#     return (next_predictor_state, next_features_prev, done)

# class ReverbFixedLengthSequenceObserver(
#     reverb_utils.ReverbAddTrajectoryObserver
# ):
#     """Reverb fixed length sequence observer.

#     This is a specialized observer similar to ReverbAddTrajectoryObserver but each
#     sequence contains a fixed number of steps and can span multiple episodes. This
#     implementation is consistent with (Schulman, 17).

#     **Note**: Counting of steps in drivers does not include boundary steps. To
#     guarantee only 1 item is pushed to the replay when collecting n steps with a
#     `sequence_length` of n make sure to set the `stride_length`.
#     """

#     def __call__(self, trajectory):
#         """Writes the trajectory into the underlying replay buffer.

#         Allows trajectory to be a flattened trajectory. No batch dimension allowed.

#         Args:
#           trajectory: The trajectory to be written which could be (possibly nested)
#             trajectory object or a flattened version of a trajectory. It assumes
#             there is *no* batch dimension.
#         """
#         self._writer.append(trajectory)
#         self._cached_steps += 1

#         self._write_cached_steps()

# def collect_datapoint_re(environment, obs, done, policy, cep=False, moe=False, pred=False, options_dict=None, 
#                        experts_context=None, context_mean=None, pred_net=None, 
#                        lstm_state=None, features_prev=None):
#     # time_step_context = environment.current_time_step()
#     label_scale_factor = context_mean
    
#     if done:
#         # done = True
#         environment.reset()
#         if pred:
#             next_lstm_state = None
#             next_features_prev = tf.zeros(features_prev.shape)
#         else:
#             next_lstm_state = lstm_state
#             next_features_prev = features_prev
#         # return (next_lstm_state, next_features_prev, done, None, None)
#     # else:
#     #     done = False
    
#     observation_context =  obs #time_step_context.observation
#     if not moe: obs_len = policy.time_step_spec.observation.shape.as_list()[0]
#     else: obs_len = options_dict[0].time_step_spec.observation.shape.as_list()[0]
        
#     if pred:
#         if options_dict[0].action_spec.shape==(): act_len = 1
#         else: act_len = options_dict[0].action_spec.shape[0]
# #     obs_len = time_step_spec.observation.shape.as_list()[0]
#     # observation = observation_context[:, 0:obs_len]
#     # label = observation_context[:, obs_len:]
#     observation = observation_context[:, -obs_len:]
#     label = observation_context[:, :-obs_len]
#     # print("observation_context = ", observation_context)
#     # print("label = ", label)
#     time_step = TimeStep(time_step_context.step_type, time_step_context.reward, time_step_context.discount, observation)
#     if cep and moe:
#         raise ValueError("Both cep and moe cannot be True.")
    
#     if cep:
#         raise NotImplementedError
#         # if pred:
#         #     features_stack = tf.expand_dims(tf.concat([observation, features_prev[0]], 1), axis=0)
#         #     lstm_layers = [layer.name for layer in lstm_model.layers]
#         #     context_, memory_state, cell_state = lstm_model.get_layer(lstm_layers[1])(features_stack, initial_state = lstm_state)
#         #     dense_layers = [layer.name for layer in dense_model.layers]
#         #     context_hat = dense_model.get_layer(dense_layers[0])(context_)
#         #     context_hat = context_hat*label_scale_factor
#         #     observation_hat = tf.concat([context_hat[0], observation], 1)
#         #     time_step_hat = TimeStep(time_step.step_type, time_step.reward, time_step.discount, observation_hat)
#         #     next_lstm_state = [memory_state, cell_state]
#         #     action, choice = ClosestExpertPolicy(time_step_hat, options_dict, experts_context, context_mean)
#         #     # next_features_prev
#         #     feature_length = observation.shape[1] + act_len
#         #     # new_feature = tf.concat([observation, action_], 1)
#         #     if tf.rank(action) != tf.rank(observation):
#         #         mod_action = tf.expand_dims(action, axis=0)
#         #         # print("action = {0}, mod_action = {1}".format(action, mod_action))
#         #     else: mod_action = action
#         #     if mod_action.dtype != observation.dtype:
#         #         # print("mod_action.dtype ({0}) not same as observation.dtype ({1}).".format(mod_action.dtype, observation.dtype))
#         #         mod_action = tf.cast(mod_action, dtype = observation.dtype)
#         #     new_feature = tf.concat([observation, mod_action], 1)
#         #     next_features_prev = tf.reshape(
#         #         tf.concat([new_feature[0], features_prev[0,0,0:-feature_length]], 0), 
#         #         features_prev.shape)
#         # else:
#         #     next_features_prev = features_prev
#         #     next_lstm_state = lstm_state
#         #     action, choice = ClosestExpertPolicy(time_step_context, options_dict, experts_context, context_mean)
#     elif moe:
#         raise NotImplementedError
#         # if pred:
#         #     action, next_lstm_state, next_features_prev, choice_step = MoeFunctionPredictor(time_step, policy, options_dict, 
#         #                                        lstm_model, dense_model, features_prev, lstm_state, label_scale_factor,
#         #                                        eng=False, dt=0.05)
#         # else:
#         #     next_features_prev = features_prev
#         #     next_lstm_state = lstm_state
#         #     action, _ = MoeFunction(time_step_context, policy, options_dict)
#     else:
#         next_features_prev = features_prev
#         next_lstm_state = lstm_state
#         # action_step = policy.action(time_step)
#         if hasattr(policy, 'action'):
#             action_step = policy.action(time_step)
#         elif hasattr(policy, 'predict'):
#             action = policy.predict(time_step.observation)
#             action_step = tuple_to_action_step(action)
#         action = action_step.action
#         # raise ValueError(f"cep = {cep}, moe = {moe}. Atleast one of them should be true.")
#     #print("action_step = ", action_step)
#     next_time_step_context = environment.step(action)

#     # Add [observation, action] to features_ds
# #     print("observation = ", observation.numpy(), observation.numpy().shape)
# #     print("action_step.action = ", action_step.action.numpy(), action_step.action.numpy().shape)
#     # feature = np.concatenate((observation.numpy(), action.numpy()), 1)
#     if observation.numpy().ndim != action_step.action.numpy().ndim:
#         temp_action = np.expand_dims(action_step.action.numpy(), axis=1)
#         feature = np.concatenate((observation.numpy(), temp_action), 1)
#     else:
#         feature = np.concatenate((observation.numpy(), action_step.action.numpy()), 1)
#     # features_list.append(feature[0])
#     # Add label to labels_ds
#     # labels_list.append(label.numpy()[0])
#     return (next_lstm_state, next_features_prev, done, feature, label.numpy())

# def collect_timesteps_re(environment, policy, features_list, labels_list, cep=False, moe=False, pred=False, options_dict=None, 
#                        experts_context=None, context_mean=None, lstm_model=None, dense_model=None, 
#                        lstm_state=None, features_prev=None, nsteps=None, nepisodes=None):
#     tfeatures = np.array([])
#     tlabels = np.array([])
#     done = False
#     if nsteps is not None and nepisodes is not None:
#         raise ValueError(f"nsteps={nsteps}, nepisodes={nepisodes}. Both cannot be nonzero.")
#     if nsteps is not None:
#         for step in range(nsteps):
#             if done:
#                 features_list.append(tfeatures)
#                 labels_list.append(tlabels)
#                 tfeatures = np.array([])
#                 tlabels = np.array([])
#                 done = False
    
#             # print(f"step {step}: tfeatures = {tfeatures}, tlabels = {tlabels}")
#             lstm_state, features_prev, done, feature, label = collect_datapoint_re(environment, policy,
#                                       cep=cep, moe=moe, pred=pred, options_dict=options_dict, 
#                                       experts_context=experts_context, context_mean=context_mean, 
#                                       lstm_model=lstm_model, dense_model=dense_model, 
#                                       lstm_state=lstm_state, features_prev=features_prev)
#             if type(feature) != type(None):
#                 # tfeatures = np.append(tfeatures, feature)
#                 tfeatures = np.vstack(tfeatures, feature)
#             # Add label to labels_ds
#             if type(label) != type(None): 
#                 tlabels = np.append(tlabels, label.numpy())
#     if nepisodes is not None:
#         episode = 0
#         while episode < nepisodes:
#             # print(f"step {step}: tfeatures = {tfeatures}, tlabels = {tlabels}")
#             lstm_state, features_prev, done, feature, label = collect_datapoint_re(environment, policy,
#                                       cep=cep, moe=moe, pred=pred, options_dict=options_dict, 
#                                       experts_context=experts_context, context_mean=context_mean, 
#                                       lstm_model=lstm_model, dense_model=dense_model, 
#                                       lstm_state=lstm_state, features_prev=features_prev)
#             if done:
#                 features_list.append(tfeatures)
#                 labels_list.append(tlabels)
#                 tfeatures = np.array([])
#                 tlabels = np.array([])
#                 done = False
#                 episode += 1
            
#             if type(feature) != type(None):
#                 # print("tfeatures = ", tfeatures)
#                 if tfeatures.size > 0:
#                     tfeatures = np.vstack((tfeatures, feature))
#                 else:
#                     tfeatures = np.append(tfeatures, feature)
#             # Add label to labels_ds
#             if type(label) != type(None):
#                 if tlabels.size > 0:
#                     tlabels = np.vstack((tlabels, label))
#                 else:
#                     tlabels = np.append(tlabels, label)

#     if nsteps is None and nepisodes is None:
#         raise ValueError("nsteps and nepisodes both are zero.")

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
    # print(f'feature={feature}')
    # features_list.append(feature[0])
    # Add label to labels_ds
    # labels_list.append(label.numpy()[0])
    # return (next_lstm_state, next_features_prev, done, feature, label)
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