import sys

import numpy as np
# import tensorflow as tf

# from tf_agents.environments import tf_py_environment
# from tf_agents.trajectories import TimeStep

# from carl_wrapper_tf_agents_py import CarlWrapper
# from moe_policies import stack_observations_tf, eng_feature_tf, tuple_to_action_step, MoeFunction, MoeFunctionPredictor, ClosestExpertPolicy

def compute_avg_return(environment, policy, num_episodes=10):
    # total_return = 0.0
    returns = []
    
    for _ in range(num_episodes):

        time_step = environment.reset()
        episode_return = 0.0

        while not time_step.is_last().numpy():
            # action_step = policy.action(time_step)
            if hasattr(policy, 'action'):
                action_step = policy.action(time_step)
            elif hasattr(policy, 'predict'):
                action = policy.predict(time_step.observation, deterministic=True)
                action_step = tuple_to_action_step(action)
            else:
                raise ValueError("Policy in argument policy is of invalid format.")
            # print("action_step.action = ", action_step.action)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        
        returns.append(episode_return.numpy()[0])
        # total_return += episode_return
        

    # avg_return = total_return / num_episodes
    avg_return = np.mean(returns)
    # print("returns = ", returns)
    # print("np.mean(returns) = ", np.mean(returns))
    stddev = np.std(returns)
    # print("stddev = ", stddev)
    return avg_return, stddev

# def test_experts_mp(carl_env_fn, test_experts_dict, context_labels, context_min, context_max, 
#                     n_points=11, hide_context=True, n_episodes=10, verbose=False):
#     if context_labels.shape[0] != context_min.shape[0]:
#         print("context_labels.shape[0] = ", context_labels.shape[0])
#         print("context_min.shape[0] = ", context_min.shape[0])
#         raise ValueError("Input passed not of the right shape.")
        
#     if context_labels.shape[0] != context_max.shape[0]:
#         print("context_labels.shape[0] = ", context_labels.shape[0])
#         print("context_max.shape[0] = ", context_max.shape[0])
#         raise ValueError("Input passed not of the right shape.")
    
#     context_test_range = np.zeros((context_labels.shape[0], n_points))
#     #g_test_range = np.linspace(g_min, g_max, n_points)
#     for i in range(context_labels.shape[0]):
#         context_test_range[i,:] = np.linspace(context_min[i], context_max[i], n_points)
        
#     mean_rewards = np.zeros((len(test_experts_dict), n_points))
#     # std_rewards = np.zeros(mean_rewards.shape)
#     context_i = {0:{x:0.0 for x in context_labels}}
#     for i in range(context_test_range.shape[1]):
#         for label_index in range(len(context_labels)):
#             context_i[0][context_labels[label_index]] = context_test_range[label_index, i]
#         py_env_i = CarlWrapper(carl_env_fn= carl_env_fn, contexts=context_i, 
#                                hide_context=hide_context, verbose=verbose)
#         env_i = tf_py_environment.TFPyEnvironment(py_env_i)
#         for j in range(len(test_experts_dict)):
#             mean_rewards[j, i], _ = compute_avg_return(env_i, test_experts_dict[j], num_episodes=n_episodes)
        
#         #print("Another iteration done.")
            
#     return (context_test_range, mean_rewards)

# def compute_avg_return_moe(environment, policy, options_dict, num_episodes=10, verbose=False):

#     # total_return = 0.0
#     returns = []
#     for _ in range(num_episodes):

#         #policy_state = policy.get_initial_state(environment.batch_size) # for rnn DQN
#         #policy_state = initial_policy_state
#         time_step = environment.reset()
#         episode_return = 0.0

#         while not time_step.is_last():
#             action, choice_step = MoeFunction(time_step, policy, options_dict)
#             #action = tf.convert_to_tensor([action_step.action], dtype=tf.float32)
#             time_step = environment.step(action)
#             #policy_state = action_step.state # for rnn DQN
#             episode_return += time_step.reward
#             if verbose:
#                 print("observation = {0}, choice = {1}, action = {2}".format(time_step.observation, 
#                                                                           choice_step.action, action))
                
#         # total_return += episode_return
#         returns.append(episode_return.numpy()[0])

#     # avg_return = total_return / num_episodes
#     avg_return = np.mean(returns)
#     stddev = np.std(returns)
#     return avg_return, stddev

# def test_policy_moe(test_policy_dict, test_experts_dict, g_min, g_max, step=0.5, hide_context=True, n_episodes=10):
#     n_points = int((g_max - g_min) // step) + 1
#     g_test_range = np.linspace(g_min, g_max, n_points)
#     reward_memory = np.zeros((len(test_policy_dict), len(g_test_range)))
#     for i in range(len(g_test_range)):
#         context_i = {0:{'g':g_test_range[i]}}
#         py_env_i = CarlPendulumWrapper(contexts=context_i, hide_context=hide_context)
#         env_i = tf_py_environment.TFPyEnvironment(py_env_i)
#         for j in range(len(test_policy_dict)):
#             reward_memory[j, i], _ = compute_avg_return_moe(env_i, test_policy_dict[j], 
#                                                                test_experts_dict, num_episodes=n_episodes)
        
#         #print("Another iteration done.")
            
#     return (g_test_range, reward_memory)

# def test_policy_moe_mp(carl_env_fn, test_policy_dict, test_experts_dict, 
#                        context_labels, context_min, context_max, n_points=40, hide_context=True, n_episodes=10):
# #     n_points = int((g_max - g_min) // step) + 1
# #     g_test_range = np.linspace(g_min, g_max, n_points)
# #     reward_memory = np.zeros((len(test_policy_dict), len(g_test_range)))
#     context_test_range = np.zeros((context_labels.shape[0], n_points))

#     for i in range(context_labels.shape[0]):
#         context_test_range[i,:] = np.linspace(context_min[i], context_max[i], n_points)
    
#     reward_memory = np.zeros((len(test_policy_dict), context_test_range.shape[1]))
#     context_i = {0:{x:0.0 for x in context_labels}}
#     for i in range(context_test_range.shape[1]):
# #         context_i = {0:{'g':g_test_range[i]}}
#         for label_index in range(len(context_labels)):
#             context_i[0][context_labels[label_index]] = context_test_range[label_index, i]
#         py_env_i = CarlWrapper(carl_env_fn=carl_env_fn, contexts=context_i, hide_context=hide_context)
#         env_i = tf_py_environment.TFPyEnvironment(py_env_i)
#         for j in range(len(test_policy_dict)):
#             reward_memory[j, i], _ = compute_avg_return_moe(env_i, test_policy_dict[j], 
#                                                                test_experts_dict, num_episodes=n_episodes)
        
#         #print("Another iteration done.")
            
#     return (context_test_range, reward_memory)
    
# def compute_avg_return_predictor_moe(environment, policy, options_dict, 
#                                      lstm_model, dense_model, label_scale_factor, num_episodes=10, verbose=False, eng=False, dt=0.05):

#     # total_return = 0.0
#     returns = []
#     if environment.action_spec().shape == (): act_len = 1
#     else: act_len = environment.action_spec().shape[0]
        
#     if eng:
#         feature_length = 2*(environment.observation_spec().shape[0] + act_len)
#     else:
#         feature_length = 5*(environment.observation_spec().shape[0] + act_len)
#     feature_dtype = environment.observation_spec().dtype
#     initial_feature = tf.reshape(tf.zeros([feature_length], feature_dtype), (1,1, feature_length))
    
#     for _ in range(num_episodes):

#         #policy_state = policy.get_initial_state(environment.batch_size) # for rnn DQN
#         lstm_state = None
#         features_last = initial_feature
#         time_step = environment.reset()
#         episode_return = 0.0
#         step_count = 0

#         while not time_step.is_last():
#             action, next_lstm_state, next_features_last, _ = MoeFunctionPredictor(time_step, policy, options_dict, 
#                                                lstm_model, dense_model, features_last, lstm_state, label_scale_factor, eng=eng, dt=dt)
# #             action = tf.convert_to_tensor([action], dtype=tf.float32)
#             time_step = environment.step(action)
#             lstm_state = next_lstm_state
#             features_last = next_features_last
#             episode_return += time_step.reward
#             step_count += 1
#             if verbose:
#                 print("observation = {0}, choice = {1}, action = {2}".format(time_step.observation, 
#                                                                           action_step.info.action, action_step.action))
#                 #print("step = {0}, context = {1}, context_hat = {2}".format(step_count, environment.get_context()))
                
#         # total_return += episode_return
#         returns.append(episode_return.numpy()[0])

#     # avg_return = total_return / num_episodes
#     avg_return = np.mean(returns)
#     stddev = np.std(returns)
#     return avg_return, stddev

# def test_policy_predictor(test_policy_dict, test_experts_dict, 
#                           test_lstm_model, test_dense_model, g_min, g_max, step=0.5, hide_context=True, n_episodes=10):
#     n_points = int((g_max - g_min) // step) + 1
#     g_test_range = np.linspace(g_min, g_max, n_points)
#     reward_memory = np.zeros((len(test_policy_dict), len(g_test_range)))
#     for i in range(len(g_test_range)):
#         context_i = {0:{'g':g_test_range[i]}}
#         py_env_i = CarlPendulumWrapper(contexts=context_i, hide_context=hide_context)
#         env_i = tf_py_environment.TFPyEnvironment(py_env_i)
#         for j in range(len(test_policy_dict)):
#             reward_memory[j, i] = compute_avg_return_predictor_moe(env_i, test_policy_dict[j], 
#                                 test_experts_dict, test_lstm_model, test_dense_model, num_episodes=n_episodes)
        
#         #print("Another iteration done.")
            
#     return (g_test_range, reward_memory)

# def test_policy_pred_mp(carl_env_fn, test_policy_dict, test_experts_dict, test_lstm_model, test_dense_model, 
#                         context_labels, context_min, context_max, n_points=40, hide_context=True, n_episodes=10,
#                         eng=False, dt=0.05):

#     context_mean = []
#     for key in carl_env_fn.get_default_context().keys():
#         if key in context_labels:
#             context_mean.append(carl_env_fn.get_default_context()[key])
#     label_scale_factor = np.ones((1, len(context_labels)))
#     for idx in range(len(context_labels)):
#         label_scale_factor[0, idx] = context_mean[idx]
#     context_test_range = np.zeros((context_labels.shape[0], n_points))
    
#     for i in range(context_labels.shape[0]):
#         context_test_range[i,:] = np.linspace(context_min[i], context_max[i], n_points)
    
#     reward_memory = np.zeros((len(test_policy_dict), context_test_range.shape[1]))
#     context_i = {0:{x:0.0 for x in context_labels}}
#     for i in range(context_test_range.shape[1]):
#         for label_index in range(len(context_labels)):
#             context_i[0][context_labels[label_index]] = context_test_range[label_index, i]
#         #context_i = {0:{'g':g_test_range[i]}}
#         py_env_i = CarlWrapper(carl_env_fn=carl_env_fn, contexts=context_i, hide_context=hide_context)
#         env_i = tf_py_environment.TFPyEnvironment(py_env_i)
#         for j in range(len(test_policy_dict)):
#             reward_memory[j, i], _ = compute_avg_return_predictor_moe(env_i, test_policy_dict[j], 
#                                 test_experts_dict, test_lstm_model, test_dense_model, label_scale_factor, num_episodes=n_episodes, 
#                                 eng=eng, dt=dt)
        
#         #print("Another iteration done.")
            
#     return (context_test_range, reward_memory)
    
# def compute_avg_return_cep(environment, options_dict, experts_context, context_mean, transform=None,
#                            lstm_model=None, dense_model=None, label_scale_factor=None, pred=False, num_episodes=10, verbose=False, eng=False, dt=0.05):

#     # total_return = 0.0
#     returns = []
#     if pred:
#         if lstm_model is None:
#             print("Error: lstm_model is None.")
#         if dense_model is None:
#             print("Error: dense_model is None.")

#         if environment.action_spec().shape == (): act_len = 1
#         else: act_len = environment.action_spec().shape[0]
        
#         if eng:
#             feature_length = 2*(environment.observation_spec().shape[0] + act_len)
#         else:
#             feature_length = 5*(environment.observation_spec().shape[0] + act_len)
        
#         feature_dtype = environment.observation_spec().dtype
#         initial_feature = tf.reshape(tf.zeros([feature_length], feature_dtype), (1,1, feature_length))
            
#     for _ in range(num_episodes):
    
#         #policy_state = policy.get_initial_state(environment.batch_size) # for rnn DQN
#         #policy_state = initial_policy_state
#         time_step = environment.reset()
#         episode_return = 0.0
#         if pred:
#             features_last = initial_feature
#             lstm_state = None

#         while not time_step.is_last():
#             observation = time_step.observation
#             if not pred:
#                 action, choice = ClosestExpertPolicy(time_step, options_dict, experts_context, context_mean, transform)
#             else:
#                 if eng:
#                     features_eng = eng_feature_tf(observation, features_last, act_len, dt=dt)
#                     # context_, memory_state, cell_state = lstm_model(features_eng, initial_state = lstm_state)
#                     lstm_layers = [layer.name for layer in lstm_model.layers]
#                     context_, memory_state, cell_state = lstm_model.get_layer(lstm_layers[1])(features_eng, initial_state = lstm_state)
#                 else:
#                     features_stack = stack_observations_tf(observation, features_last)
#                     # context_, memory_state, cell_state = lstm_model(features_stack, initial_state = lstm_state)
#                     lstm_layers = [layer.name for layer in lstm_model.layers]
#                     context_, memory_state, cell_state = lstm_model.get_layer(lstm_layers[1])(features_stack, initial_state = lstm_state)
#                 next_lstm_state = [memory_state, cell_state]
#                 if type(dense_model) is dict:
#                     context_mid = dense_model[0](context_)
#                     for model_index in range(1, len(dense_model)):
#                         context_scalar = dense_model[model_index](context_)
#                         context_mid = tf.concat([context_mid, context_scalar], axis=2)
#                     context_hat = context_mid
#                 else:
#                     # context_hat = dense_model(context_)
#                     dense_layers = [layer.name for layer in dense_model.layers]
#                     context_hat = dense_model.get_layer(dense_layers[0])(context_)
#                 # upscaling context_hat
#                 # print(f"context_hat = {context_hat}, label_scale_factor = {label_scale_factor}.")
#                 context_hat = context_hat*label_scale_factor
#                 # print('context_hat = ', context_hat)
#                 # passing context_hat to classifier_policy
#                 # observation_hat = tf.concat([observation, context_hat[0]], 1)
#                 observation_hat = tf.concat([context_hat[0], observation], 1)
#                 time_step_hat = TimeStep(time_step.step_type, time_step.reward, time_step.discount, observation_hat)
#                 action, choice = ClosestExpertPolicy(time_step_hat, options_dict, experts_context, context_mean, transform)
#                 feature_length = observation.shape[1] + act_len
#                 # new_feature = tf.concat([observation, action], 1)
#                 if tf.rank(action) != tf.rank(observation):
#                     mod_action = tf.expand_dims(action, axis=0)
#                     # print("action_ = {0}, mod_action = {1}".format(action_, mod_action))
#                 else: mod_action = action
#                 if mod_action.dtype != observation.dtype:
#                     # print("mod_action.dtype ({0}) not same as observation.dtype ({1}).".format(mod_action.dtype, observation.dtype))
#                     mod_action = tf.cast(mod_action, dtype = observation.dtype)
#                 new_feature = tf.concat([observation, mod_action], 1)
#                 next_features_last = tf.reshape(
#                     tf.concat([new_feature[0], features_last[0,0,0:-feature_length]], 0), 
#                     features_last.shape)
#                 features_last = next_features_last
#                 lstm_state = next_lstm_state
#             #action = tf.convert_to_tensor([action_step.action], dtype=tf.float32)
#             time_step = environment.step(action)
#             #policy_state = action_step.state # for rnn DQN
#             episode_return += time_step.reward
#             if verbose:
#                 print("observation = {0}, choice = {1}, action = {2}".format(time_step.observation, 
#                                                                           choice, action))
                
#         # total_return += episode_return
#         returns.append(episode_return.numpy()[0])

#     # avg_return = total_return / num_episodes
#     avg_return = np.mean(returns)
#     stddev = np.std(returns)
#     return avg_return, stddev

# def test_cep_mp(carl_env_fn, test_experts_dict, test_experts_context, context_mean, 
#                 context_labels, context_min, context_max, transform=None, lstm_model=None, dense_model=None,
#                 pred=False, eng=False, n_points=40, hide_context=True, n_episodes=10):
#     # n_points = int((g_max - g_min) // step) + 1
#     # g_test_range = np.linspace(g_min, g_max, n_points)
#     # reward_memory = np.zeros((len(test_policy_dict), len(g_test_range)))
#     # print("context_labels = ", context_labels)
#     # if pred:
#     context_mean = []
#     for key in carl_env_fn.get_default_context().keys():
#         if key in context_labels:
#             context_mean.append(carl_env_fn.get_default_context()[key])
#     label_scale_factor = np.ones((1, len(context_labels)))
#     for idx in range(len(context_labels)):
#         label_scale_factor[0, idx] = context_mean[idx]
#     context_test_range = np.zeros((context_labels.shape[0], n_points))

#     for i in range(context_labels.shape[0]):
#         context_test_range[i,:] = np.linspace(context_min[i], context_max[i], n_points)
    
#     reward_memory = np.zeros((1, context_test_range.shape[1]))
#     context_i = {0:{x:0.0 for x in context_labels}}
#     for i in range(context_test_range.shape[1]):
#         # context_i = {0:{'g':g_test_range[i]}}
#         for label_index in range(len(context_labels)):
#             context_i[0][context_labels[label_index]] = context_test_range[label_index, i]
#         # print("context_i = ", context_i)
#         py_env_i = CarlWrapper(carl_env_fn=carl_env_fn, contexts=context_i, hide_context=hide_context)
#         env_i = tf_py_environment.TFPyEnvironment(py_env_i)
#         # for j in range(len(test_policy_dict)):
#         reward_memory[0, i], _ = compute_avg_return_cep(env_i, test_experts_dict, test_experts_context, 
#                                                      context_mean, transform, lstm_model, dense_model, label_scale_factor, pred=pred,
#                                                      num_episodes=n_episodes, eng=eng)
#         # avg_return = compute_avg_return_cep(temp_env, tf_experts_dict, context_experts, 
#         #                             lstm_model, dense_model, 
#         #                             pred=True, num_episodes=10, eng=True)
        
#         #print("Another iteration done.")
            
#     return (context_test_range, reward_memory)

def compute_avg_return_uposi(osi_env, up_policy, num_episodes=10):
    returns = []
    for _ in range(num_episodes):
        obs, _ = osi_env.reset()
        episode_return = 0.0
        done = False
        while not done:
            action, _ = up_policy.predict(obs, deterministic=True)
            obs, reward, done, _ = osi_env.step(action)
            episode_return += reward
        returns.append(episode_return)
    avg_return = np.mean(returns)
    stddev = np.std(returns)
    return avg_return, stddev

def compute_returns_uposi(osi_env, up_policy, num_episodes=10):
    returns = []
    for _ in range(num_episodes):
        obs, _ = osi_env.reset()
        episode_return = 0.0
        done = False
        while not done:
            action, _ = up_policy.predict(obs, deterministic=True)
            obs, reward, done, _ = osi_env.step(action)
            episode_return += reward
        returns.append(episode_return)
    # avg_return = np.mean(returns)
    # stddev = np.std(returns)
    # return avg_return, stddev
    return returns