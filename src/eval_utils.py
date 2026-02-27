import numpy as np

# from policies_th import ClosestExpertPolicy, kNNExpertPolicy

def compute_returns(environment, policy, num_episodes=10, deterministic=True):
    returns = []
    observation, _ = environment.reset()
    episode_return = 0.0
    episode_count = 0
    while episode_count < num_episodes:
        action = policy.predict(observation, deterministic=deterministic)[0]
        new_observation, reward, term, trunc, _ = environment.step(action)
        episode_return += reward
        if term or trunc:
            episode_count += 1
            new_observation, _ = environment.reset()
            returns.append(episode_return)
            episode_return = 0.0
        observation = new_observation
    return returns

def compute_avg_return(environment, policy, num_episodes=10, deterministic=True):
    returns = []
    observation, _ = environment.reset()
    episode_return = 0.0
    episode_count = 0
    while episode_count < num_episodes:
        action = policy.predict(observation, deterministic=deterministic)[0]
        new_observation, reward, term, trunc, _ = environment.step(action)
        episode_return += reward
        if term or trunc:
            episode_count += 1
            new_observation, _ = environment.reset()
            returns.append(episode_return)
            episode_return = 0.0
        observation = new_observation
    
    avg_return = np.mean(returns)
    # print("returns = ", returns)
    # print("np.mean(returns) = ", np.mean(returns))
    stddev = np.std(returns)
    # print("stddev = ", stddev)
    return avg_return, stddev

# def compute_avg_return_cep(environment, experts_dict, experts_context, context_mean, transform=None,
#                           pred_net=None, label_scale_factor=None, pred=False, num_episodes=10,
#                           verbose=False, eng=False, dt=0.05):
#     returns = []
#     if pred and pred_net is None:
#         raise ValueError("Error: pred_net is None.")
#     observation, _ = environment.reset()
#     episode_return = 0.0
#     episode_count = 0
#     # print(f'Begin Episode no: {episode_count+1}')
#     while episode_count < num_episodes:
#         if not pred:
#             action, choice = ClosestExpertPolicy(observation, experts_dict, experts_context, context_mean, transform)
#         else:
#             raise NotImplementedError
#         new_observation, reward, term, trunc, _ = environment.step(action)
#         episode_return += reward
#         if term or trunc:
#             episode_count += 1
#             new_observation, _ = environment.reset()
#             # print(f'Episode no:{episode_count}, Current context:{environment.context}')
#             # print(f'Episode no:{episode_count}, Current context:{environment.get_wrapper_attr("context")}')
#             # print(f'Begin Episode no: {episode_count+1}')
#             returns.append(episode_return)
#             episode_return = 0.0
#         if verbose:
#                 print("observation = {0}, choice = {1}, action = {2}"
#                       .format(observation, choice, action))
#         observation = new_observation
#     avg_return = np.mean(returns)
#     stddev = np.std(returns)
#     return avg_return, stddev

# def compute_avg_return_knnep(environment, options_dict, experts_context, context_mean, knnclf=None, 
#                              pred_net=None, label_scale_factor=None, pred=False, 
#                              num_episodes=10, verbose=False, eng=False, dt=0.05):

#     # total_return = 0.0
#     returns = []
    
#     observation, _ = environment.reset()
#     episode_return = 0.0
#     episode_count = 0
#     # for _ in range(num_episodes):
#     print(f'knnclf = {knnclf}')
#     while episode_count < num_episodes:
#         # time_step = environment.reset()
#         # episode_return = 0.0
#         # if pred:
#         #     features_last = initial_feature
#         #     lstm_state = None

#         # while not time_step.is_last():
#             # observation = time_step.observation
#         if not pred:
#             # action, choice = kNNExpertPolicy(time_step, options_dict, experts_context, context_mean, knnclf)
#             action, choice = kNNExpertPolicy(observation, options_dict, experts_context, context_mean, knnclf)
#         else:
#             raise NotImplementedError
#         new_observation, reward, term, trunc, _ = environment.step(action)
#         episode_return += reward
#         if term or trunc:
#             episode_count += 1
#             new_observation, _ = environment.reset()
#             # print(f'Episode no:{episode_count}, Current context:{environment.context}')
#             # print(f'Episode no:{episode_count}, Current context:{environment.get_wrapper_attr("context")}')
#             # print(f'Begin Episode no: {episode_count+1}')
#             returns.append(episode_return)
#             episode_return = 0.0
#         if verbose:
#             print("observation = {0}, choice = {1}, action = {2}".format(observation, 
#                                                                           choice, action))
#         observation = new_observation
#         # returns.append(episode_return.numpy()[0])

#     avg_return = np.mean(returns)
#     stddev = np.std(returns)
#     return avg_return, stddev