from collections import deque
import copy
import math

import multiprocessing
import numpy as np
import ot
from scipy.spatial import cKDTree
from scipy.special import rgamma
import torch

def pad_and_batch(dataiter, batchsize):
    starti = 0
    dataiter_new = []
    while starti < len(dataiter):
        tlens = [arr.shape[0] for arr, _ in zip(dataiter[starti:], range(batchsize))]
        maxlen = max(tlens)
        print()
        batch_i = np.zeros((batchsize, maxlen, dataiter[0].shape[1]))
        for x, i in zip(dataiter[starti:], range(batchsize)):
            if x.ndim < 3:
                x = np.expand_dims(x, axis=0)
            if maxlen - x.shape[1] > 0:
                padding = np.zeros((x.shape[0], maxlen-x.shape[1], x.shape[2]))
                x_mod = np.concatenate((x, padding), axis=1)
            else:
                x_mod = x
            batch_i[i] = x_mod
            
        dataiter_new.append(batch_i)
        starti += batchsize

    return dataiter_new

'''
Unpad zeros at the end of trajectories
Arguments:
dataiter: Array of padded trajectories. Shape=(#trajs, len(traj), len(feature)).
Output: List of unpadded trajectories.
'''
def unpad(dataiter):
    dataiter_unpad = []
    for i in range(dataiter.shape[0]):
        dataiter_unpad.append(dataiter[i][~np.all(dataiter[i]==0.0, axis=1)])
    return dataiter_unpad

'''
v_t = (s_t - s_{t-1})/dt, v_t_prev=v_{t-1}= (s_{t-1}-s_{t-2})/dt
accel is short for acceleration or the second order derivative.
accel = (v_t - v_{t-1})/dt
'''
def eng_feature(x, obs_len=3, dt=0.05):
    act_len = x.shape[2] - obs_len
    feat_len = obs_len + act_len
    x_eng = x
    delay = 1
    x_delay = np.roll(x, shift=delay, axis=1)
    padding_zero = np.zeros(x[:,0:delay,:].shape)
    x_delay[:,0:delay,:] = padding_zero
    x_eng = np.concatenate((x_eng, x_delay), axis=2)
    x_fwd = np.roll(x, shift=-delay, axis=1)
    padding_zero = np.zeros(x[:,-delay:,:].shape)
    x_fwd[:,-delay:,:] = padding_zero
    x_eng = np.concatenate((x_fwd[:,:,0:obs_len], x_eng), axis=2)
    v_t = (x_eng[:,:,0:obs_len] - x_eng[:,:,obs_len:2*obs_len])/dt
    max_v_t = np.expand_dims(np.max(np.abs(v_t), axis=2), axis=2)
    v_t_prev = (x_eng[:,:,obs_len:2*obs_len] -x_eng[:,:,obs_len+feat_len:2*obs_len+feat_len])/dt
    max_v_t_prev = np.expand_dims(np.max(np.abs(v_t_prev), axis=2), axis=2)
    max_v = np.expand_dims(np.max(np.concatenate((max_v_t, max_v_t_prev), axis=2), axis=2), axis=2)
    v_t_scale = v_t/max_v
    v_t_prev_scale = v_t_prev/max_v 
    accel = (v_t_scale - v_t_prev_scale)/dt
    x_eng = np.concatenate((x_eng, v_t_scale, v_t_prev_scale, accel), axis=2)
    return x_eng

'''
input: a batch of observations x_hat[t] = [x[t]] shape=(batch size, sequence length, feature length)
output: a stack of observations x_hat[t] = [x[t], x[t-1], ..., x[t-k]] shape=(batch size, sequence length, feature length*k)
k is stack heigth
input and output are numpy arrays
'''
def stack_observations(x, obs_len=3, stack_height=0):
    act_len = x.shape[2] - obs_len
    feat_len = obs_len + act_len
    x_stack = x
    x_fwd = np.roll(x, shift=-1, axis=1)
    padding_zero = np.zeros(x[:,-1:,:].shape)
    x_fwd[:,-1:,:] = padding_zero
    x_stack = np.concatenate((x_fwd[:,:,0:obs_len], x_stack), axis=2)
    for delay in range(1, stack_height+1):
        x_delay = np.roll(x, shift=delay, axis=1)
        padding_zero = np.zeros(x[:,0:delay,:].shape)
        x_delay[:,0:delay,:] = padding_zero
        x_stack = np.concatenate((x_stack, x_delay), axis=2)
        
    return x_stack

def pred_loss(pred_net, x, y, training, k_d=0.5, k_i=0.1, k_b=0.5, obs_len=3, stack_height=0, eng=False, device='cpu'):
    # y_ = [Batch size, sequence length, prediction dimension/length]
    #Adding batch dimension if missing
    loss_object = torch.nn.MSELoss()
    if x.ndim < 3:
        x = np.expand_dims(x, axis=0)
    if y.ndim < 3:
        y = np.expand_dims(y, axis=0)
    if eng:
        x_eng = eng_feature(x)
        x_transform = x_eng
    else:
        x_stack = stack_observations(x, obs_len=obs_len, stack_height=stack_height)
        x_transform = x_stack

    device = torch.device(device)
    x_transform = torch.from_numpy(x_transform).to(device, dtype=torch.float32)
    y = torch.from_numpy(y).to(device, dtype=torch.float32)
    if training:
        pred_net.train()
    else:
        pred_net.eval()
    y_pred = pred_net(x_transform)[0]
    
    y_delay = torch.roll(y_pred, shifts=1, dims=1)
    y_delay[:,0,:] = y_pred[:,0,:]
    
    mse_loss = loss_object(y, y_pred)
    # difference penalty = (y_pred[t-1] - y_pred[t])^2
    # to penalize the lstm_predictor for changing its prediction too quickly
    difference_penalty = loss_object(y_pred, y_delay)
    # accumulation penalty
    y_error = torch.subtract(y_pred, y)
    y_squared_error = torch.square(y_error)
    accumulation_penalty = torch.mean(torch.cumsum(y_squared_error, axis=1))
    # bias penalty
    y_pred_cumsum = torch.cumsum(y_pred, axis=1)
    inverse_t = np.zeros(y_pred_cumsum.shape)
    for i in range(inverse_t.shape[1]):
        inverse_t[:, i] = (1/(i+1))*np.ones((inverse_t.shape[0], inverse_t.shape[2]))
    inverse_t_th = torch.from_numpy(inverse_t).to(device, dtype=torch.float32)
    y_pred_runavg = torch.multiply(inverse_t_th, y_pred_cumsum)
    bias_penalty = loss_object(y_pred_runavg, y)
    return mse_loss + k_d*difference_penalty + k_i*accumulation_penalty + k_b*bias_penalty

def pred_eval(pred_net, features_eval_np, labels_eval_np, label_scale_factor, 
              k_d=0.5, k_i=0.1, k_b=0.5, obs_len=3, stack_height=0, eng=False, device='cpu'):
    eval_metric = torch.nn.MSELoss()
    total_eval_loss = 0
    total_eval_metric = 0
    total_batches = 0
    label_sf_th = torch.from_numpy(label_scale_factor).to(device, dtype=torch.float32)
    pred_net.eval() # Set to evaluation mode
    for x, y in zip(features_eval_np, labels_eval_np):
        if x.ndim < 3:
            x = np.expand_dims(x, axis=0)
        if y.ndim < 3:
            y = np.expand_dims(y, axis=0)
        if eng:
            x_transform = eng_feature(x)
        else:
            x_transform = stack_observations(x, obs_len=obs_len, stack_height=stack_height)
        y_scale = y/label_scale_factor
        x_transform = torch.from_numpy(x_transform).to(device, dtype=torch.float32)
        y = torch.from_numpy(y).to(device, dtype=torch.float32)
        y_pred = pred_net(x_transform)[0]
        
        # upscaling y_pred
        y_pred_scale = y_pred*label_sf_th
        total_eval_loss += pred_loss(pred_net, x, y_scale, training=False, 
                                k_d=k_d, k_i=k_i, k_b=k_b, obs_len=obs_len, stack_height=stack_height, eng=eng, device=device)
        total_eval_metric += eval_metric(y, y_pred_scale).cpu().detach().numpy()
        total_batches += 1
        
    avg_eval_loss = total_eval_loss.cpu().detach().numpy() / total_batches
    avg_eval_metric = total_eval_metric / total_batches
    
    return (avg_eval_loss, avg_eval_metric)

class LstmOptimizer:
    def __init__(self):
        self.loss = torch.nn.MSELoss()

    def calc_loss(
        self, 
        features_batch, 
        labels_batch, 
        label_scale_factor, 
        stack_height, 
        obs_len,
    ):
        total_loss = 0
        for x, y in zip(features_batch, labels_batch):
            if x.ndim < 3:
                x = np.expand_dims(x, axis=0)
            if y.ndim < 3:
                y = np.expand_dims(y, axis=0)
            y_scale = y/label_scale_factor # downscaling true y
            x_stack = stack_observations(x, obs_len=obs_len, stack_height=stack_height)
            x_transform = x_stack
            device = torch.device(self.pred_net.device)
            x_transform = torch.from_numpy(x_transform).to(device, dtype=torch.float32)
            y_pred = self.pred_net(x_transform)[0]
            y_scale = torch.from_numpy(y_scale).to(device, dtype=torch.float32)
            mse_loss = self.loss(y_scale, y_pred)
            total_loss += mse_loss
        return total_loss

    def fit_data(
        self,
        pred_net,
        features_list_train,
        labels_list_train,
        features_list_val,
        labels_list_val,
        updater,
        label_scale_factor,
        stack_height,
        obs_len,
        batchsize=2,
        num_epochs=50,
        max_patience=3,
    ):
        self.pred_net = pred_net
        features_batch_train = pad_and_batch(features_list_train, batchsize)
        labels_batch_train = pad_and_batch(labels_list_train, batchsize)

        features_batch_val = pad_and_batch(features_list_val, batchsize)
        labels_batch_val = pad_and_batch(labels_list_val, batchsize)

        best_pred_net = copy.deepcopy(self.pred_net)
        best_val_loss = float("inf")
    
        for epoch in range(num_epochs):
            epoch_loss_avg = [] #Mean_th()
            epoch_accuracy = []
            self.pred_net.train(True)

            for x, y in zip(features_batch_train, labels_batch_train):
                # Optimize the lstm_predictor
                #Adding batch dimension if missing:
                if x.ndim < 3:
                    x = np.expand_dims(x, axis=0)
                if y.ndim < 3:
                    y = np.expand_dims(y, axis=0)
                y_scale = y/label_scale_factor # downscaling true y
                updater.zero_grad()
                l = self.calc_loss(x, y, label_scale_factor, stack_height, obs_len)
                l.backward()
                state_a = self.pred_net.state_dict().__str__() # verification code line
                updater.step()
                # verification code start
                state_b = self.pred_net.state_dict().__str__()
                if state_a == state_b:
                    print("Network pred_net not updating.")
                    
                # verification code end
                # Track progress
                epoch_loss_avg.append(l.cpu().detach().numpy())

            # End epoch
            # Calculate validation loss
            self.pred_net.eval()
            with torch.no_grad():
                val_loss = self.calc_loss(features_batch_val, labels_batch_val, label_scale_factor, stack_height, obs_len)
                val_loss = val_loss.cpu().numpy()
            if val_loss <= best_val_loss:
                best_val_loss = val_loss
                patience = max_patience
                # next line is temporary only for verification
                assert best_pred_net.state_dict().__str__() != self.pred_net.state_dict().__str__(), 'best_pred_net is synchronised with self.pred_net.'
                best_pred_net = copy.deepcopy(self.pred_net)
            else:
                print("Epoch {:02d}. Patience reduced. current val_loss: {:.3} best_val_loss: {:.3}".format(epoch+1, val_loss, best_val_loss))
                patience -= 1

            if epoch % 1 == 0:
                print("Epoch {:02d}: Loss: {:.4f}, Val_loss: {:.4f}"
                      .format(epoch+1, np.mean(epoch_loss_avg), val_loss))
            if epoch >= 20:
                if patience <= 0:
                    print(f"Early stop at epoch {epoch+1}.")
                    break
        self.pred_net = best_pred_net
        # maybe can return loss at each epoch

def squared_hellinger_distance(X, Y, k=5):
    """
    Estimate the squared Hellinger distance between two multivariate continuous distributions
    using k-nearest neighbors as described in Section 4.3 of the referenced paper.

    Parameters:
    - X: np.ndarray
        Sample from the first distribution with shape (n_samples_X, n_features).
    - Y: np.ndarray
        Sample from the second distribution with shape (n_samples_Y, n_features).
    - k: int
        Number of nearest neighbors to use in the estimation.

    Returns:
    - float
        Estimated squared Hellinger distance.
    """
    n = X.shape[0]
    m = Y.shape[0]
    d = X.shape[1]

    # Build KD-Trees for efficient neighbor searches
    tree_X = cKDTree(X)
    tree_Y = cKDTree(Y)

    # Compute distances to the k-th nearest neighbor within the same sample
    r_X = tree_X.query(X, k=k+1)[0][:, -1]  # k+1 because the point itself is included
    r_Y = tree_Y.query(Y, k=k+1)[0][:, -1]

    # Compute distances to the k-th nearest neighbor in the opposite sample
    s_X = tree_Y.query(X, k=k)[0][:, -1]
    s_Y = tree_X.query(Y, k=k)[0][:, -1]

    # Compute the Hellinger affinity estimator
    A_XY = (1/n) * np.sqrt((n-1)/m) * np.sum((r_X / s_X) ** (d / 2))
    A_YX = (1/m) * np.sqrt((m-1)/n) * np.sum((r_Y / s_Y) ** (d / 2))
    
    H_XY_sq = 1 - (A_XY*(math.factorial(k-1)**2))*(rgamma(k-0.5)*rgamma(k+0.5))
    H_YX_sq = 1 - (A_YX*(math.factorial(k-1)**2))*(rgamma(k-0.5)*rgamma(k+0.5))
    
    # H_sq = 1 - (alpha + beta) / (n + m)
    H_sq = (H_XY_sq + H_YX_sq)/2
    # print(f'H_XY_sq = {H_XY_sq:.4f}, H_YX_sq = {H_YX_sq:.4f}')
    
    return H_sq

def wasserstein_distance(X, Y, p=2):
    """Compute p-Wasserstein distance between two empirical distributions."""
    n = X.shape[0]
    m = Y.shape[0]
    a = np.ones((n,)) / n  # uniform weights
    b = np.ones((m,)) / m
    M = ot.dist(X, Y, metric='euclidean') ** p
    W_p = ot.emd2(a, b, M, numItermax=100000, numThreads=1)  # returns the p-th power of the distance
    return W_p ** (1 / p)

distance_type_dict = {
    'hellinger': squared_hellinger_distance,
    'wasserstein': wasserstein_distance
}

def compute_distance(traj, samples_b, dist_fn, kwargs):
    # traj, samples_b, dist_fn, kwargs = args
    try:
        return dist_fn(X=traj, Y=samples_b, **kwargs)
    except Exception as e:
        import os
        print(f'Process with id={os.getpid()}. traj.shape={traj.shape}, samples_b.shape={samples_b.shape}.')
        print(f'Error:\n{e}')
        return None
        

def rank_trajectories(trajs_1, trajs_b, obs_len, act_len, distance_type='hellinger', stack_size_obs=None, stack_size_act=None, **kwargs):
    '''
    Compare individual trajectories in trajs_1 against trajs_b.
    Output: Array with the rank of each trajectory.
    '''
    # print(f'trajs_1[0].shape = {trajs_1[0].shape}, trajs_b[0] = {trajs_b[0].shape}')
    if trajs_1[0].shape[1] > obs_len + act_len:
        obs_range = list(range(0,obs_len))
        act_range = list(range(obs_len*stack_size_obs, obs_len*stack_size_obs+act_len))
        for i in range(len(trajs_1)):
            # if i==0: print(f'(Before) trajs_1[{i}][:7, :] = {trajs_1[i][:7, :]}')
            trajs_1[i] = trajs_1[i][:, (*obs_range, *act_range)]
            # if i==0: print(f'(After) trajs_1[{i}][:7, :] = {trajs_1[i][:7, :]}')
        for i in range(len(trajs_b)):
            trajs_b[i] = trajs_b[i][:, (*obs_range, *act_range)]
    # Process trajectories in trajs_b
    # Process = do appropriate stacking and flatten the trajectories
    for i in range(len(trajs_b)):
        traj = np.expand_dims(trajs_b[i], axis=0)
        traj_new = stack_observations(traj, obs_len=obs_len, stack_height=0)
        # Remove the last timestep of traj_new
        traj_new = traj_new[:,:-1,:]
        if i > 0:
            samples_b = np.append(samples_b, traj_new.squeeze(axis=0), axis=0)
        else:
            samples_b = traj_new.squeeze(axis=0)
    
    # Process trajectories in trajs_1
    # Process = do appropriate stacking
    trajs_1_process = []
    for i in range(len(trajs_1)):
        traj = np.expand_dims(trajs_1[i], axis=0)
        traj_new = stack_observations(traj, obs_len=obs_len, stack_height=0)
        # Remove the last timestep of traj_new
        traj_new = traj_new[:,:-1,:]
        trajs_1_process.append(traj_new.squeeze(axis=0))
        # print(f'len(trajs_1_process) = {len(trajs_1_process)}, trajs_1_process[{i}].shape = {trajs_1_process[i].shape}')

    print(f'samples_b.shape = {samples_b.shape}, trajs_1_process[0].shape = {trajs_1_process[0].shape}')
    # Calculate squared Hellinger distance
    dist_fn = distance_type_dict[distance_type]
    args_list = [(trajs_1_process[i], samples_b, dist_fn, kwargs) for i in range(len(trajs_1_process))]
    with multiprocessing.Pool(processes=10) as pool:
        # temp_result = pool.starmap(dist_fn, args_list)
        temp_result = pool.starmap(compute_distance, args_list)
    
    dists = np.array(temp_result)

    # Assign rank based on Hellinger distance
    # Index of trajectories arranged in ascending order of distance [index of closest traj, index of 2nd closest traj, ...]
    # So if you print H_sqs(np.argsort(H_sqs)) you will have distances in ascending order 
    # H_sort_index = np.argsort(H_sqs)
    dist_sort_index = np.argsort(dists)
    print(f'dists[dist_sort_index] = {dists[dist_sort_index]}')

    # return H_sort_index
    return dist_sort_index

def print_round_context(idx, round_len, labels):
    round_nos = -np.ones(idx.shape, dtype=int)
    labels_1 = np.array([traj[0] for traj in labels])
    for i in range(len(round_len)):
        temp_bool = np.logical_and(idx/np.sum(round_len[:(i+1)])<1.0, round_nos==-1)
        # print(f'temp_bool = {temp_bool}')
        round_nos[temp_bool] = i
    print(f'Indexes = {idx}')
    print(f'Corresponding round number: {round_nos}')
    for i in range(len(round_len)):
        temp_bool = round_nos == i
        print(f'Number of deleted trajectories from round {i}: {np.sum(temp_bool)}')
        print(f'Percentage of deleted trajectories from round {i}: {np.mean(temp_bool)*100:.2f}')
    print(f'Corresponding context: {labels_1[idx]}')

def calc_dist_distrs_helper(args):
    '''
    Helper to the calc_dist_distrs.
    Return the distance between traj_set_1 and traj_set_2.
    '''
    try: 
        traj_set_1, traj_set_2, obs_len, act_len, distance_type, stack_size_obs, stack_size_act, kwargs = args
        if traj_set_1[0].shape[1] > obs_len + act_len:
            obs_range = list(range(0,obs_len))
            act_range = list(range(obs_len*stack_size_obs, obs_len*stack_size_obs+act_len))
            for i in range(len(traj_set_1)):
                # if i==0: print(f'(Before) trajs_1[{i}][:7, :] = {trajs_1[i][:7, :]}')
                traj_set_1[i] = traj_set_1[i][:, (*obs_range, *act_range)]
                # if i==0: print(f'(After) trajs_1[{i}][:7, :] = {trajs_1[i][:7, :]}')
            for i in range(len(traj_set_2)):
                traj_set_2[i] = traj_set_2[i][:, (*obs_range, *act_range)]
        
        # Process trajectories in traj_set_1
        # Process = do appropriate stacking and flatten the trajectories
        for i in range(len(traj_set_1)):
            traj = np.expand_dims(traj_set_1[i], axis=0)
            traj_new = stack_observations(traj, obs_len=obs_len, stack_height=0)
            # Remove the last timestep of traj_new
            traj_new = traj_new[:,:-1,:]
            if i > 0:
                samples_1 = np.append(samples_1, traj_new.squeeze(axis=0), axis=0)
            else:
                samples_1 = traj_new.squeeze(axis=0)
        
        # Process trajectories in traj_set_2
        # Process = do appropriate stacking and flatten the trajectories
        for i in range(len(traj_set_2)):
            traj = np.expand_dims(traj_set_2[i], axis=0)
            traj_new = stack_observations(traj, obs_len=obs_len, stack_height=0)
            # Remove the last timestep of traj_new
            traj_new = traj_new[:,:-1,:]
            if i > 0:
                samples_2 = np.append(samples_2, traj_new.squeeze(axis=0), axis=0)
            else:
                samples_2 = traj_new.squeeze(axis=0)
        
        # print(f'samples_1.shape = {samples_1.shape}, samples_2.shape = {samples_2.shape}')
        # Calculate the distance
        dist_fn = distance_type_dict[distance_type]
        return dist_fn(X=samples_1, Y=samples_2, **kwargs)
    except Exception as e:
        import os
        traj_set_1, traj_set_2, obs_len, act_len, distance_type, stack_size_obs, stack_size_act, kwargs = args
        print(f'Process with id={os.getpid()}. len(traj_set_1)={len(traj_set_1)}, len(traj_set_2)={len(traj_set_2)}.')
        print(f'Error:\n{e}')
        return None
    

def calc_dist_distrs(trajs_1, trajs_b, obs_len, act_len, distance_type='hellinger', stack_size_obs=None, stack_size_act=None, **kwargs):
    '''
    Input:
    trajs_1 = list of set of trajectories
    trajs_b = set of trajectories
    Output: Print distance between trajs_b and each set of trajectories in trajs_1
    '''
    args_list = [(trajs_1[i], trajs_b, obs_len, act_len, distance_type, stack_size_obs, stack_size_act, kwargs) for i in range(len(trajs_1))]
    with multiprocessing.Pool(processes=10) as pool:
        temp_result = pool.map(calc_dist_distrs_helper, args_list)

    dists = np.array(temp_result)
    print(f'Distance between trajs_b and each set of trajectories in trajs_1, dists = {dists}')

def calc_spread_helper(args):
    '''
    Helper for calc_spread function
    Return the distance between each trajectory in trajs_1 and the trajectory traj_b
    '''
    try:
        trajs_1, traj_b, distance_type, kwargs = args
        dists = [0]*len(trajs_1)
        dist_fn = distance_type_dict[distance_type]
        for i in range(len(trajs_1)):
            dists[i] = dist_fn(X=trajs_1[i], Y=traj_b, **kwargs)
        return dists
    except Exception as e:
        import os
        trajs_1, traj_b, distance_type, kwargs = args
        print(f'Process with id={os.getpid()}. len(trajs_1)={len(trajs_1)}, traj_b.shape={traj_b.shape}.')
        print(f'Error:\n{e}')
        return None

def calc_spread(trajs_1, obs_len, act_len, distance_type='hellinger', stack_size_obs=None, stack_size_act=None, **kwargs):
    '''
    Calculate the spread of a distribution
    Spread is defined as the mean diameter of the distribution
    '''
    if trajs_1[0].shape[1] > obs_len + act_len:
        obs_range = list(range(0,obs_len))
        act_range = list(range(obs_len*stack_size_obs, obs_len*stack_size_obs+act_len))
        for i in range(len(trajs_1)):
            trajs_1[i] = trajs_1[i][:, (*obs_range, *act_range)]

    # Process trajectories in trajs_1
    trajs_1_process = []
    for i in range(len(trajs_1)):
        traj = np.expand_dims(trajs_1[i], axis=0)
        traj_new = stack_observations(traj, obs_len=obs_len, stack_height=0)
        # Remove the last timestep of traj_new
        traj_new = traj_new[:,:-1,:]
        trajs_1_process.append(traj_new.squeeze(axis=0))

    dists = np.zeros((len(trajs_1), len(trajs_1)))
    args_list = [(trajs_1[i+1:], trajs_1[i], distance_type, kwargs) for i in range(len(trajs_1)-1)]
    with multiprocessing.Pool(processes=10) as pool:
        temp_result = pool.map(calc_spread_helper, args_list)

    for i in range(dists.shape[0]-1):
        dists[i, i+1:] = temp_result[i]
        # Filling the lower triangular part
        # The lower triangular part is transpose of the upper triangular part
        if i > 0:
            dists[i, :i] = dists[:i, i]
    dists[-1, :-1] = dists[:-1, -1]
    # print(f'dists = {dists}')
    diameters = np.max(dists, axis=1)
    mean_diameter = np.mean(diameters)
    std_diameter = np.std(diameters)
    print(f'Diameter: {mean_diameter:.3f} +/- {std_diameter:.3f}')

class TrajectoryBuffer:
    def __init__(self, max_steps):
        self.max_steps = max_steps
        self.buffer = deque()
        self.current_steps = 0

    def add_trajectory(self, trajectory):
        # Add new trajectory
        self.buffer.append(trajectory)
        self.current_steps += trajectory.shape[0]

        # Remove old trajectories while we exceed the limit
        while self.current_steps > self.max_steps:
            removed_traj = self.buffer.popleft()
            self.current_steps -= removed_traj.shape[0]

    def get_data(self):
        return list(self.buffer)