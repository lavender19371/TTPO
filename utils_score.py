import os

# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import numpy as np
import pyiqa
import glob

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def get_BT_reward(reward1, reward2, lowerbetter=False):
    if not lowerbetter:
        return sigmoid(reward1-reward2)
    return sigmoid(reward2-reward1)

def get_zscore_matrix(iqa_matrix: np.ndarray, lowerbetter):
    mean = np.mean(iqa_matrix, axis=0)
    std = np.std(iqa_matrix, axis=0)
    z_score_iqa_matrix = (iqa_matrix - mean) / std
    z_score_iqa_matrix = z_score_iqa_matrix * (-1)**np.array(lowerbetter, dtype=int)
    return z_score_iqa_matrix

def get_simple_win_reject(reward_matrix):
    averaged_reward = np.mean(reward_matrix, axis=1)
    # print(averaged_reward)
    return np.argmax(averaged_reward), np.argmin(averaged_reward)

def get_entropy_win_reject(reward_matrix):
    weights = entropy_weights_np(reward_matrix)
    weighted_reward = reward_matrix @ weights
    return np.argmax(weighted_reward), np.argmin(weighted_reward)

def get_iqa_matrix(path_list, metric_list, aes_mark=None):
    lowerbetter_list = []
    iqa_matrix = np.zeros((len(path_list), len(metric_list)))
    for idxm in range(len(metric_list)):
        metric = metric_list[idxm]
        lowerbetter_list.append(metric.lower_better)
        for idxp in range(len(path_list)):
            if metric.__class__.__name__.lower() == 'inferencemodel':
                iqa_matrix[idxp, idxm] = metric(path_list[idxp]).cpu().item()
            else:
                iqa_matrix[idxp, idxm] = metric.get_score(path_list[idxp]).cpu().item()

    return iqa_matrix, lowerbetter_list

def entropy_weights_np(reward_matrix):
    reward_matrix = reward_matrix - np.min(reward_matrix, axis=0) + 0.000001
    p = reward_matrix / reward_matrix.sum(axis=0, keepdims=True)
    entropy = -np.sum(p * np.log(p), axis=0) / np.log(reward_matrix.shape[0])
    return (1-entropy) / (1-entropy).sum()

def cal_zscore(file_list, metric_list, aes_mark=None):
    iqa_metrix, lowerbetter_list = get_iqa_matrix(file_list, metric_list, aes_mark)
    reward_matrix = get_zscore_matrix(iqa_metrix, lowerbetter_list)
    win, reject = get_simple_win_reject(reward_matrix)
    # win, reject = get_entropy_win_reject(reward_matrix)  ## you can also try entropy-based auto-weighting
    return file_list[win], file_list[reject]
    
