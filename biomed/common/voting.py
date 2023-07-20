from typing import List
import numpy as np
from biomed.common.matrix_construction import indexes_for_token


def guard_dimension(matrix, num_tokens: int):
    """
    :param matrix: 2D numpy array -- prediction matrix (number of sequences X number of classes)
    :param num_tokens: number of tokens
    :return: None (raise Error if dimension is wrong, unrecoverable error)
    """
    if num_tokens == matrix.shape[0]:
        return True
    if num_tokens != (matrix.shape[0] + matrix.shape[1] - 1):
        raise ValueError("Dimension Error: Dimension of Matrix and token list don't match")


def vote_majority(class_matrix, num_tokens: int, window_size: int):
    """
    :param class_matrix: a matrix of dimension [#number of seqs, time_step] type is integer indicating one class.
    :param num_tokens: length of token list
    :return: a list of predicted class for each token.
    """
    predicted_labels = list()

    # self.guard_dimension(class_matrix, num_tokens)

    for i in range(num_tokens):
        x_dim, y_dim = indexes_for_token(i, window_size, class_matrix.shape[0])
        curr_voting = class_matrix[tuple(x_dim), tuple(y_dim)]
        counts = np.bincount(curr_voting)
        predicted_labels.append(np.argmax(counts))

    return np.array(predicted_labels)


def vote_with_weight(prob_matrix, num_tokens: int, window_size: int, use_max=False):
    """
    Combines predictions for each token across multiple sequences (e.g. timesteps), assuming stride length of 1
    :param: prob_matrix: array of predicted probabilities of dimension [num_sequences, window_size, num_classes]
    :param: use_max, if True, use the prediction vector with the maximum probability for the 2nd class
     if False, take the average of the prediction vector
    :return: 2d numpy array, (i,j) is the probability for ith token to be in jth class.
    """

    guard_dimension(prob_matrix, num_tokens)

    num_class = prob_matrix.shape[2]
    # numpy ndarray holding the weighted probability for each token for each class
    voted_prob_matrix = np.zeros((num_tokens, num_class))
    for i in range(num_tokens):
        # identify the sequence (x), window (y) coordinates of a token's representation
        x_dim, y_dim = indexes_for_token(i, window_size, prob_matrix.shape[0])
        curr_prob_matrix = prob_matrix[tuple(x_dim), tuple(y_dim)]

        # maybe apply log transformation before column-wise bc some prob value close to 0,
        # sum curr_prob_matrix for each column and normalize them could also softmax the summarized vector
        # instead of simple summation, prob for jth class = exp(pj)/sum{i}(exp(pi)), now just pj/sum(pj)
        # TODO: maximum is maximum of the 2nd class? if we want max sequence, we can implement, but I don't see why that'd be preferable
        if use_max:
            voted_prob_matrix[i] = curr_prob_matrix[np.argmax(curr_prob_matrix[:, 1]), :]
        else:
            # TODO: This is equivalent to average
            total = np.sum(np.sum(curr_prob_matrix, axis=0))
            # expected to return np.nan if divide by zero
            voted_prob_matrix[i] = np.sum(curr_prob_matrix, axis=0) / total
    return voted_prob_matrix


def construct_sample_weight(class_weight: dict, y_reshaped_np: np.ndarray):
    """
    :param class_weight: dict, key=label, value=weight; assumes weight=1 for unspecified labels
    :param y_reshaped_np: the training label Y, shape=(n_sequences, n_timesteps, n_classes)
        Can also be 2D array with sparse labels (not one-hot)
    :return: 2D numpy array each element (i, j) is the weight for ith seq, jth timestamp,
    should be looked up from phi_weight by the class type
    """
    sample_weights = np.ones((y_reshaped_np.shape[0], y_reshaped_np.shape[1]))
    is_one_hot_labels = y_reshaped_np.ndim > 2
    for i in range(y_reshaped_np.shape[0]):
        for j in range(y_reshaped_np.shape[1]):
            key = np.argmax(y_reshaped_np[i][j]) if is_one_hot_labels else y_reshaped_np[i][j]
            if key in class_weight:
                sample_weights[i][j] = class_weight[key]
            elif str(key) in class_weight:
                sample_weights[i][j] = class_weight[str(key)]
    return sample_weights


def sparse_sample_weight(class_weight, label_list: List[int]):
    """
    :param class_weight: dict, key=label, value=weight; assumes weight=1 for unspecified labels
    :param label_list: the training labels for a single window, sparse encoding
        ie, list of the label index values
    :return: 2D numpy array each element (i, j) is the weight for ith seq, jth timestamp,
    should be looked up from phi_weight by the class type
    """
    window_len = len(label_list)
    sample_weights = np.ones(window_len)
    for j in range(window_len):
        key = label_list[j]
        if key in class_weight.keys():
            sample_weights[j] = class_weight[key]
        elif str(key) in class_weight:
            sample_weights[j] = class_weight[str(key)]
    return sample_weights
