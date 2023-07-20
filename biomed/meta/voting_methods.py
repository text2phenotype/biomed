from typing import Dict, Callable

import numpy as np
from scipy.stats import entropy
from sklearn.preprocessing import normalize

from text2phenotype.common.log import operations_logger
from biomed.common.predict_results import PredictResults
from biomed.constants.model_enums import ModelType, VotingMethodEnum
from biomed.models.model_cache import ModelCache
from biomed.models.get_model import get_model_filename_from_model_folder
from biomed.models.voter_model import VoterModel

# default threshold for threshold voter
DEFAULT_THRESHOLD = 0.5

# list of trainable voting methods; these methods will require training in order to be used as a voting method
TRAINABLE_VOTING_METHODS = {VotingMethodEnum.rf_classifier}


def is_trained_voter(voting_method: VotingMethodEnum):
    """Test if given voting method is trainable"""
    return voting_method in TRAINABLE_VOTING_METHODS


def ensemble_encoding_from_input_dict(
    input_dict: Dict[str, PredictResults], num_tokens: int, num_classes: int
) -> np.ndarray:
    """
    Concatenate collection of model predicted_probs
    Model order is concatenated by ascending sort of the constituent model names

    :param input_dict: dictionary of PredictResults, keyed by model name
    :param num_tokens: how many tokens we expect to be in the second dimension of the ndarray
    :param num_classes: how many label classes we expect for the given model type
    :return: np.ndarray, prediction probabilities with shape (num_models, num_tokens, num_classes)
    """
    num_models = len(input_dict)
    ensemble_encoding = np.zeros((num_models, num_tokens, num_classes))
    for idx, model_name in enumerate(sorted(input_dict.keys())):
        prediction: PredictResults = input_dict[model_name]

        # encoding_matrix is [num_tokens, num_classes]
        encoding_matrix = prediction.predicted_probs
        # ensemble_encoding is [model_idx, num_tokens, num_classses]
        ensemble_encoding[idx, 0:encoding_matrix.shape[0], 0:encoding_matrix.shape[1]] = encoding_matrix

    operations_logger.debug("Beginning formatting ensemble output")
    return ensemble_encoding


def entropy_weight_matrix_from_ensemble_encoding(ensemble_encoding: np.ndarray):
    """
    Create a (transposed) weight matrix based on the inverse entropy for predictions within each model.
    The entropy scores are normalized for each model, so that they sum to 1

    :param ensemble_encoding: ndarray, ensemble probabilities, shape (n_models, n_tokens, n_classes)
    :return: np.ndarray
        entropy weight matrix, shape (n_tokens, n_models)
    """
    weight_matrix = np.zeros((ensemble_encoding.shape[1], ensemble_encoding.shape[0]))
    for model_ix in range(ensemble_encoding.shape[0]):
        # weight vector is len = num tokens
        # entropy is calculated at the token level, entropy is low for "confident" predictions
        # if there's only one class then weight == 1
        weight_vector = 1 / entropy(np.transpose(ensemble_encoding[model_ix, :, :]))
        # weight matrix is [num_tokens, num_models]
        weight_matrix[0: len(weight_vector), model_ix] = weight_vector

    # normalize the weight matrix such that weights for each model on any token sum to 1
    return normalize(weight_matrix, norm="l1", axis=1)


def weighted_entropy_voting(ensemble_encoding: np.ndarray, **kwargs) -> PredictResults:
    """
    Return the inverse entropy weighted average of token probabilities across models

    :param ensemble_encoding: ndarray, shape (n_models, n_tokens, n_classes), contains the class probabilities for
        each model, token, and class (including NA)
    :param kwargs: allows for blanket usage of any voting system though the enum
        not used in weighted_entropy_voting
    :return: PredictResults Object
    This function assigns the probability score based on weighted_entropy voting
     and assigns the category to be the one with the highest weighted_entropy voting score
    """
    num_tokens, num_classes = ensemble_encoding.shape[1:]
    weight_matrix = entropy_weight_matrix_from_ensemble_encoding(ensemble_encoding)
    predicted_probs = np.zeros((num_tokens, num_classes))

    for i in range(num_tokens):
        # turns out that iterating over tokens is more efficient than using eigsum for large arrays
        predicted_probs[i, :] = np.dot(weight_matrix[i, :], ensemble_encoding[:, i, :])

    predicted_cat = np.argmax(predicted_probs, axis=1)

    predict_res = PredictResults(predicted_probs=predicted_probs, predicted_cat=predicted_cat)
    return predict_res


def threshold_voting(ensemble_encoding: np.ndarray, **kwargs) -> PredictResults:
    """
    This function assigns the probability score based on weighted_entropy voting
    and assigns the category:
    If any model believed that the probability of the entity belonging to a category with col_index != 0 is greater
        than the threshold, then the category is whichever col_index > 0 has the highest weighted_entropy score
    Otherwise, the NA label is selected bc it's known to be NA;
        scaled probabilities are still returned for the NA tokens

    :param ensemble_encoding: ndarray, shape (n_models, n_tokens, n_classes), contains the class probabilities for
        each model, token, and class (including NA)
    :param kwargs: allows for blanket usage of any voting system though the enum
        threshold_voting expects:
            threshold: float, probability used for returning entities with NA probs LESS THAN this value
    :return: PredictResults Object
    """
    num_tokens, num_classes = ensemble_encoding.shape[1:]
    weight_matrix = entropy_weight_matrix_from_ensemble_encoding(ensemble_encoding)
    predicted_probs = np.zeros((num_tokens, num_classes))

    # Find scaled probabilities for all tokens
    # NOTE that we don't use the scaled probabilities for any token that the min NA class is above threshold
    for i in range(num_tokens):
        predicted_probs[i, :] = np.dot(weight_matrix[i, :], ensemble_encoding[:, i, :])

    predicted_cat = np.zeros(num_tokens)
    # indices of the tokens where the minimum NA probability across models is LESS than threshold
    # This should yield tokens where the probability is gated on the NA being below threshold
    potential_pos_index = np.where(
        ensemble_encoding[:, :, 0].min(axis=0) < kwargs.get("threshold", DEFAULT_THRESHOLD)
    )[0]
    # taking the argmax of positive classes for non na potentials
    predicted_cat[potential_pos_index] = np.argmax(predicted_probs[potential_pos_index, 1:], axis=1) + 1

    predict_res = PredictResults(predicted_probs=predicted_probs, predicted_cat=predicted_cat)
    return predict_res


def threshold_category_voting(ensemble_encoding: np.ndarray, **kwargs) -> PredictResults:
    """
    This function assigns the probability score based on weighted_entropy voting
     and assigns the category:
        if any model believed that the probability of the entity belonging to a "threshold category" is greater than the
         threshold, then the category assigned will be the threshold category with the highest weighted_entropy
          probability score
        if no model believed that an entity belonged to a threshold category then the category is assigned to be
        the highest weighted_entropy prob

    :param ensemble_encoding: ndarray, shape (n_models, n_tokens, n_classes), contains the class probabilities for
        each model, token, and class (including NA)
    :param kwargs: allows for blanket usage of any voting system though the enum
        threshold_category_voting() expects:
            threshold: float, probability used for returning non_na entity in listed categories greater than this value
            threshold_categories: list of integers for the label classes to use for thresholding
    :return: PredictResults Object

    """
    num_tokens, num_classes = ensemble_encoding.shape[1:]
    weight_matrix = entropy_weight_matrix_from_ensemble_encoding(ensemble_encoding)

    predicted_probs = np.zeros((num_tokens, num_classes))

    threshold_categories = kwargs.get("threshold_categories")
    threshold = kwargs.get("threshold", DEFAULT_THRESHOLD)

    non_na_index = np.where(
        ensemble_encoding[:, :, threshold_categories].max(axis=2).max(axis=0) > threshold
    )[0]

    for i in range(num_tokens):
        predicted_probs[i, :] = np.dot(weight_matrix[i, :], ensemble_encoding[:, i, :])

    predicted_category = np.argmax(predicted_probs, axis=1)
    # taking argmax of threshold categories for non na potentials
    predicted_category[non_na_index] = np.array(threshold_categories)[
        np.argmax(predicted_probs[non_na_index][:, threshold_categories], axis=1)
    ]

    predict_res = PredictResults(predicted_probs=predicted_probs, predicted_cat=predicted_category)
    return predict_res


def model_avg_voting(ensemble_encoding: np.ndarray, **kwargs) -> PredictResults:
    """
    Use uniform average of prediction probabilities for each label over all models

    :param ensemble_encoding: ndarray, shape (n_models, n_tokens, n_classes), contains the class probabilities for
        each model, token, and class (including NA)
    :param kwargs: allows for blanket usage of any voting system though the enum
    :return: PredictResults Object
    """
    n_models = ensemble_encoding.shape[0]
    weights = np.ones(n_models) / n_models
    return model_weighted_avg_voting(ensemble_encoding, weights=weights)


def model_weighted_avg_voting(ensemble_encoding: np.ndarray, **kwargs) -> PredictResults:
    """
    Use selected weights (or uniform weights for regular averaging) to scale the probabilities by "trust"
    The weights may come from a trained optimizer (eg scipy.optimize.differential_evolution)
    that identifies the weights that produce the best output based on an evaluation dataset (eg PHI data)

    NOTE: this method does NOT use entropy scaling. Applying entropy scaling on top of the model scaling may
    improve results. Or not.

    TODO: how do we get hardcoded model-specific values into this method?
     - EnsembleMetadata must contain the `weights` array; define as constant?

    :param ensemble_encoding: ndarray, shape (n_models, n_tokens, n_classes), contains the class probabilities for
        each model, token, and class (including NA)
    :param kwargs: allows for blanket usage of any voting system though the enum
        model_weighted_avg_voting() expects:
            weights: arraylike, length=n_models, multiplier across all entities and probabilities
                Acts as a "trust" coefficient, for how much weight we should give a model in support
                of a given set of model preditions for a single entity
    :return: PredictResults Object
    """
    weights = kwargs.get("weights")
    n_models = ensemble_encoding.shape[0]
    # use uniform weights if no weights are passed in
    weights = np.ones(n_models) / n_models if weights is None else weights
    # NOTE(MJP): should this be L1 or L2 norm?
    predicted_probs = normalize(np.average(ensemble_encoding, weights=weights, axis=0), norm='l1', axis=1)
    predicted_category = np.argmax(predicted_probs, axis=1)
    predict_res = PredictResults(predicted_probs=predicted_probs, predicted_cat=predicted_category)
    return predict_res


def rf_classifier_voting(ensemble_encoding: np.ndarray, model_cache: ModelCache, model_type: ModelType,
                         voting_model_folder: str, **kwargs) -> PredictResults:
    """
    Use a trained model specified in "$model_type/$voting_model_folder" to combine the model probabilities
    in ensemble_encoding
    The model will be loaded into model_cache the first time it is loaded, and will be cached for all subsequent calls

    :param ensemble_encoding: ndarray, shape (n_models, n_tokens, n_classes), contains the class probabilities for
        each model, token, and class (including NA)
    :param model_cache: ModelCache object, maintains persistence of voting model to avoid reloading
    :param model_type: ModelType, defines the model type (collection of expected label classes),
        used as the name under resources/files to find the voting model
    :param voting_model_folder: str, the folder name holding the target voting model
        TODO: should this have the filename? Do we want a consistent filename, or just find the first .joblib file?
    :param kwargs: allows for blanket usage of any voting system though the enum
    :return: PredictResults Object
    """
    if ensemble_encoding.shape[0] == 1:
        # only one model, don't even load the voter
        predicted_probs = ensemble_encoding.squeeze(axis=0)
    else:
        model_file_path = get_model_filename_from_model_folder(voting_model_folder, model_type=model_type, suffix=".joblib")
        # load a cached model
        estimator = model_cache.model_sklearn(model_type, model_file_path)
        ensemble_encoding_reshape = VoterModel.reshape_3d_to_2d(ensemble_encoding)
        # vote
        predicted_probs = estimator.predict_proba(ensemble_encoding_reshape)

    predicted_category = np.argmax(predicted_probs, axis=1)
    predict_res = PredictResults(predicted_probs=predicted_probs, predicted_cat=predicted_category)
    return predict_res


# dict from a voting method enum constant to the function itself
# the constant comes from model_enums, we define the methods in this module
VOTING_METHOD_MAP = {
    VotingMethodEnum.weighted_entropy: weighted_entropy_voting,
    VotingMethodEnum.threshold: threshold_voting,
    VotingMethodEnum.threshold_categories: threshold_category_voting,
    VotingMethodEnum.model_avg: model_avg_voting,
    VotingMethodEnum.model_weighted_avg: model_weighted_avg_voting,
    VotingMethodEnum.rf_classifier: rf_classifier_voting,
}


def get_voting_func(voting_method: VotingMethodEnum) -> Callable:
    """
    Thin wrapper to retrieve method from a VotingMethod enum
    :param voting_method: VotingMethod
    :return: Callable
        The associated function handle to the voting function
    """
    return VOTING_METHOD_MAP[voting_method]


def vote(
    prediction_dict: Dict[str, PredictResults],
    num_tokens: int,
    num_classes: int,
    voting_method: VotingMethodEnum,
    include_raw_probs: bool = False,
    **kwargs
) -> PredictResults:
    """
    Return the probabilities and predictions collapsed across all ensembled models, using the given voting_method

    :param prediction_dict: dictionary created during ensembling of model_id to Predict Results objects
    :param num_tokens: number of tokens for which there are predictions
    :param num_classes: number of label classes (including NA)
    :param voting_method: VotingMethodEnum, enum to which voting functions can be used in voting
    :param include_raw_probs: bool flag to store full ensemble probabilities in PredictResults
    :param kwargs: allows for blanket usage of any voting system though the enum
    :return: PredictResults Object
    """
    ensemble_encoding = ensemble_encoding_from_input_dict(
        input_dict=prediction_dict, num_tokens=num_tokens, num_classes=num_classes
    )
    voter = get_voting_func(voting_method)
    pred_results = voter(ensemble_encoding=ensemble_encoding, **kwargs)

    if include_raw_probs:
        pred_results.raw_probs = ensemble_encoding

    return pred_results
