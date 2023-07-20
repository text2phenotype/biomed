"""
Collection of model metrics

TODO: add calculations done in ModelReports (currently part of ModelBase) here
"""

from typing import Iterable

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

from text2phenotype.common.log import operations_logger

# where we generally expect the 'na' label to be in the label index
DEFAULT_NA_INDEX = 0

def recall_binary(y_true, y_pred):
    """
    Calculate recall, collapsing all non-negative classes into a positive class

    NOTE: this will NOT give the same result as other categorical recall metrics, eg macro or micro averaging
    It simply collapses all positive categories (not 'na')

    :param y_true: tf.Tensor
    :param y_pred: tf.Tensor
    :return:
    """
    y_true_pos = K.clip(y_true, 0, 1)
    true_positives = K.cast(K.sum(K.round(K.clip(y_true_pos * y_pred, 0, 1))), "float")
    all_positives = K.cast(K.sum(y_true_pos), "float")
    recall_out = true_positives / (all_positives + K.epsilon())
    return recall_out


def precision_binary(y_true, y_pred):
    """
    Calculate precision, collapsing all non-negative classes into a positive class

    NOTE: this will NOT give the same result as other categorical recall metrics, eg macro or micro averaging
    It simply collapses all positive categories (not 'na')
    :param y_true: tf.Tensor
    :param y_pred: tf.Tensor
    :return:
    """
    y_true_pos = K.clip(y_true, 0, 1)
    true_positives = K.cast(K.sum(K.round(K.clip(y_true_pos * y_pred, 0, 1))), "float")
    predicted_positives = K.cast(K.sum(K.round(K.clip(y_pred, 0, 1))), "float")
    precision_out = true_positives / (predicted_positives + K.epsilon())
    return precision_out


def f1_score_binary(y_true, y_pred):
    prec = precision_binary(y_true, y_pred)
    rec = recall_binary(y_true, y_pred)
    return (2 * ((prec * rec) / (prec + rec + K.epsilon()))).numpy()


def f1_score_no_na(y_true, y_pred):
    """
    Calculate the micro F1 score, ignoring all true positive matches on the 'na' tokens

    Doing it the slow way here because the input type may be a tf.Tensor
    :param y_true: Iterable
    :param y_pred: Iterable
    :return:
    """
    tp_count = 0
    fp_count = 0
    fn_count = 0
    for y, y_hat in zip(y_true, y_pred):
        if y == y_hat and y == 0:
            continue
        if y == y_hat:
            tp_count += 1
        else:
            if y:
                fn_count += 1
            if y_hat:
                fp_count += 1
    f1_score_out = tp_count / (tp_count + 0.5 * (fp_count + fn_count))
    return f1_score_out


def f1_score_no_na_numpy(y_true, y_pred):
    """
    Calculate the micro F1 score, ignoring all true positive matches on the 'na' tokens

    A slightly smarter way of doing the calculations using numpy. But keras.Metrics doesnt like this.
    :param y_true: ndarray
    :param y_pred: ndarray
    :return:
    """
    non_na_matches = np.logical_and(y_true == y_pred, y_true != 0)
    tp_count = non_na_matches.sum()
    fp_count = np.logical_and(~non_na_matches, y_pred != 0).sum()
    fn_count = np.logical_and(~non_na_matches, y_true != 0).sum()
    f1_score_out = tp_count / (tp_count + 0.5 * (fp_count + fn_count))
    return f1_score_out


class SparseAccuracyNoPad(tf.keras.metrics.Metric):
    """
    Calculate the accuracy over labels, skipping any subtokens with the padding label
    """
    def __init__(
            self,
            name="accuracy",
            pad_label: int = -100,
            exclude_na: bool = False,
            na_label: int = DEFAULT_NA_INDEX,
            **kwargs):
        """
        Calculate accuracy, ignoring any tokens with `pad_label`
        :param name: sets metric name, as per keras.metrics.Metric convention
        :param pad_label: the label value used for [PAD], [CLS], [SEP], or extended subtokens
        :param exclude_na: whether or not to include the 'na' token labels in the accuracy metric
            This only excludes the elements that correctly identified the 'na' token as an 'na' token
        :param na_label: the value of the label associated with the 'na' class
        :param kwargs: passthrough to keras.metrics.Metric
        """
        super().__init__(name, **kwargs)
        self.pad_label = pad_label
        self.exclude_na = exclude_na
        self.NA_LABEL = na_label
        self.n_matches = self.add_weight(name="n_matches", initializer="zeros", dtype="int64")
        self.count = self.add_weight(name="count", initializer="zeros", dtype="int64")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = K.flatten(y_true)
        active_labels = tf.reshape(y_true, (-1,)) != self.pad_label
        y_true = tf.boolean_mask(y_true, active_labels)
        y_pred = K.flatten(K.argmax(y_pred, axis=-1))
        y_pred = K.cast(tf.boolean_mask(y_pred, active_labels), dtype=y_true.dtype)

        if self.exclude_na:
            # true where neither y_true or y_pred have na label
            not_na_matches = tf.math.logical_not(
                K.all(K.stack([K.equal(y_true, y_pred), K.equal(y_true, self.NA_LABEL)], axis=0), axis=0)
            )
            matches = K.sum(
                K.cast(K.all(K.stack([not_na_matches, K.equal(y_pred, y_true)], axis=0), axis=0), "int64")
            )
            count = K.sum(K.cast(not_na_matches, "int64"))
        else:
            matches = K.sum(K.cast(K.equal(y_true, y_pred), "int64"))
            count = K.sum(K.cast(active_labels, "int64"))

        self.count.assign_add(count)
        self.n_matches.assign_add(matches)

    def result(self):
        accuracy = self.n_matches / self.count
        return accuracy

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.n_matches.assign(0)
        self.count.assign(0)


class MicroF1ScoreNoNa(tf.keras.metrics.Metric):
    """
    Calculate categorical true positive counts, with weights
    y_pred is the softmax output, so needs to be argmax'd
    """

    def __init__(self, num_classes, name="f1_score_no_na", pad_label=-100, **kwargs):
        super().__init__(name, **kwargs)
        self.num_classes = num_classes
        self.pad_label = pad_label
        self.tp = self.add_weight(name="tp", initializer="zeros")
        self.fn = self.add_weight(name="fn", initializer="zeros")
        self.fp = self.add_weight(name="fp", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = K.flatten(K.argmax(y_true, axis=-1))
        y_pred = K.flatten(K.argmax(y_pred, axis=-1))

        # mask where samples are not both "na"=(0, 0)
        not_na_matches = tf.math.logical_not(
            K.all(K.stack([K.equal(y_true, y_pred), K.equal(y_true, 0)], axis=0), axis=0)
        )
        true_pos = K.sum(
            K.cast(K.all(K.stack([not_na_matches, K.equal(y_pred, y_true)], axis=0), axis=0), "float32")
        )
        false_neg = K.sum(
            K.cast(K.all(K.stack([not_na_matches, K.equal(y_pred, 0)], axis=0), axis=0), "float32")
        )
        false_pos = K.sum(
            K.cast(K.all(K.stack([not_na_matches, K.equal(y_true, 0)], axis=0), axis=0), "float32")
        )
        self.tp.assign_add(true_pos)
        self.fn.assign_add(false_neg)
        self.fp.assign_add(false_pos)

    def result(self):
        operations_logger.info(f"Getting F1 result: tp={self.tp}, fp={self.fp}, fn={self.fn}")
        f1_score_out = self.tp / (self.tp + 0.5 * (self.fp + self.fn))
        return f1_score_out

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.tp.assign(0.0)
        self.fn.assign(0.0)
        self.fp.assign(0.0)


class SparseMicroF1ScoreNoNa(tf.keras.metrics.Metric):
    """
    Calculate categorical true positive counts, with weights
    y_pred is the softmax output, so needs to be argmax'd in here

    """

    def __init__(self, num_classes, name="f1_score_no_na", pad_label=-100, **kwargs):
        super().__init__(name, **kwargs)
        self.num_classes = num_classes
        self.pad_label = pad_label
        self.NA_LABEL = 0
        self.tp = self.add_weight(name="tp", initializer="zeros")
        self.fn = self.add_weight(name="fn", initializer="zeros")
        self.fp = self.add_weight(name="fp", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = K.flatten(y_true)
        active_labels = tf.reshape(y_true, (-1,)) != self.pad_label
        y_true = tf.boolean_mask(y_true, active_labels)
        y_pred = K.flatten(K.argmax(y_pred, axis=-1))
        y_pred = K.cast(tf.boolean_mask(y_pred, active_labels), dtype=y_true.dtype)

        # mask, true where samples are not both "na"=(0, 0)
        not_na_matches = tf.math.logical_not(
            K.all(K.stack([K.equal(y_true, y_pred), K.equal(y_true, self.NA_LABEL)], axis=0), axis=0)
        )
        true_pos = K.sum(
            K.cast(K.all(K.stack([not_na_matches, K.equal(y_pred, y_true)], axis=0), axis=0), "float32")
        )
        false_neg = K.sum(
            K.cast(K.all(K.stack([not_na_matches, K.equal(y_pred, self.NA_LABEL)], axis=0), axis=0), "float32")
        )
        false_pos = K.sum(
            K.cast(K.all(K.stack([not_na_matches, K.equal(y_true, self.NA_LABEL)], axis=0), axis=0), "float32")
        )
        self.tp.assign_add(true_pos)
        self.fn.assign_add(false_neg)
        self.fp.assign_add(false_pos)

    def result(self):
        f1_score_out = self.tp / (self.tp + 0.5 * (self.fp + self.fn))
        return f1_score_out

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.tp.assign(0.0)
        self.fn.assign(0.0)
        self.fp.assign(0.0)


class CategoricalTruePositives(tf.keras.metrics.Metric):
    """
    Calculate categorical true positive counts, with weights
    y_pred is the softmax output, so needs to be argmax'd
    """

    def __init__(self, num_classes, name="categorical_tp", pad_label=-100, **kwargs):
        super(CategoricalTruePositives, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.pad_label = pad_label
        self.cat_true_positives = self.add_weight(name="ctp", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        # check to see if the last shape is the same as num_classes, to check for one-hot
        # this is a horrible hack, and we shouldnt be doingn it
        y_true = K.flatten(K.argmax(y_true, axis=-1))
        y_pred = K.flatten(K.argmax(y_pred, axis=-1))

        not_na_matches = tf.math.logical_not(
            K.all(K.stack([K.equal(y_true, y_pred), K.equal(y_true, 0)], axis=0), axis=0)
        )
        true_pos = K.sum(K.cast(
            K.all(K.stack([not_na_matches, K.equal(y_pred, y_true)], axis=0), axis=0),
            dtype="float32"))
        self.cat_true_positives.assign_add(true_pos)

    def result(self):
        return tf.cast(self.cat_true_positives, "int32").numpy()

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.cat_true_positives.assign(0.0)


class SparseCategoricalTruePositives(tf.keras.metrics.Metric):
    """
    Calculate categorical true positive counts, with weights
    y_pred is the softmax output, so needs to be argmax'd
    """

    def __init__(self, num_classes, name="categorical_tp", pad_label=-100, **kwargs):
        super(SparseCategoricalTruePositives, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.pad_label = pad_label
        self.cat_true_positives = self.add_weight(name="ctp", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        # not one hot, filter out any labels that match the pad_label
        y_true = K.flatten(y_true)
        active_labels = tf.reshape(y_true, (-1,)) != self.pad_label
        y_true = tf.boolean_mask(y_true, active_labels)
        y_pred = K.flatten(K.argmax(y_pred, axis=-1))
        y_pred = K.cast(tf.boolean_mask(y_pred, active_labels), dtype=y_true.dtype)

        not_na_matches = tf.math.logical_not(
            K.all(K.stack([K.equal(y_true, y_pred), K.equal(y_true, 0)], axis=0), axis=0)
        )
        true_pos = K.sum(K.cast(
            K.all(K.stack([not_na_matches, K.equal(y_pred, y_true)], axis=0), axis=0),
            dtype="float32"))
        self.cat_true_positives.assign_add(true_pos)

    def result(self):
        return tf.cast(self.cat_true_positives, "int32").numpy()

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.cat_true_positives.assign(0.0)
