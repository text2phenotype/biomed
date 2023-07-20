from typing import List

import tensorflow as tf

from biomed.models.bert_utils import BERT_PADDING_LABEL
from text2phenotype.common.log import operations_logger

def shape_list(x: tf.Tensor) -> List[int]:
    """
    Deal with dynamic shape in tensorflow cleanly.

    Args:
        x (:obj:`tf.Tensor`): The tensor we want the shape of.

    Returns:
        :obj:`List[int]`: The shape of the tensor as a list.
    """
    static = x.shape.as_list()
    dynamic = tf.shape(x)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]


class BertTokenClassificationLoss:
    """
    Loss function suitable for token classification.

    .. note::
        Any label of -100 will be ignored (along with the corresponding logits) in the loss computation.

    """
    @staticmethod
    def compute_loss(labels: tf.Tensor, logits: tf.Tensor, sample_weight: tf.Tensor = None):
        """

        :param labels: y_true, expects shape (batch_size, window_size)
        :param logits: returned logits from model output, expects shape (batch_size, window_size, n_classes)
        :param sample_weight: Optional, expects shape (batch_size, window_size)
        :return: Loss values with shape (batch_size, window_size)
        """
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction=tf.keras.losses.Reduction.NONE
        )
        # make sure only labels that are not equal to -100 are taken into account as loss
        active_loss = tf.reshape(labels, (-1,)) != BERT_PADDING_LABEL
        # logits need to keep the shape of the last dimension, which isnt in the labels or sample_weight
        reduced_logits = tf.boolean_mask(tf.reshape(logits, (-1, shape_list(logits)[2])), active_loss)
        reduced_labels = tf.boolean_mask(tf.reshape(labels, (-1,)), active_loss)
        if sample_weight is not None:
            sample_weight = tf.boolean_mask(tf.reshape(sample_weight, (-1,)), active_loss)
        out = loss_fn(reduced_labels, reduced_logits, sample_weight=sample_weight)
        # if len(out.shape) == 1:
        #     operations_logger.info(f"Too much squeeze: {out.shape}")
        #     out = tf.expand_dims(out, axis=0)
        #     # out = tf.reshape(out, (shape_list(labels)[0],))

        return out