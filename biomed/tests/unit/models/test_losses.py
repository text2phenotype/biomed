import unittest

import numpy as np
import tensorflow as tf
from transformers.modeling_tf_utils import TFTokenClassificationLoss

from biomed.models.losses import BertTokenClassificationLoss


class TestBertTokenClassificationLoss(unittest.TestCase):
    def test_compute_loss_one_window(self):
        tf_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction=tf.keras.losses.Reduction.NONE
        )
        logits = np.array([[
            [2.3069983, -0.85339093, -0.80355066],
            [2.306845, -0.85335857, -0.8035123],
            [2.3053787, -0.85253316, -0.8027111],
            [2.307844, -0.8538495, -0.8040178],
            [2.3055103, -0.8526848, -0.8028527],
            [2.3044097, -0.8521269, -0.802356],
            [2.3080986, -0.85395974, -0.8040392],
            [2.3020146, -0.85095656, -0.8012646],
            [2.3065584, -0.853231, -0.80340344],
            [2.306578, -0.8531774, -0.8033886]]]
        )
        labels = np.array([[0, 2, -100, -100, 0, 0, 0, 0, 0, 0]])
        mask = np.array([[True, True, False, False, True, True, True, True, True, True, ]])

        cropped_labels = labels[mask]
        cropped_logits = logits[mask, :]

        # We expect a 1d (window, token) array of losses if reduction=NONE
        basic_expected_loss = np.array(
            [0.08340846, 3.1937809, 0.08358388, 0.08371446, 0.08327828,
             0.08399799, 0.08345595, 0.08345708])
        tf_expected_loss = tf_loss_fn(cropped_labels, cropped_logits)

        # Test that the keras sparse CE gives us an expected result
        # We will use this basic_expected_loss for comparison to the huggingface loss
        # and our custom loss
        np.testing.assert_array_almost_equal(basic_expected_loss, tf_expected_loss, 6)

        # test on the Huggingface method, which flattens the batch
        huggingface_loss = TFTokenClassificationLoss().compute_loss(
            tf.convert_to_tensor(labels),
            tf.convert_to_tensor(logits))
        np.testing.assert_array_almost_equal(basic_expected_loss.flatten(), huggingface_loss, 6)

        # test on our method, which also flattens the batch
        loss = BertTokenClassificationLoss().compute_loss(tf.convert_to_tensor(labels), tf.convert_to_tensor(logits))
        np.testing.assert_array_almost_equal(basic_expected_loss.flatten(), loss, 6)

    def test_compute_loss_multi_window(self):
        tf_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction=tf.keras.losses.Reduction.NONE
        )
        logits = np.array([
            [[2.3069983, -0.85339093, -0.80355066],
             [2.3069983, -0.85339093, -0.80355066],
             [2.306845, -0.85335857, -0.8035123],
             [2.3053787, -0.85253316, -0.8027111],
             [2.307844, -0.8538495, -0.8040178],
             [2.3055103, -0.8526848, -0.8028527],
             [2.3044097, -0.8521269, -0.802356],
             [2.3080986, -0.85395974, -0.8040392],
             [2.3020146, -0.85095656, -0.8012646],
             [2.3065584, -0.853231, -0.80340344],
             [2.306578, -0.8531774, -0.8033886],
             [2.3069983, -0.85339093, -0.80355066],
             ],
            [[2.3069983, -0.85339093, -0.80355066],
             [2.3069983, -0.85339093, -0.80355066],
             [2.306845, -0.85335857, -0.8035123],
             [2.3053787, -0.85253316, -0.8027111],
             [2.307844, -0.8538495, -0.8040178],
             [2.3055103, -0.8526848, -0.8028527],
             [2.3044097, -0.8521269, -0.802356],
             [2.3080986, -0.85395974, -0.8040392],
             [2.3020146, -0.85095656, -0.8012646],
             [2.3065584, -0.853231, -0.80340344],
             [2.306578, -0.8531774, -0.8033886],
             [2.3069983, -0.85339093, -0.80355066],
             ],
        ])
        labels = np.array([
            [-100, 0, 2, -100, -100, 0, 0, 0, 0, 0, 0, -100],
            [-100, 0, 2, -100, -100, 0, 0, 0, 0, 0, -100, -100]
        ])
        mask = np.array([
            [False, True, True, False, False, True, True, True, True, True, True, False],
            [False, True, True, False, False, True, True, True, True, True, False, False],
        ])

        # get the flattened masked labels and logits
        cropped_labels = labels[mask]
        cropped_logits = logits[mask, :]
        n_expected_values = mask.sum()
        # We expect a 1d (window, token) array of losses if reduction=NONE
        # So we flatten the labels and logits by the mask for the basic TF sparse loss
        basic_expected_loss = np.array(
            [0.083408, 3.193781, 0.083584, 0.083714, 0.083278, 0.083998,
                0.083456, 0.083457, 0.083408, 3.193781, 0.083584, 0.083714,
                0.083278, 0.083998, 0.083456])
        tf_expected_loss = tf_loss_fn(cropped_labels, cropped_logits)
        np.testing.assert_array_almost_equal(basic_expected_loss, tf_expected_loss, 6)

        # test on the Huggingface method, which flattens the batch
        huggingface_loss = TFTokenClassificationLoss().compute_loss(
            tf.convert_to_tensor(labels),
            tf.convert_to_tensor(logits))
        np.testing.assert_array_almost_equal(basic_expected_loss, huggingface_loss, 6)

        # test on our method, which also flattens the batch
        loss = BertTokenClassificationLoss().compute_loss(tf.convert_to_tensor(labels), tf.convert_to_tensor(logits))
        np.testing.assert_array_almost_equal(basic_expected_loss, loss, 6)

    def test_compute_loss_sample_weights(self):
        # batch_size is 1; we need to have the outer array for the loss fn to work
        logits = np.array([[
            [2.3069983, -0.85339093, -0.80355066],
            [2.306845, -0.85335857, -0.8035123],
            [2.3053787, -0.85253316, -0.8027111],
            [2.307844, -0.8538495, -0.8040178],
            [2.3055103, -0.8526848, -0.8028527],
            [2.3044097, -0.8521269, -0.802356],
            [2.3080986, -0.85395974, -0.8040392],
            [2.3020146, -0.85095656, -0.8012646],
            [2.3065584, -0.853231, -0.80340344],
            [2.306578, -0.8531774, -0.8033886]]]
        )
        labels = np.array([[0, 2, -100, -100, 0, 0, 0, 0, 0, 0]])
        expected_sample_weights = np.array([[0, 2, 1, 1, 1, 1, 1, 1, 1, 1]])

        # get the expected loss from the huggingface loss method
        # The result is squeezed, so we lose the batch_size dim
        unscaled_expected_loss = TFTokenClassificationLoss().compute_loss(
            tf.convert_to_tensor(labels),
            tf.convert_to_tensor(logits))

        expected_loss = unscaled_expected_loss * expected_sample_weights[0, labels[0, :] != -100]

        loss = BertTokenClassificationLoss.compute_loss(
            tf.convert_to_tensor(labels),
            tf.convert_to_tensor(logits),
            sample_weight=tf.convert_to_tensor(expected_sample_weights)
        )
        np.testing.assert_array_almost_equal(expected_loss, loss, 6)

    def test_compute_loss_sample_weights_batch_3(self):
        # batch_size is 1; we need to have the outer array for the loss fn to work
        batch_size = 3
        logits = np.array([[
            [2.3069983, -0.85339093, -0.80355066],
            [2.306845, -0.85335857, -0.8035123],
            [2.3053787, -0.85253316, -0.8027111],
            [2.307844, -0.8538495, -0.8040178],
            [2.3055103, -0.8526848, -0.8028527],
            [2.3044097, -0.8521269, -0.802356],
            [2.3080986, -0.85395974, -0.8040392],
            [2.3020146, -0.85095656, -0.8012646],
            [2.3065584, -0.853231, -0.80340344],
            [2.306578, -0.8531774, -0.8033886]]]
        )
        batch_logits = np.repeat(logits, batch_size, axis=0)
        labels = np.array([[0, 2, -100, -100, 0, 0, 0, 0, 0, 0]])
        batch_labels = np.repeat(labels, batch_size, axis=0)
        expected_sample_weights = np.array([[0, 2, 1, 1, 1, 1, 1, 1, 1, 1]])
        batched_expected_sample_weights = np.repeat(expected_sample_weights, batch_size, axis=0)

        # get the expected loss from the huggingface loss method
        # The result is squeezed, so we lose the batch_size dim
        unscaled_expected_loss = TFTokenClassificationLoss().compute_loss(
            tf.convert_to_tensor(batch_labels),
            tf.convert_to_tensor(batch_logits))

        valid_labels = batch_labels.reshape((-1)) != -100
        expected_loss = unscaled_expected_loss * batched_expected_sample_weights.reshape((-1,))[valid_labels]

        loss = BertTokenClassificationLoss.compute_loss(
            tf.convert_to_tensor(batch_labels),
            tf.convert_to_tensor(batch_logits),
            sample_weight=tf.convert_to_tensor(batched_expected_sample_weights)
        )
        np.testing.assert_array_almost_equal(expected_loss, loss, 6)


if __name__ == '__main__':
    unittest.main()
