import unittest
import numpy as np
import tensorflow as tf

from biomed.models import metrics


class TestMicroF1NoNaFunction(unittest.TestCase):
    def test_f1_score_skip_na(self):
        y_true = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        y_pred = np.array([0, 0, 1, 1, 1, 0, 0, 1, 2])
        self.assertEqual(3 / 5.5, metrics.f1_score_no_na(y_true, y_pred))

    def test_f1_score_skip_na_numpy(self):
        y_true = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        y_pred = np.array([0, 0, 1, 1, 1, 0, 0, 1, 2])
        self.assertEqual(3 / 5.5, metrics.f1_score_no_na_numpy(y_true, y_pred))


class TestFlatFunctions(unittest.TestCase):
    def test_recall_binary(self):
        y_true = tf.constant([0, 0, 0, 1, 1, 1, 2, 2, 2])
        y_pred = tf.constant([0, 0, 1, 1, 1, 0, 0, 1, 2])
        self.assertEqual(4 / 6, metrics.recall_binary(y_true, y_pred))

    def test_precision_binary(self):
        y_true = tf.constant([0, 0, 0, 1, 1, 1, 2, 2, 2])
        y_pred = tf.constant([0, 0, 1, 1, 1, 0, 0, 1, 2])
        self.assertEqual(4 / 5, metrics.precision_binary(y_true, y_pred))

    def test_f1_score_binary(self):
        y_true = tf.constant([0, 0, 0, 1, 1, 1, 2, 2, 2])
        y_pred = tf.constant([0, 0, 1, 1, 1, 0, 0, 1, 2])
        tp = 4
        fp = 1
        fn = 2
        self.assertAlmostEqual(tp / (tp + 0.5*(fp+fn)), metrics.f1_score_binary(y_true, y_pred), 6)


class TestCategoricalTruePositives(unittest.TestCase):
    def test_cat_tp(self):
        num_classes = 3
        y_true = tf.constant([0, 0, 0, 1, 1, 1, 2, 2, 2])
        y_true = tf.keras.utils.to_categorical(y_true, num_classes=num_classes)
        y_pred = tf.constant([0, 0, 1, 1, 1, 0, 0, 1, 2])
        y_pred = tf.keras.utils.to_categorical(y_pred, num_classes=num_classes) * 0.9

        m = metrics.CategoricalTruePositives(num_classes=num_classes)
        m.update_state(y_true, y_pred)
        self.assertEqual(3, m.result())


class TestSparseCategoricalTruePositives(unittest.TestCase):

    def test_cat_tp_sparse_labels(self):
        num_classes = 3
        y_true = tf.constant([-100, 0, 0, 0, 1, 1, 1, 2, 2, 2, -100, -100])
        y_pred = tf.constant([0, 0, 0, 1, 1, 1, 0, 0, 1, 2, 0, 0])
        y_pred = tf.keras.utils.to_categorical(y_pred, num_classes=num_classes) * 0.9

        m = metrics.SparseCategoricalTruePositives(num_classes=num_classes)
        m.update_state(y_true, y_pred)
        self.assertEqual(3, m.result())


class TestSparseAccuracyNoPad(unittest.TestCase):
    def test_sparse_accuracy_no_pad(self):
        num_classes = 3
        y_true = tf.constant([[-100, 0, 0, 1, 1, 1, 2, 2, 2, -100, -100]])
        y_pred = tf.constant([[   0, 0, 1, 1, 1, 0, 0, 1, 2, 0, 0]])
        y_pred = tf.keras.utils.to_categorical(y_pred, num_classes=num_classes+1) * 0.9
        expected_accuracy = 4 / 8

        m = metrics.SparseAccuracyNoPad()
        m.update_state(y_true, y_pred)
        self.assertAlmostEqual(expected_accuracy, m.result().numpy(), 6)

    def test_sparse_accuracy_no_pad_na_na(self):
        num_classes = 3
        y_true = tf.constant([[-100, 0, 0, 1, 1, 1, 2, 2, 2, -100, -100]])
        y_pred = tf.constant([[   0, 0, 1, 1, 1, 0, 0, 1, 2, 0, 0]])
        y_pred = tf.keras.utils.to_categorical(y_pred, num_classes=num_classes+1) * 0.9
        expected_accuracy = 3 / 7

        m = metrics.SparseAccuracyNoPad(exclude_na=True)
        m.update_state(y_true, y_pred)
        self.assertAlmostEqual(expected_accuracy, m.result().numpy(), 6)


class TestMicroF1ScoreNoNa(unittest.TestCase):
    @staticmethod
    def _f1_score(tp, fp, fn):
        return tp / (tp + 0.5 * (fp + fn))

    def test_micro_f1_no_na(self):
        num_classes = 3
        y_true = tf.constant([0, 0, 0, 1, 1, 1, 2, 2, 2])
        y_true = tf.keras.utils.to_categorical(y_true, num_classes=num_classes)
        y_pred = tf.constant([0, 0, 1, 1, 1, 0, 0, 1, 2])
        y_pred = tf.keras.utils.to_categorical(y_pred, num_classes=num_classes) * 0.9

        m = metrics.MicroF1ScoreNoNa(num_classes=num_classes)
        m.update_state(y_true, y_pred)
        tp = 3
        fn = 2
        fp = 1
        self.assertAlmostEqual(self._f1_score(tp, fp, fn), m.result(), 6)

    def test_micro_f1_no_na_3d(self):
        num_classes = 3
        y_true = tf.constant([[0, 0, 0, 1, 1, 1, 2, 2, 2]])
        y_true = tf.keras.utils.to_categorical(y_true, num_classes=num_classes)
        y_pred = tf.constant([[0, 0, 1, 1, 1, 0, 0, 1, 2]])
        y_pred = tf.keras.utils.to_categorical(y_pred, num_classes=num_classes) * 0.9

        m = metrics.MicroF1ScoreNoNa(num_classes=num_classes)
        m.update_state(y_true, y_pred)
        tp = 3
        fn = 2
        fp = 1
        self.assertAlmostEqual(self._f1_score(tp, fp, fn), m.result(), 6)

    def test_micro_f1_no_na_sparse_labels(self):
        num_classes = 3
        y_true = tf.constant([[-100, 0, 0, 1, 1, 1, 2, 2, 2, -100, -100]])
        y_pred = tf.constant([[   0, 0, 1, 1, 1, 0, 0, 1, 2, 0, 0]])
        y_pred = tf.keras.utils.to_categorical(y_pred, num_classes=num_classes+1) * 0.9
        # y_pred = tf.constant([[-100, 0, 1, 1, 1, 0, 0, 1, 2, 0, 0]])

        m = metrics.SparseMicroF1ScoreNoNa(num_classes=num_classes)
        m.update_state(y_true, y_pred)
        tp = 3
        fn = 2
        fp = 1
        self.assertAlmostEqual(self._f1_score(tp, fp, fn), m.result().numpy(), 6)


if __name__ == '__main__':
    unittest.main()
