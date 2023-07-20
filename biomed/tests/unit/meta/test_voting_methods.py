import unittest

from biomed.meta.voting_methods import *


class TestDeserializeVotingMethodEnum(unittest.TestCase):
    def test_deserialize_enum(self):
        self.assertEqual(VotingMethodEnum['weighted_entropy'], VotingMethodEnum.weighted_entropy)
        self.assertEqual(VotingMethodEnum['threshold'], VotingMethodEnum.threshold)
        self.assertEqual(VotingMethodEnum['threshold_categories'], VotingMethodEnum.threshold_categories)
        self.assertEqual(VotingMethodEnum["model_avg"], VotingMethodEnum.model_avg)
        self.assertEqual(VotingMethodEnum["model_weighted_avg"], VotingMethodEnum.model_weighted_avg)
        self.assertEqual(VotingMethodEnum["rf_classifier"], VotingMethodEnum.rf_classifier)


class TestVotingFunctions(unittest.TestCase):
    TOKEN_COUNT = 4
    PREDICTION_1 = np.array(
        [
            [0, .8, .2],
            [.9999, 0.0001, 0],
            [.3, .3, .4],
            [0.2, 0.7, 0.1]
        ]
    )
    PREDICTION_2 = np.array(
        [
            [.3, .1, .6],
            [.5, 0.25, 0.25],
            [.375, .375, .25],
            [0.2, 0.6, 0.2]
        ]
    )
    PREDICTION_3 = np.array(
        [
            [.9, .1, 0],
            [.9, 0.1, 0],
            [.9, .1, 0],
            [.9, 0.1, 0]
        ]
    )
    DICT_INPUT = {
        'model_1': PredictResults(predicted_probs=PREDICTION_1),
        'model_2': PredictResults(predicted_probs=PREDICTION_2),
        'model_3': PredictResults(predicted_probs=PREDICTION_3)
    }

    def test_vote_with_weight(self):
        output = vote(
            prediction_dict=self.DICT_INPUT, num_tokens=4, num_classes=3,
            voting_method=VotingMethodEnum.weighted_entropy)
        self.assertIsInstance(output, PredictResults)
        expected_output = np.array(
            [
                [5.01378221e-01, 3.26055735e-01, 1.72566044e-01],
                [9.99098622e-01, 6.56879966e-04, 2.44497562e-04],
                [6.89340122e-01, 1.89006974e-01, 1.21652904e-01],
                [6.00565994e-01, 3.37081763e-01, 6.23522432e-02]
            ]
        )
        expected_categories = np.array([0, 0, 0, 0])

        self.assertTrue(np.allclose(output.predicted_category, expected_categories, atol=1e-7))
        self.assertTrue(np.allclose(output.predicted_probs, expected_output, atol=1e-7))

    def test_vote_threshold(self):
        output = vote(
            prediction_dict=self.DICT_INPUT, num_tokens=4, num_classes=3, voting_method=VotingMethodEnum.threshold)
        self.assertIsInstance(output, PredictResults)
        expected_output = np.array(
            [
                [0.50137822, 0.32605574, 0.17256604],
                [0.999098622, 6.56879966e-04, 2.44497562e-04],
                [0.68934012, 0.18900697, 0.1216529],
                [0.60056599, 0.33708176, 0.06235224]])
        expected_categories = np.array([1, 0, 1, 1])
        self.assertTrue(np.allclose(output.predicted_category,  expected_categories, atol=1e-7))
        self.assertTrue(np.allclose(output.predicted_probs, expected_output, atol=1e-7))

    def test_vote_threshold_category(self):
        output = vote(
            prediction_dict=self.DICT_INPUT, num_tokens=4, num_classes=3, threshold_categories=[1],
            voting_method=VotingMethodEnum.threshold_categories)
        self.assertIsInstance(output, PredictResults)
        expected_output = np.array(
            [[5.01378221e-01, 3.26055735e-01, 1.72566044e-01],
             [9.99098622e-01, 6.56879966e-04, 2.44497562e-04],
             [6.89340122e-01, 1.89006974e-01, 1.21652904e-01],
             [6.00565994e-01, 3.37081763e-01, 6.23522432e-02]])
        expected_categories = np.array([1, 0, 0, 1])
        self.assertTrue(np.allclose(output.predicted_category, expected_categories, atol=1e-7))
        self.assertTrue(np.allclose(output.predicted_probs, expected_output, atol=1e-7))

    def test_model_avg_voting(self):
        output = vote(
            prediction_dict=self.DICT_INPUT, num_tokens=4, num_classes=3,
            voting_method=VotingMethodEnum.model_avg)
        self.assertIsInstance(output, PredictResults)
        expected_probs = np.array(
            [[0.4, 0.33333333, 0.26666667],
             [0.79996667, 0.1167, 0.08333333],
             [0.525, 0.25833333, 0.21666667],
             [0.43333333, 0.46666667, 0.1]])
        expected_categories = np.array([0, 0, 0, 1])
        self.assertTrue(np.allclose(output.predicted_category, expected_categories, atol=1e-7))
        self.assertTrue(np.allclose(output.predicted_probs, expected_probs, atol=1e-7))

    def test_model_weighted_avg_voting_default_weights(self):
        output = vote(
            prediction_dict=self.DICT_INPUT, num_tokens=4, num_classes=3,
            voting_method=VotingMethodEnum.model_weighted_avg)
        self.assertIsInstance(output, PredictResults)
        expected_probs = np.array(
            [[0.4, 0.33333333, 0.26666667],
             [0.79996667, 0.1167, 0.08333333],
             [0.525, 0.25833333, 0.21666667],
             [0.43333333, 0.46666667, 0.1]])
        expected_categories = np.array([0, 0, 0, 1])
        self.assertTrue(np.allclose(output.predicted_category, expected_categories, atol=1e-7))
        self.assertTrue(np.allclose(output.predicted_probs, expected_probs, atol=1e-7))

    def test_model_weighted_avg_voting_selected_weights(self):
        weights = [0.2, 0.5, 0.3]  # should be normalized so sum to 1
        output = vote(
            prediction_dict=self.DICT_INPUT, num_tokens=4, num_classes=3,
            voting_method=VotingMethodEnum.model_weighted_avg,
            weights=weights
        )
        self.assertIsInstance(output, PredictResults)
        expected_probs = np.array([
            [0.42, 0.24, 0.34],
            [0.71998, 0.15502, 0.125],
            [0.5175 , 0.2775 , 0.205],
            [0.41, 0.47, 0.12]])
        expected_categories = np.array([0, 0, 0, 1])
        self.assertTrue(np.allclose(output.predicted_category, expected_categories, atol=1e-7))
        self.assertTrue(np.allclose(output.predicted_probs, expected_probs, atol=1e-7))

    def test_model_rf_classifier_voting(self):
        # always requires a model folder name
        model_cache = ModelCache()
        output = vote(
            prediction_dict=self.DICT_INPUT, num_tokens=4, num_classes=3,
            voting_method=VotingMethodEnum.rf_classifier,
            model_cache=model_cache,
            model_type=ModelType.diagnosis,
            voting_model_folder="voter_rf_diagnosis_20210611"
        )
        self.assertIsInstance(output, PredictResults)
        self.assertEqual((4, 3), output.predicted_probs.shape)
        self.assertEqual((4,), output.predicted_category.shape)


if __name__ == '__main__':
    unittest.main()
