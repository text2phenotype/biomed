import unittest

from text2phenotype.common.featureset_annotations import Vectorization

from text2phenotype.constants.features import FeatureType

from biomed.common.matrix_construction import get_3d_matrix


class TestBiomed1269(unittest.TestCase):
    TEST_INPUT = Vectorization(json_input_dict={FeatureType.address: {0:[0]*66, 1: [0]*66, 2:[0]*66, 3:[0]*66},
                                                'defaults': {'address': [0]*66}})

    def test_3d_mat_ws_less_text(self):
        # ensure that the 3d matrix created when window size < num_tokens exists and has correct dimensions
        mat_3d, feature_col_mapping = get_3d_matrix(self.TEST_INPUT, 4, 2, [FeatureType.address])
        self.assertListEqual(list(mat_3d.shape), [4, 2, 66])

    def test_3d_mat_ws_bigger_text(self):
        # ensure that the 3d matrix created when window size > num_tokens exists and has correct dimensions
        mat_3d, feature_col_mapping = get_3d_matrix(self.TEST_INPUT, 4, 10, [FeatureType.address])
        self.assertListEqual(list(mat_3d.shape), [4, 10, 66])

    def test_no_tokens(self):
        # ensure that the 3d matrix created when token length = 0 is None
        test_input = Vectorization(json_input_dict={FeatureType.address: {}, 'defaults': {'address': [0]*66}})
        mat_3d, feature_col_mapping = get_3d_matrix(test_input, 0, 10, [FeatureType.address])
        self.assertEqual(mat_3d.shape, (0, 10, 66))

