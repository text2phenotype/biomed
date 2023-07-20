import unittest
import numpy
from biomed.common.voting import vote_majority, vote_with_weight, construct_sample_weight


class TestVoting(unittest.TestCase):
    NUMBER_TOKENS = 4
    WINDOW_SIZE = 7
    NUM_CLASSES = 3

    def test_vote_majority_empty(self):
        # input dimension is # seqs x window size
        empty_mat = numpy.zeros((self.NUMBER_TOKENS, self.WINDOW_SIZE), dtype=numpy.int64)
        voted = vote_majority(empty_mat, self.NUMBER_TOKENS, self.WINDOW_SIZE)
        self.assertEqual(list(voted), [0, 0, 0, 0])

    def test_vote_majority(self):
        input_mat = numpy.array([[0, 2, 1, 2, 0, 0, 0],
                                 [1, 2, 1, 0, 0, 0, 0],
                                 [2, 1, 0, 0, 0, 0, 0],
                                 [1, 0, 0, 0, 0, 0, 0]])
        voted = vote_majority(input_mat, self.NUMBER_TOKENS, self.WINDOW_SIZE)
        self.assertEqual(list(voted), [0, 1, 2, 1])

    def test_vote_with_weight_empty(self):
        # input is [number_of_seqs, time_step, number of classes]
        empty_mat = numpy.zeros((self.NUMBER_TOKENS, self.WINDOW_SIZE, self.NUM_CLASSES))
        voted = vote_with_weight(empty_mat, num_tokens=self.NUMBER_TOKENS, window_size=self.WINDOW_SIZE)
        for row in voted:
            for i in row:
                self.assertTrue(numpy.isnan(i))

    def test_vote_with_weight(self):
        input_lists = [
            [[0, .3, .7], [0, .8, .2], [.35, .3, .35], [0, .25, .75], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]],
            [[0, .9, .1], [.3, .5, .2], [0, .53, .47], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]],
            [[.5, .2, .3], [0, .75, .25], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]],
            [[0, .5, .5], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]]]
        voted = vote_with_weight(numpy.array(input_lists), self.NUMBER_TOKENS, self.WINDOW_SIZE)
        expected_rows = numpy.array([[0, .3, .7], [0, .85, .15], [.3833, .3333, .2833], [0, .5075, .4925]])
        diff_mat = voted - expected_rows
        for row in diff_mat:
            for i in row:
                self.assertLess(abs(i), .001)

    def test_sample_weight(self):
        class_weight = {'1': 3}
        y_np = numpy.array([[[1., 0., 0., 0., 0.],
                             [0, 1, 0., 0., 0.],
                             [1., 0., 0., 0., 0.],
                             [1., 0., 0., 0., 0.],
                             [1., 0., 0., 0., 0.]],
                            [[0, 1., 0., 0., 0.],
                             [1., 0., 0., 0., 0.],
                             [1., 0., 0., 0., 0.],
                             [1., 0., 0., 0., 0.],
                             [1., 0., 0., 0., 0.]],
                            [[1., 0., 0., 0., 0.],
                             [1., 0., 0., 0., 0.],
                             [1., 0., 0., 0., 0.],
                             [1., 0., 0., 0., 0.],
                             [1., 0., 0., 0., 0.]]])
        sample_weight = construct_sample_weight(class_weight, y_np)
        self.assertEqual(sample_weight[0, 1], 3)
        self.assertEqual(sample_weight[0, 0], 1)
        self.assertEqual(sample_weight[1, 0], 3)
