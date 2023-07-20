import unittest
import numpy as np

from biomed.common.matrix_construction import (
    indexes_for_token
)


class TestIndexesForToken(unittest.TestCase):

    def test_indexes_for_token_matrix(self):
        window_size = 3
        tokens = "this is a sentence".split()
        # number of sequences is 1 for initial window + tokens left from initial window
        # only works for stride length =1
        max_x_size = 1 + (len(tokens) - window_size)
        seqs = np.array((tokens[:window_size],
                        tokens[1:window_size+1]))

        for i, token in enumerate(tokens):
            x_dim, y_dim = indexes_for_token(i, window_size, max_x_size)
            print([tuple(x_dim), tuple(y_dim)])
            print(token, seqs[tuple(x_dim), tuple(y_dim)])
            # all indices should be the same token
            assert all(x==token for x in seqs[tuple(x_dim), tuple(y_dim)])

    def test_indexes_for_token(self):
        # below taken from PR 943
        token_idx = 3
        window_size = 10
        max_x_size = 4
        expected_indexes = ([3, 2, 1, 0], [0, 1, 2, 3])
        indexes = indexes_for_token(token_idx, window_size, max_x_size)
        self.assertEqual(expected_indexes, indexes)

        token_idx = 12
        window_size = 10
        max_x_size = 4
        expected_indexes = ([3], [9])
        indexes = indexes_for_token(token_idx, window_size, max_x_size)
        self.assertEqual(expected_indexes, indexes)

        token_idx = 10
        window_size = 10
        max_x_size = 21
        given_indexes = (list(range(10, 0, -1)), list(range(0, 10)))
        indexes = indexes_for_token(token_idx, window_size, max_x_size)
        self.assertEqual(given_indexes, indexes)

        token_idx = 28
        window_size = 10
        max_x_size = 21
        given_indexes = ([20, 19], [8, 9])
        indexes = indexes_for_token(token_idx, window_size, max_x_size)
        self.assertEqual(given_indexes, indexes)


if __name__ == '__main__':
    unittest.main()
