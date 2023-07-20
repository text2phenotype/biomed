#!/usr/bin/env python
import unittest

from biomed.genetics.genetics import split_genes


class TestGenetics(unittest.TestCase):
    def test_split_genes_single_gene(self):
        gene_token = 'BRCA'
        self.assertListEqual([(gene_token, [100, 104])], split_genes(gene_token, [100, 104]))

    def test_split_genes_multiple_genes(self):
        self.assertListEqual([('HR+', [100, 103]), ('HER2-', [104, 109]), ('PR-', [110, 113])],
                             split_genes('HR+/HER2-/PR-', [100, 113]))


if __name__ == '__main__':
    unittest.main()
