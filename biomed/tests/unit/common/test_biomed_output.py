#!/usr/bin/env python
import unittest

from biomed.common.biomed_ouput import GeneticsOutput

from text2phenotype.constants.features.label_types import GeneticsLabel


class TestGeneticsOutput(unittest.TestCase):
    def test_ctor_gene_positive(self):
        gene_name = 'BRCA'
        gene_result = '+'
        token = f'{gene_name}{gene_result}'
        output = GeneticsOutput(label=GeneticsLabel.gene.value.persistent_label,
                                text=token,
                                range=[100, 105])
        self.assertEqual(token, output.text)
        self.assertEqual(gene_name, output.preferredText)
        self.assertEqual(gene_result, output.polarity)
        self.assertListEqual([100, 105], output.range)

    def test_ctor_gene_negative(self):
        gene_name = 'BRCA'
        gene_result = '-'
        token = f'{gene_name}{gene_result}'
        output = GeneticsOutput(label=GeneticsLabel.gene.value.persistent_label,
                                text=token,
                                range=[100, 105])
        self.assertEqual(token, output.text)
        self.assertEqual(gene_name, output.preferredText)
        self.assertEqual(gene_result, output.polarity)
        self.assertListEqual([100, 105], output.range)

    def test_ctor_gene_no_interp(self):
        gene_name = 'PDL-1'
        output = GeneticsOutput(label=GeneticsLabel.gene.value.persistent_label,
                                text=gene_name,
                                range=[100, 105])
        self.assertEqual(gene_name, output.text)
        self.assertEqual(gene_name, output.preferredText)
        self.assertIsNone(output.polarity)
        self.assertListEqual([100, 105], output.range)


if __name__ == '__main__':
    unittest.main()
