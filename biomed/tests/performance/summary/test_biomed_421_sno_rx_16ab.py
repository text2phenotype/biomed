import unittest

from text2phenotype.common import common
from text2phenotype.common.feature_data_parsing import from_bag_of_words

from feature_service.nlp import autocode
from feature_service.nlp.nlp_reader import ClinicalReader

from biomed.tests.samples import MTSAMPLES_DIR


class TestBiomed421(unittest.TestCase):

    def test_frequency(self):
        # October 5th
        #
        # sno_rx_16ab: less than 1% (1326)/(313568+32864+1326)
        # SNOMEDCT: 90%
        # RXNORM: 9%

        list_concept_vocab = list()

        for f in common.get_file_list(MTSAMPLES_DIR, '.txt'):
            text = common.read_text(f)
            res = autocode.autocode(text, autocode.PipelineURL.original)

            reader = ClinicalReader(res)
            list_concept_vocab += reader.list_concept_vocab()

            vocab = from_bag_of_words(list_concept_vocab)

            sno_rx_16ab = int(vocab.get('sno_rx_16ab', 0))
            SNOMEDCT_US = int(vocab.get('SNOMEDCT_US', 0))
            RXNORM = int(vocab.get('RXNORM', 0))

            self.assertTrue((SNOMEDCT_US > sno_rx_16ab) and (RXNORM > sno_rx_16ab),
                            f"unacceptable frequent usage of sno_rx_16ab : {vocab}")

            self.assertTrue(SNOMEDCT_US + RXNORM > (100 * sno_rx_16ab),
                            f"sno_rx_16ab terms were detected at greater than 1% frequency, actual counts : {vocab}")
