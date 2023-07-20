import unittest

from feature_service.common import bsv, data_parsing
from biomed.tests import samples

from text2phenotype.common.log import operations_logger


class TestBiomed474(unittest.TestCase):

    def test_expert_loinc_modifiers(self):
        loinc = bsv.parse_bsv_list(samples.SHRINE_LOINC_BSV, 'code|text')
        text_list = [i.get('text') for i in loinc]

        bag = list()

        for text in text_list:
            bag += text.split()

        loinc_term_frequency = data_parsing.sort_tf(loinc)

        operations_logger.info(loinc_term_frequency)
