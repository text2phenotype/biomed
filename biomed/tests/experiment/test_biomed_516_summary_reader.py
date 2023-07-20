import os
import unittest

from text2phenotype.common import common

from feature_service.nlp.nlp_reader import SummaryReader

from biomed.tests.samples import MTSAMPLES_DIR

UNCERTAIN_SUMMARY_JSON = os.path.join(MTSAMPLES_DIR, 'new_summary_mtsamples', 'orthopedic-consult-5.txt-clean.txt')


class TestBiomed516(unittest.TestCase):

    def test_summary_reader(self):
        res = common.read_json(UNCERTAIN_SUMMARY_JSON)

        text2summary = res['text2summary']

        reader = SummaryReader(text2summary)
