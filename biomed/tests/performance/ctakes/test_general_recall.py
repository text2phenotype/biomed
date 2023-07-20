#!/usr/bin/env python
import os
import unittest

from text2phenotype.common import common
from text2phenotype.common.log import operations_logger

from feature_service.nlp import autocode

from biomed.tests.performance.ctakes.base import CtakesAnnotator


class DiagnosesAnnotator(CtakesAnnotator):
    """Diagnoses term annotator."""
    def _get_autocode_entities(self, text: str):
        return autocode.lab_value(text)['labValues']
        # general vs bsv_master, bsv_minus
        # shrine-icd9
        # shrine-icd10
        # medgen
        # return nlp.lab_value(text)['labValues']


class TestRecall(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__annotator = DiagnosesAnnotator()

    # @unittest.skip
    def test_MIMIC(self):
        self.__test_recall('MIMIC_diagnoses.txt', 'MIMIC_unknown_diagnoses.txt')

    def __test_recall(self, term_file, out_file):
        found, missed, fmatches = self.__annotator.recall_expected(os.path.join(os.path.dirname(__file__), term_file))

        total = len(found) + len(missed)
        operations_logger.info(
            f"Did not get a result for {len(missed)} of {total} ({100 * len(missed) / total}%) diagnoses.")
        operations_logger.info(f'Full term matches: {fmatches}')

        common.write_text('\n'.join(sorted(missed)), out_file)
        operations_logger.info(f"Unrecognized medications written to {out_file}.\n\n")


if __name__ == '__main__':
    unittest.main()
