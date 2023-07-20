#!/usr/bin/env python
import os
import unittest

from feature_service.nlp import autocode

from biomed.tests.performance.ctakes.base import CtakesAnnotator


class MedicationAnnotator(CtakesAnnotator):
    """Medication term annotator."""
    def _get_autocode_entities(self, text: str):
        # return nlp.drug_ner(text)['drugEntities']
        return autocode.drug_ner(text, 'rxnorm-syn')['drugEntities']


class TestMedicationRecall(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__annotator = MedicationAnnotator()

    def test_MIMIC_recall_expected_drugs(self):
        self.__test_recall('MIMIC_med_terms.csv', 'MIMIC_unknown_meds.txt')

    def test_SANDS_recall_expected_drugs(self):
        self.__test_recall('SANDS_med_synonyms.csv', 'SANDS_unknown_meds.txt')

    def test_i2b2_recall_expected_drugs(self):
        self.__test_recall('i2b2_medication_challenge.txt', 'i2b2_unknown_meds.txt')

    def test_deleys_bluebutton_recall_expected_drugs(self):
        self.__test_recall('deleys_bluebutton_medication.txt', 'deleys_bluebutton_unknown_meds.txt')

    def __test_recall(self, term_file, out_file):
        self.__annotator.recall_expected(os.path.join(os.path.dirname(__file__), term_file),
                                         os.path.join(os.path.dirname(__file__), out_file))


if __name__ == '__main__':
    unittest.main()
