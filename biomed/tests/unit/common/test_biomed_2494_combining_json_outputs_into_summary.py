import copy
import unittest

from biomed.common.biomed_summary import combine_all_biomed_outputs
from text2phenotype.constants.common import VERSION_INFO_KEY
from text2phenotype.tasks.task_enums import TaskOperation


class TestCombineBiomedOutputs(unittest.TestCase):
    output_1 = {
        "doctype": [{
            "text": "ROI (retrieval of information) cover sheet", "range": [0, 2001],
            "score": 0.9892567992210388, "label": "non_clinical", "page": 1},
            {"text": "demographics east tucson geriatrics ram subbureddi", "range": [2002, 2937],
             "score": 0.7984766960144043, "label": "non_clinical", "page": 2},
            {"text": "exactcare 8333 rockside road valley mew 44125 877 ", "range": [2938, 4743],
             "score": 0.5865712761878967, "label": "other_clinical_doc", "page": 3}],
        "VersionInfo": [{
            "product_id": None, "product_version": "2.07", "tags": [], "active_branch": None, "commit_id": None,
            "docker_image": None, "commit_date": None, "model_versions": {"doc_type": "0.00"}}]}
    output_2 = {
        'DiseaseDisorder': [
            {"text": "Essential hypertension", "range": [33893, 33915], "score": 0.9995164898768885,
             "label": "diagnosis", "page": 27, "polarity": "positive", "code": "59621000", "cui": "C0085580",
             "tui": "T047", "vocab": "SNOMEDCT_US", "preferredText": "Essential Hypertension"},
            {"text": "Coronary artery disezse", "range": [33947, 33970], "score": 0.9924938015321777,
             "label": "diagnosis", "page": 27, "polarity": None, "code": None, "cui": None, "tui": None, "vocab": None,
             "preferredText": None},
            {"text": "pectoris", "range": [34035, 34043], "score": 0.7831046189334715, "label": "diagnosis", "page": 27,
             "polarity": None, "code": None, "cui": None, "tui": None, "vocab": None, "preferredText": None},
            {"text": "Noorheumatic aortic valve stenosis", "range": [34064, 34098], "score": 0.9965815369682053,
             "label": "diagnosis", "page": 27, "polarity": "negative", "code": "60573004", "cui": "C0003507",
             "tui": "T047", "vocab": "SNOMEDCT_US", "preferredText": "Aortic Valve Stenosis"}],
        'SignSymptom': [],
        'Medication': [
            {"text": "LISINOPRIL", "range": [5475, 5485], "score": 0.999962982901923, "label": "med", "page": 4,
             "polarity": "positive", "code": "29046", "cui": "C0065374", "tui": "T116", "vocab": "RXNORM",
             "preferredText": "Lisinopril", "date": "1933-11-01", "medFrequencyNumber": [], "medFrequencyUnit": [],
             "medStrengthNum": [], "medStrengthUnit": ["MG", 5488, 5490]}
        ],
        'Allergy': []
    }

    def test_single_upload(self):
        full_summary = combine_all_biomed_outputs([self.output_1])
        self.assertEqual(full_summary.to_json(), {k: v for k, v in self.output_1.items() if k != VERSION_INFO_KEY})

    def test_combine_mult_same(self):
        full_summary = combine_all_biomed_outputs([copy.deepcopy(self.output_1), copy.deepcopy(self.output_1)])
        self.assertEqual(full_summary.to_json(), {k: v for k, v in self.output_1.items() if k != VERSION_INFO_KEY})

    def test_combine_mult(self):
        full_summary = combine_all_biomed_outputs([copy.deepcopy(self.output_1), copy.deepcopy(self.output_2)])
        for k in self.output_1:
            if k != VERSION_INFO_KEY:
                self.assertEqual(full_summary.to_json()[k], self.output_1[k])
        for k in self.output_2:
            if k != VERSION_INFO_KEY:
                self.assertEqual(full_summary.to_json()[k], self.output_2[k])
