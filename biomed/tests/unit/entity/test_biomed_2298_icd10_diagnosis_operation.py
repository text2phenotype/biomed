import unittest

from text2phenotype.common.featureset_annotations import MachineAnnotation
from text2phenotype.tasks.task_enums import TaskOperation
from text2phenotype.constants.features.feature_type import FeatureType

from biomed.common.helpers import feature_list_helper, get_operation_representation_features
from biomed.diagnosis.diagnosis import get_best_icd10_code_representation, get_biomed_output_from_clinical_code_feature


class TestBiomed2298(unittest.TestCase):
    def test_operation_representation_features(self):
        feats = get_operation_representation_features({TaskOperation.icd10_diagnosis})
        self.assertSetEqual(feats, {FeatureType.icd10_diagnosis, FeatureType.clinical_code_icd10})

    def test_feature_helper(self):
        features = feature_list_helper({TaskOperation.icd10_diagnosis})
        self.assertIn(FeatureType.clinical_code_icd10, features)
        self.assertIn(FeatureType.icd10_diagnosis, features)

    def test_get_icd10_code_from_annot_single(self):
        annot_single = [
            {
                "DiseaseSign": [{
                    "code": "732", "cui": "C0002679", "tui": ["T195"], "tty": [], "preferredText": "Amphotericin B",
                    "codingScheme": "RXNORM"}], "polarity": "positive"}]

        umls = get_best_icd10_code_representation(annot_single)
        self.assertEqual(umls['code'], '732')

    def test_get_code_from_annot_multiple(self):
        annot_single = [
            {
                "DiseaseSign": [{
                    "code": "M46"}, {'code': 'M46.15'}], "polarity": "positive"}]

        umls = get_best_icd10_code_representation(annot_single)
        self.assertEqual(umls['code'], 'M46.15')

    def test_get_code_from_annot_none(self):
        annot_single = [
            {
                "DiseaseSign": [{
                    "tui": "M46"}, {'cui': 'M46.15'}], "polarity": "positive"}]

        umls = get_best_icd10_code_representation(annot_single)
        self.assertEqual(umls, None)

    def test_predict_get_icd_10_codes(self):
        tokens = MachineAnnotation(json_dict_input={'clinical_code_icd10': {
            '2': [{
                'SignSymptom': [{
                    'code': 'M54.9',
                    'codingScheme': 'ICD10CM',
                    'cui': 'C0004604',
                    'preferredText': 'Back pain NOS',
                    'tty': ['AB', 'PT', 'ET'],
                    'tui': ['T184']},
                    {'code': 'M54.9',
                     'codingScheme': 'ICD10',
                     'cui': 'C0004604',
                     'preferredText': 'Back pain NOS',
                     'tty': ['PT'],
                     'tui': ['T184']},
                    {'code': 'M54',
                     'codingScheme': 'ICD10CM',
                     'cui': 'C0004604',
                     'preferredText': 'Back pain NOS',
                     'tty': ['AB', 'HT'],
                     'tui': ['T184']},
                    {'code': 'M54',
                     'codingScheme': 'ICD10',
                     'cui': 'C0004604',
                     'preferredText': 'Back pain NOS',
                     'tty': ['HT'],
                     'tui': ['T184']}],
                'polarity': 'positive'}]},
            'range': [[0, 4], [4, 5], [7, 13]],
            'speech': ['NN', ':', 'NN'],
            'token': ['code', ':', 'M54.16']
        })

        resp = get_biomed_output_from_clinical_code_feature(tokens=tokens)
        self.assertEqual(len(resp), 1)
        self.assertDictEqual(
            resp[0].to_dict(),
            {
                'text': 'M54.16',
                'range': [7, 13],
                'score': 0,
                'label': None,
                'polarity': None,
                'code': 'M54.9',
                'cui': 'C0004604',
                'tui': 'T184',
                'vocab': 'ICD10CM',
                'preferredText': 'Back pain NOS',
                'page': None
            })
