import unittest

from text2phenotype.apiclients.feature_service import FeatureServiceClient
from text2phenotype.tasks.task_enums import TaskOperation

from biomed.diagnosis.diagnosis import get_icd_response
from biomed.common.helpers import feature_list_helper


class TestICDResponse(unittest.TestCase):
    TEXT = "Diagnosis List: \n Colon Cancer (C18.9) \n Type 1 diabetes mellitus with other diabetic arthropathy (E10.618)\n"

    def test_get_icd_represented_use_case(self):
        features = feature_list_helper({TaskOperation.icd10_diagnosis})
        tokens, vectors = FeatureServiceClient().annotate_vectorize(text=self.TEXT, features=features)
        icd_output = get_icd_response(text=self.TEXT, tokens=tokens, vectors=vectors)

        expected = {
            'VersionInfo': [
                {'product_id': None,
                 'product_version': '2.04',
                 'tags': [],
                 'active_branch': None,
                 'commit_id': None,
                 'docker_image': None,
                 'commit_date': None,
                 'model_versions': {'diagnosis': '2.01'}}],
            'DiseaseDisorder': [
                {'text': 'Colon Cancer',
                 'range': [18, 30],
                 'score': 0.9994293357348234,
                 'label': 'diagnosis',
                 'polarity': 'positive',
                 'code': 'C18.9',
                 'cui': 'C0007102',
                 'tui': 'T191',
                 'vocab': 'ICD10CM',
                 'preferredText': 'Malignant neoplasm of colon, unspecified site',
                 'page': None},
                {'text': 'Type 1 diabetes mellitus with other diabetic',
                 'range': [41, 85],
                 'score': 0.9993852236637917,
                 'label': 'diagnosis',
                 'polarity': 'positive',
                 'code': 'E10.618',
                 'cui': 'C2874058',
                 'tui': 'T047',
                 'vocab': 'ICD10CM',
                 'preferredText': 'Type 1 diabetes mellitus with other diabetic arthropathy',
                 'page': None},
                {'text': 'arthropathy',
                 'range': [86, 97],
                 'score': 0.9977976608274041,
                 'label': 'diagnosis',
                 'polarity': 'positive',
                 'code': 'E10.618',
                 'cui': 'C2874058',
                 'tui': 'T047',
                 'vocab': 'ICD10CM',
                 'preferredText': 'Type 1 diabetes mellitus with other diabetic arthropathy',
                 'page': None}],
            'SignSymptom': [],
            'ICD10ClinicalCode': [
                {'text': 'C18.9',
                 'range': [32, 37],
                 'score': 0,
                 'label': None,
                 'polarity': None,
                 'code': 'C18.9',
                 'cui': 'C0007102',
                 'tui': 'T191',
                 'vocab': 'ICD10CM',
                 'preferredText': 'Malignant neoplasm of colon, unspecified',
                 'page': None},
                {'text': 'E10.618',
                 'range': [99, 106],
                 'score': 0,
                 'label': None,
                 'polarity': None,
                 'code': 'E10',
                 'cui': 'C3837958',
                 'tui': 'T047',
                 'vocab': 'ICD10CM',
                 'preferredText': 'ketosis-prone diabetes (mellitus)',
                 'page': None}]}

        # ensure that icd10 clinical code mapping part is the same
        for entry in icd_output['ICD10ClinicalCode']:
            self.assertIn(entry, expected['ICD10ClinicalCode'])

        # ensure that all diagnosis and signsymptoms ahve the icd10 representation
        for key in ['DiseaseDisorder', 'SignSymptom']:
            for entry in icd_output[key]:
                self.assertIn('ICD10', entry['vocab'])
