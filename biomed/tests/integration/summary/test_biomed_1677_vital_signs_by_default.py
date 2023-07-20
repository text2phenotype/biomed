import unittest

from biomed.common.helpers import feature_list_helper
from biomed.vital_signs.vital_signs import get_vital_signs
from text2phenotype.common.common import read_text
from text2phenotype.apiclients.feature_service import FeatureServiceClient

from biomed.tests.fixtures import example_file_paths
from text2phenotype.tasks.task_enums import TaskOperation


class TestVitalSigns(unittest.TestCase):
    def test_run_through_carolyn_blose(self):
        text = read_text(example_file_paths.carolyn_blose_txt_filepath)
        features = feature_list_helper({TaskOperation.vital_signs})
        fs_client = FeatureServiceClient()
        tokens, vectors = fs_client.annotate_vectorize(text=text, features=features)
        vital_signs = get_vital_signs(tokens=tokens, vectors=vectors, text=text)
        expected = {
            'VitalSigns': [
                {'text': '84', 'range': [3039, 3041], 'label': 'heart_rate', 'date': None, 'page': None},
                {'text': '168/74', 'range': [3061, 3067], 'label': 'blood_pressure', 'date': None, 'page': None},
                {'text': '16', 'range': [3100, 3102], 'label': 'respiratory_rate', 'date': None, 'page': None},
                {'text': '168/74', 'range': [6163, 6169], 'label': 'blood_pressure', 'date': '2017-07-19', 'page': None},
                {'text': '84', 'range': [6177, 6179], 'label': 'heart_rate', 'date': '2017-07-19', 'page': None}],
            'VersionInfo': [
                {'product_id': None, 'product_version': '0.01', 'tags': [], 'active_branch': None, 'commit_id': None,
                 'docker_image': None, 'commit_date': None, 'model_versions': {'vital_signs': '0.00'}}]}
        vital_signs_list = vital_signs.get('VitalSigns')
        self.assertEqual(len(vital_signs_list), 5)
        for i in range(len(vital_signs_list)):
            for k, v in vital_signs_list[i].items():
                if k != 'score':
                    self.assertEqual(vital_signs_list[i][k], expected['VitalSigns'][i][k])

