import unittest
import os

from biomed.constants.model_constants import ModelType
from biomed.models.model_cache import get_full_path
from biomed.biomed_env import BiomedEnv

class TestBiomed2307(unittest.TestCase):
    def test_base_assumption(self):
        path = get_full_path(model_file_name='abc/abc.h5', model_type_name=ModelType.diagnosis.name)
        self.assertEqual(path,
                         os.path.join(
                             BiomedEnv.BIOMED_NON_SHARED_MODEL_PATH.value,
                             'resources/files/diagnosis/abc/abc.h5'))

    def test_new_path(self):
        BiomedEnv.BIOMED_NON_SHARED_MODEL_PATH.value = '/foo'
        path = get_full_path(model_file_name='abc/abc.h5', model_type_name=ModelType.diagnosis.name)
        self.assertEqual(path,
                         os.path.join(
                            '/foo',
                             'resources/files/diagnosis/abc/abc.h5'))
        BiomedEnv.BIOM_MODELS_PATH.value = '/tmp'