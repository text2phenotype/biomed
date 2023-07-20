from unittest import TestCase
import os

from text2phenotype.tests.decorators import skip_on_docker_build

from biomed.biomed_env import BiomedEnv
from biomed.constants.model_constants import MODEL_TYPE_2_CONSTANTS
from biomed.models.get_model import (
    get_model_from_model_folder, get_model_metadata_fp_from_model_folder, get_model_folder_path
)
from biomed.models.model_base import ModelBase

from biomed.constants.constants import ModelType, BIOMED_VERSION_TO_MODEL_VERSION, get_version_model_folders
from biomed.models.model_wrapper import ModelWrapper


@skip_on_docker_build
class TestBiomed1493(TestCase):
    def test_models_initialization(self):
        for biomed_version in BIOMED_VERSION_TO_MODEL_VERSION:
            for model_type in ModelType:
                model_constants = MODEL_TYPE_2_CONSTANTS[model_type]
                if model_constants.production:
                    # if any model type other than mpi/meta aren't included in versioning info
                    if model_type not in BIOMED_VERSION_TO_MODEL_VERSION[biomed_version]:
                        raise ValueError(f'Model Type {model_type.name} not included in BIOMED VERSION'
                                         f' {biomed_version} mapping')
                    model_files = get_version_model_folders(model_type, biomed_version=biomed_version)

                    for model_file in model_files:
                        metadata_file = get_model_metadata_fp_from_model_folder(
                            model_folder=model_file, model_type=model_type)
                        self.assertTrue(os.path.isfile(metadata_file))
                        pm = get_model_from_model_folder(model_folder=model_file, base_model_type=model_type)
                        try:
                            model_wrapper = pm.get_cached_model()
                            self.assertIsInstance(model_wrapper, ModelWrapper, f'model_type: {model_type} model_file: {model_file}')
                        except ValueError as e:
                            raise ValueError(f"Error in caching {model_type.name}, file {model_file}") from e
                        except:
                            raise ValueError(f"NO file found for {model_type.name}, file {model_file}")

    def test_bad_model_folder(self):
        i = 1
        model_type_to_ver = BIOMED_VERSION_TO_MODEL_VERSION[BiomedEnv.DEFAULT_BIOMED_VERSION.value]
        model_type, production_ver = list(model_type_to_ver.items())[0]  # just grab the first one
        with self.assertRaises(ValueError):
            _ = ModelBase(model_type=model_type, model_folder_name="my_fake_folder")

    def test_model_path(self):
        self.assertEqual(
            get_model_folder_path('abc', ModelType.diagnosis).split('resources')[1],
            '/files/diagnosis/abc')
        with self.assertRaises(AssertionError):
            get_model_folder_path('abc', None)
        with self.assertRaises(AssertionError):
            get_model_folder_path('abc', model_type=None, model_type_name=None)
