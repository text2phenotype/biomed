import unittest

from biomed.constants.constants import BiomedVersionInfo, TaskOperation, OperationToModelType,\
    BIOMED_VERSION_TO_MODEL_VERSION, ModelType
from biomed.constants.model_constants import MODEL_TYPE_2_CONSTANTS


class TestVersionInfo(unittest.TestCase):
    def test_task_model_version_info(self):
        for task in TaskOperation:
            if task in TaskOperation.biomed_operations():
                biomed_version_info = BiomedVersionInfo(task_operation=task)
                # make sure that biomed version included is accepted
                self.assertIn(biomed_version_info.product_version, BIOMED_VERSION_TO_MODEL_VERSION)
                expected_model_types = {mt for mt in OperationToModelType[task]}
                self.assertEqual(len(biomed_version_info.model_versions), len(expected_model_types))
                # assert valid model_versions
                for model_type in expected_model_types:
                    self.assertIn(biomed_version_info.model_versions[model_type.name],
                                  MODEL_TYPE_2_CONSTANTS[model_type].model_version_mapping())

    def test_write_versioning(self):
        for model in ModelType:
            if model not in [ModelType.meta, ModelType.disability]:
                print(model)
                biomed_verison_info = BiomedVersionInfo(model_type=model)

                biomed_version_json = biomed_verison_info.to_dict()
                self.assertListEqual(list(biomed_version_json['model_versions']), [model.name])


