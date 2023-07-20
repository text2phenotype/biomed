import unittest

from biomed.constants.model_constants import MODEL_TYPE_2_CONSTANTS

from biomed.models.model_metadata import ModelType


class TestFeatureLabels(unittest.TestCase):
    def test_label_enums_for_all_model_type(self):
        for model_type in MODEL_TYPE_2_CONSTANTS:

            model_constants = MODEL_TYPE_2_CONSTANTS[model_type]
            model_enum = model_constants.label_class
            if not model_constants.production or model_type == ModelType.date_of_service:
                continue
            if model_type != ModelType.smoking:
                self.assertEqual(model_enum.get_from_int(0).name, 'na', f'Failed for model_type {model_type.name}')
            for entry in model_enum:
                self.assertEqual(model_enum.get_from_int(entry.value.column_index), entry, f'Failed for entry: {entry.name} '
                                                                                      f'in model_type {model_type.name}')
