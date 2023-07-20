import unittest

from biomed.common.combined_model_label import CombinedAnnotation
from biomed.constants.constants import DATE_OF_SERVICE_CATEGORY
from biomed.constants.model_constants import MODEL_TYPE_2_CONSTANTS, ModelType
from biomed.constants.response_mapping import KEY_TO_ASPECT_OUTPUT_CLASSES
from biomed.diagnosis.diagnosis import ICD10_CATEGORY
from openemr-demo.tasks.task_enums import TaskOperation


class TestKeyToAspectClass(unittest.TestCase):
    def test_all_labels_represented(self):
        combined = False
        for model_type in MODEL_TYPE_2_CONSTANTS:
            if model_type in [ModelType.disability, ModelType.meta, ModelType.doc_type, ModelType.date_of_service]:
                continue  # skip dos and doc type bc their category names are derived differently,
                # skip others bc not in prod
            model_constants = MODEL_TYPE_2_CONSTANTS[model_type]
            model_label = model_constants.label_class

            # loop through enum, handle teh fact that combined outputs result in two seperate output categories
            for label in model_label:
                if isinstance(label.value, CombinedAnnotation):
                    combined = True
                    break
                else:
                    combined = False
                    break

            if combined:
                for label in model_label:
                    self.assertIn(label.value.category_label, KEY_TO_ASPECT_OUTPUT_CLASSES)
            else:
                self.assertIn(model_label.get_category_label().persistent_label, KEY_TO_ASPECT_OUTPUT_CLASSES)

    def test_additional_keys_included(self):
        # some operations use other labels, this ensures they are included
        other_category_headers = [TaskOperation.doctype.value, ICD10_CATEGORY, DATE_OF_SERVICE_CATEGORY]
        for cat_header in other_category_headers:
            self.assertIn(cat_header, KEY_TO_ASPECT_OUTPUT_CLASSES)
