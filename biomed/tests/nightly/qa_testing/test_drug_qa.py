from biomed.constants.constants import ModelType
from biomed.models.testing_reports import ConfusionPrecisionMisclassReport
from biomed.tests.nightly.qa_testing.base_test_case import BaseQATest


class DrugQA(BaseQATest):
    @classmethod
    def initialize(cls):
        cls.modelType = ModelType.drug
        cls.baseline_report = ConfusionPrecisionMisclassReport.parse_classification_text_to_df(
            """            precision    recall  f1-score   support

         na       1.00      1.00      1.00     71076
    allergy       0.73      0.78      0.76       176
        med       0.85      0.86      0.85      1980

avg / total       0.84      0.85      0.85      2156""")
