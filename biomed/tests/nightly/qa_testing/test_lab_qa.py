import pandas
from biomed.constants.constants import ModelType
from biomed.models.testing_reports import ConfusionPrecisionMisclassReport
from biomed.tests.nightly.qa_testing.base_test_case import BaseQATest


class LabQA(BaseQATest):
    @classmethod
    def initialize(cls):
        cls.modelType = ModelType.lab
        cls.baseline_report = ConfusionPrecisionMisclassReport.parse_classification_text_to_df("""             precision    recall  f1-score   support

         na       0.99      0.99      0.99     68002
        lab       0.92      0.88      0.90      1998
  lab_value       0.37      0.84      0.51       101
   lab_unit       0.00      0.00      0.00         6
lab_interp       0.00      0.00      0.00        29

avg / total       0.88      0.73      0.78      2132
""")