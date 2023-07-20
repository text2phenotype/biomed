from biomed.constants.constants import ModelType
from biomed.models.testing_reports import ConfusionPrecisionMisclassReport
from biomed.tests.nightly.qa_testing.base_test_case import BaseQATest


class CancerQA(BaseQATest):
    @classmethod
    def initialize(cls):
        cls.modelType = ModelType.oncology
        cls.baseline_report = ConfusionPrecisionMisclassReport.parse_classification_text_to_df("""             precision    recall  f1-score   support
         na       0.99      1.00      0.99     31831
topography       0.96      0.72      0.82       211
morphology       0.99      0.94      0.97       537
   behavior       0.94      0.93      0.94       192
      grade       1.00      0.95      0.97       182
      stage       1.00      0.77      0.87        18
avg / total       0.98      0.90      0.93      1140""")