from biomed.constants.constants import ModelType
from biomed.models.testing_reports import ConfusionPrecisionMisclassReport
from biomed.tests.nightly.qa_testing.base_test_case import BaseQATest

#TODO: Update these results
class DiseaseSignQA(BaseQATest):
    @classmethod
    def initialize(cls):
        cls.modelType = ModelType.diagnosis
        cls.use_threshold = True
        cls.baseline_report = ConfusionPrecisionMisclassReport.parse_classification_text_to_df("""             precision    recall  f1-score   support

         na       0.99      0.96      0.97     41080
  diagnosis       0.53      0.85      0.65      1349
signsymptom       0.33      0.63      0.44       360

avg / total       0.49      0.81      0.61       977
""")
