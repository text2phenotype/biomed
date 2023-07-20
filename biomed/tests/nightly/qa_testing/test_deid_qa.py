import pandas
from biomed.constants.constants import ModelType
from biomed.models.testing_reports import ConfusionPrecisionMisclassReport
from biomed.tests.nightly.qa_testing.base_test_case import BaseQATest


class DeidQA(BaseQATest):
    @classmethod
    def initialize(cls):
        cls.modelType = ModelType.deid
        cls.use_threshold = True
        cls.baseline_report = ConfusionPrecisionMisclassReport.parse_classification_text_to_df(
            """      precision    recall  f1-score   support

            na       1.00      0.99      1.00     41708
          date       0.93      0.95      0.94      2290
      hospital       0.00      0.00      0.00         0
           age       0.00      0.00      0.00         0
        street       0.00      0.00      0.00         0
           zip       0.00      0.00      0.00         0
          city       0.00      0.00      0.00         0
         state       0.00      0.00      0.00         0
       country       0.00      0.00      0.00         0
location_other       0.00      0.00      0.00         0
         phone       0.00      0.00      0.00         0
           url       0.00      0.00      0.00         0
           fax       0.00      0.00      0.00         0
         email       0.00      0.00      0.00         0
         idnum       0.00      0.00      0.00         0
         bioid       0.00      0.00      0.00         0
  organization       0.00      0.00      0.00         0
    profession       0.00      0.00      0.00         0
       patient       0.00      0.00      0.00         0
        doctor       0.00      0.00      0.00         0
 medicalrecord       0.00      0.00      0.00         0
      username       0.00      0.00      0.00         0
        device       0.00      0.00      0.00         0
    healthplan       0.00      0.00      0.00         0

   avg / total       0.87      0.97      0.92      2290""")
