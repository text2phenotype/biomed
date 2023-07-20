from biomed.constants.constants import ModelType
from biomed.models.testing_reports import ConfusionPrecisionMisclassReport
from biomed.tests.nightly.qa_testing.base_test_case import BaseQATest


class DemographicQA(BaseQATest):
    @classmethod
    def initialize(cls):
        cls.modelType = ModelType.demographic
        cls.use_threshold = True
        cls.baseline_report = ConfusionPrecisionMisclassReport.parse_classification_text_to_df("""           precision    recall  f1-score   support
                   na       0.99      0.99      0.99     49289
                  ssn       1.00      1.00      1.00         5
                  mrn       0.59      0.88      0.72       120
            pat_first       0.84      0.93      0.75       119
           pat_middle       0.85      1.00      0.78         6
             pat_last       0.77      0.85      0.78       184
         pat_initials       0.00      0.00      0.00         0
              pat_age       0.55      0.95      0.67        23
           pat_street       0.92      0.61      0.43       139
              pat_zip       0.88      0.42      0.05        38
             pat_city       0.86      0.38      0.09        65
            pat_state       0.96      0.50      0.29        48
            pat_phone       0.83      0.90      0.78        84
            pat_email       0.00      0.00      0.00         0
            insurance       0.00      0.00      0.00         1
        facility_name       0.73      0.94      0.84       485
             dr_first       0.75      0.86      0.81       268
            dr_middle       1.00      0.30      0.46        10
              dr_last       0.80      0.87      0.78       330
          dr_initials       0.00      0.00      0.00         0
            dr_street       0.76      0.97      0.86       596
               dr_zip       0.87      0.96      0.88       170
              dr_city       0.80      0.93      0.78       189
             dr_state       0.85      0.99      0.85       170
             dr_phone       0.90      0.88      0.88       315
               dr_fax       0.85      1.00      0.96        23
             dr_email       0.00      0.00      0.00         4
                dr_id       0.00      0.00      0.00         0
               dr_org       0.00      0.00      0.00         0
                  sex       0.82      0.98      0.91        78
                  dob       0.85      0.98      0.91       151
                 race       0.00      0.00      0.00         0
            ethnicity       0.47      0.65      0.54        54
             language       0.00      0.00      0.00         0
          avg / total       0.80      0.89      0.78      3675""")