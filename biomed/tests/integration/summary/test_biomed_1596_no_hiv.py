from biomed.tests.integration.summary.patient_test_case import PatientTestCase


class TestEndToEndCarolynBlose(PatientTestCase):
    __test__ = True

    @classmethod
    def initialize(cls):
        cls.text = "HIV antibodiess, patient tested positive for HIV. History of HIV"


