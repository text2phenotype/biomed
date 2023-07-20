import unittest

from biomed.demographic.best_demographics import FhirDemographicEnforcement


class TestBiomed1463(unittest.TestCase):
    def test_ssn_not_phone(self):
        input = [['648921023', .4], ['679-19-2338', .3]]
        output = FhirDemographicEnforcement.pat_phone.value(input)
        self.assertEqual(output, [])


