import unittest

from biomed.common.biomed_ouput import VitalSignOutput


class TestVitalSignOutput(unittest.TestCase):
    def assert_match(self, vital_str, expected_vital_value, expected_vital_unit):
        unit, value = VitalSignOutput.split_vital_sign(vital_str)
        self.assertEqual(expected_vital_value,  value)
        self.assertEqual(expected_vital_unit, unit)

    def test_simple_number_parse(self):
        self.assert_match('101', 101, None)
        self.assert_match('0', 0, None)
        self.assert_match('0.3', 0.3, None)

    def test_non_numeric_parse(self):
        self.assert_match('*F', None, '*F')

    def test_split_parse(self):
        self.assert_match('101*F', 101, '*F')
        self.assert_match('101.1*F', 101.1, '*F')
        self.assert_match('^&65()', 65, '^&()')
        # this is current behaviour and I believe solving for this use case is out of scope for vitals
        self.assert_match('1/3', 1, '/3')


