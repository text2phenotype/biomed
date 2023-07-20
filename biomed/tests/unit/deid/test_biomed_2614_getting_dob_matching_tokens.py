import datetime
import unittest

from biomed.deid.global_redaction_helper_functions import dates_match_strict, dob_date_time_from_demographic_json


class TestBiomed2614(unittest.TestCase):
    def test_dates_match(self):
        self.assertTrue(
            dates_match_strict({'year': 2012, 'month': 8, 'day': 20}, datetime.datetime(month=8, day=20, year=2012)))
        self.assertFalse(
            dates_match_strict({'year': 2022, 'month': 8, 'day': 20}, datetime.datetime(month=8, day=20, year=2012)))
        self.assertFalse(
            dates_match_strict({'year': 2012, 'month': 10, 'day': 20}, datetime.datetime(month=8, day=20, year=2012)))
        self.assertFalse(
            dates_match_strict({'year': 2012, 'month': 8, 'day': 21}, datetime.datetime(month=8, day=20, year=2012)))

    def test_dob_datetime(self):
        self.assertIsNone(dob_date_time_from_demographic_json(demographic_json={'dob': []}))
        self.assertEqual(
            dob_date_time_from_demographic_json(demographic_json={'dob': [['03/04/1999']]}),
            datetime.datetime(month=3, day=4, year=1999))
