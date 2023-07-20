import unittest

from text2phenotype.constants.features import DemographicEncounterLabel

from biomed.demographic.demographic import add_zipcode_demographics


class TestZipCodeDemographics(unittest.TestCase):
    def test_extract_city_state_valid_zip(self):
        test_input = [{'city': 'PALO ALTO', 'state': 'CA', 'country': 'US'}]
        zip_dems = add_zipcode_demographics(test_input, 'dr')
        zip_dems_json = [dem.to_dict() for dem in zip_dems]
        self.assertTrue(isinstance(zip_dems, list))
        self.assertIn({'text': 'PALO ALTO', 'label': DemographicEncounterLabel.dr_city.value.persistent_label, 'score': 0, 'range': [0, 0], 'page': None}, zip_dems_json)
        self.assertIn({'text': 'CA', 'label': DemographicEncounterLabel.dr_state.value.persistent_label, 'score': 0, 'range': [0, 0], 'page': None}, zip_dems_json)

    def test_extract_zip_without_annotation(self):
        zip_dems = add_zipcode_demographics(None, 'dr')
        self.assertTrue(isinstance(zip_dems, list))
        self.assertEqual(len(zip_dems), 0)

    def test_extract_zip_with_empty_annotation(self):
        zip_dems = add_zipcode_demographics([], 'dr')
        self.assertTrue(isinstance(zip_dems, list))
        self.assertEqual(len(zip_dems), 0)
