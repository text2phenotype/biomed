import unittest
import json
from biomed.deid.utils import demographics_chunk_to_deid
from text2phenotype.constants.features import DemographicEncounterLabel, DEM_to_PHI


class TestDeidFromDemographics(unittest.TestCase):
    demographics_tokens = [
        {'text': 'Page', 'range': [0, 4], 'score': 0.002, 'label': 'pat_first'},
        {'text': 'Blose', 'range': [20, 25], 'score': 0.954, 'label': 'pat_last'},
        {'text': 'Carolyn', 'range': [27, 34], 'score': 0.917, 'label': 'pat_first'},
        {'text': '07/29/1932', 'range': [37, 47], 'score': 0.972, 'label': 'dob'},
        {'text': 'ViSolve', 'range': [74, 81], 'score': 0.117, 'label': 'facility_name'},
        {'text': 'Clinic', 'range': [82, 88], 'score': 0.487, 'label': 'facility_name'},
        {'text': '000-000-0000', 'range': [89, 101], 'score': 0.475, 'label': 'dr_phone'},
        {'text': 'ViSolve', 'range': [102, 109], 'score': 0.119, 'label': 'facility_name'},
        {'text': 'Clinic', 'range': [110, 116], 'score': 0.38, 'label': 'facility_name'},
        {'text': '000-000-0000', 'range': [117, 129], 'score': 0.379, 'label': 'dr_phone'},
        {'text': 'Carolyn', 'range': [130, 137], 'score': 0.532, 'label': 'pat_first'},
        {'text': 'Blose', 'range': [138, 143], 'score': 0.896, 'label': 'pat_last'},
        {'text': 'Carolyn', 'range': [197, 204], 'score': 0.925, 'label': 'pat_first'},
        {'text': 'V.', 'range': [205, 207], 'score': 0.777, 'label': 'pat_middle'},
        {'text': 'Blose', 'range': [208, 213], 'score': 0.9, 'label': 'pat_last'},
        {'text': '530-79-5301', 'range': [227, 238], 'score': 0.96, 'label': 'ssn'},
        {'text': '1932-07-29', 'range': [244, 254], 'score': 0.887, 'label': 'dob'},
        {'text': 'Female', 'range': [260, 266], 'score': 0.995, 'label': 'sex'},
        {'text': '84-year-old', 'range': ['a', 1051], 'score': 0.728, 'label': 'pat_age'},
        {'text': 'female', 'range': [1052, None], 'score': 0.509, 'label': 'sex'}]


    def test_all_demographics_to_deid(self):
        deid_list = demographics_chunk_to_deid(self.demographics_tokens).response_list
        self.assertEqual(len(deid_list), len(self.demographics_tokens)- 3)

    def test_json_write_read(self):
        dem_tokens = json.loads(json.dumps(self.demographics_tokens))
        deid_list = demographics_chunk_to_deid(dem_tokens).response_list
        self.assertEqual(len(deid_list), len(self.demographics_tokens) - 3)

    def test_dem_to_deid(self):
        for dem_type in DemographicEncounterLabel:
            self.assertIsNotNone(DEM_to_PHI.get(DemographicEncounterLabel.from_brat(dem_type.name)), dem_type)
            self.assertIsNotNone(
                DEM_to_PHI.get(DemographicEncounterLabel.from_brat(dem_type.value.persistent_label)), dem_type)
