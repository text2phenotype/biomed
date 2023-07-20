import unittest

from biomed.demographic.best_demographics import get_standardized_race, RaceCodesToRegexMapping, \
    get_standardized_ethnicity, EthnicityCodesToRegex, get_all_races, get_best_ethnicity
from biomed.demographic.demographics_manipulation import FetchedDemographics, get_best_demographics
from biomed.reassembler import reassemble_functions
from text2phenotype.constants.common import VERSION_INFO_KEY


class TestRaceDemographicsParts(unittest.TestCase):
    DEM_TOKEN_PRED = {
        "Demographic": [
            {"text": "Shannon", "range": [0, 7], "score": 0.98, "label": "pat_first"},
            {"text": "Fee", "range": [8, 11], "score": 0.76, "label": "pat_last"},
            {"text": "White", "range": [23, 28], "score": 0.67, "label": "race"},
            {"text": "non-hispanic", "range": [30, 42], "score": 0.5, "label": "ethnicity"},
        ]}

    def test_fetched_demographics(self):
        fetched_dem = FetchedDemographics(demographics_list=self.DEM_TOKEN_PRED['Demographic'])
        self.assertEqual(fetched_dem.demographics.race, [('White', 0.67)])
        self.assertEqual(fetched_dem.demographics.ethnicity, [('non-hispanic', 0.5)])
        self.assertEqual(fetched_dem.demographics.pat_full_name, [('Shannon Fee', 0.87)])

    def test_race_standardization(self):
        self.assertEqual(get_standardized_race("white"), RaceCodesToRegexMapping.WHITE.name)
        self.assertEqual(get_standardized_race("black"), RaceCodesToRegexMapping.BLACK_OR_AFRICAN_AMERICAN.name)
        self.assertEqual(get_standardized_race("african"), RaceCodesToRegexMapping.BLACK_OR_AFRICAN_AMERICAN.name)
        self.assertEqual(
            get_standardized_race("black or african american"),
            RaceCodesToRegexMapping.BLACK_OR_AFRICAN_AMERICAN.name)
        self.assertEqual(
            get_standardized_race("african american"),
            RaceCodesToRegexMapping.BLACK_OR_AFRICAN_AMERICAN.name)
        self.assertEqual(
            get_standardized_race("native american"),
            RaceCodesToRegexMapping.AMERICAN_INDIAN_OR_ALASKA_NATIVE.name)
        self.assertEqual(
            get_standardized_race("american indian or alaska native"),
            RaceCodesToRegexMapping.AMERICAN_INDIAN_OR_ALASKA_NATIVE.name)
        self.assertEqual(
            get_standardized_race("american indian"),
            RaceCodesToRegexMapping.AMERICAN_INDIAN_OR_ALASKA_NATIVE.name)
        self.assertEqual(
            get_standardized_race("alaska native"),
            RaceCodesToRegexMapping.AMERICAN_INDIAN_OR_ALASKA_NATIVE.name)
        self.assertEqual(
            get_standardized_race("native hawaiian"),
            RaceCodesToRegexMapping.NATIVE_HAWAIIAN_OR_OTHER_PACIFIC_ISLANDER.name)
        self.assertEqual(
            get_standardized_race("hawaiian"),
            RaceCodesToRegexMapping.NATIVE_HAWAIIAN_OR_OTHER_PACIFIC_ISLANDER.name)
        self.assertEqual(
            get_standardized_race("Pacific islander"),
            RaceCodesToRegexMapping.NATIVE_HAWAIIAN_OR_OTHER_PACIFIC_ISLANDER.name)
        self.assertEqual(get_standardized_race("MULTIPLE"), RaceCodesToRegexMapping.MULTIPLE.name)
        self.assertEqual(get_standardized_race("RACE UNKNOWN"), RaceCodesToRegexMapping.UNKNOWN.name)

    def test_ethnicity_standardization(self):
        self.assertEqual(get_standardized_ethnicity("NOT HISPANIC"), EthnicityCodesToRegex.Not_Hispanic_or_Latino.name)
        self.assertEqual(get_standardized_ethnicity('non hispanic'), EthnicityCodesToRegex.Not_Hispanic_or_Latino.name)
        self.assertEqual(get_standardized_ethnicity('hispanic'), EthnicityCodesToRegex.Hispanic_or_Latino.name)
        self.assertEqual(get_standardized_ethnicity('latino'), EthnicityCodesToRegex.Hispanic_or_Latino.name)
        self.assertEqual(get_standardized_ethnicity('random_text'), EthnicityCodesToRegex.Unknown.name)
        self.assertEqual(get_standardized_ethnicity('nonhispanic'), EthnicityCodesToRegex.Not_Hispanic_or_Latino.name)
        self.assertEqual(get_standardized_ethnicity('non-hispanic'), EthnicityCodesToRegex.Not_Hispanic_or_Latino.name)

    def test_get_all_race(self):
        race_input = [('black', 0.2), ('african american', 0.8), ('white', 0.7), ('multiple', 0.4), ('aksdhfj', 0.1)]
        all_out = get_all_races(race_input)
        expected = [
            (RaceCodesToRegexMapping.BLACK_OR_AFRICAN_AMERICAN.name, 0.8),
            (RaceCodesToRegexMapping.WHITE.name, 0.7),
            (RaceCodesToRegexMapping.MULTIPLE.name, 0.4)
        ]
        self.assertEqual(all_out, expected)

    def test_get_best_ethnicity(self):
        ethnicity_input = [('non hispanic', 0.8), ('not hispanic', 0.4), ('hispanic', 0.8)]
        expected = [(EthnicityCodesToRegex.Not_Hispanic_or_Latino.name, 0.8)]
        self.assertEqual(get_best_ethnicity(ethnicity_input), expected)

        ethnicity_2 = [('abda;kdf', 0.8), ('hispanic', 0.8)]
        expected_2 = [(EthnicityCodesToRegex.Hispanic_or_Latino.name, 0.8)]
        self.assertEqual(get_best_ethnicity(ethnicity_2), expected_2)

        ethnicity_3 = [('unk', 0.7)]
        expected_3 = [(EthnicityCodesToRegex.Unknown.name, 0.7)]
        self.assertEqual(get_best_ethnicity(ethnicity_3), expected_3)

    def test_race_normalized_in_get_best_demographics(self):
        fetched_dem = FetchedDemographics(demographics_list=self.DEM_TOKEN_PRED['Demographic'])
        demographics_out_dict = get_best_demographics(fetched_dem).to_final_dict()
        expected = {
            'ssn': [], 'mrn': [], 'sex': [], 'dob': [], 'pat_first': [('Shannon', 0.98)], 'pat_last': [('Fee', 0.76)],
            'pat_age': [], 'pat_street': [], 'pat_zip': [], 'pat_city': [], 'pat_state': [], 'pat_phone': [],
            'pat_email': [], 'insurance': [], 'facility_name': [], 'dr_first': [], 'dr_last': [],
            'pat_full_name': [('Shannon Fee', 0.87)], 'dr_full_names': [],
            'race': [('WHITE', 0.67)], 'ethnicity': [('Not_Hispanic_or_Latino', 0.5)]}
        self.assertEqual(demographics_out_dict, expected)

    def test_reassemble_race_ethnicity_demographics(self):
        out = reassemble_functions.reassemble_demographics([([0, 100], self.DEM_TOKEN_PRED)])
        expected = {
            'ssn': [], 'mrn': [], 'sex': [], 'dob': [], 'pat_first': [('Shannon', 0.98)], 'pat_last': [('Fee', 0.76)],
            'pat_age': [], 'pat_street': [], 'pat_zip': [], 'pat_city': [], 'pat_state': [], 'pat_phone': [],
            'pat_email': [], 'insurance': [], 'facility_name': [], 'dr_first': [], 'dr_last': [],
            'pat_full_name': [('Shannon Fee', 0.87)], 'dr_full_names': [],
            'race': [('WHITE', 0.67)], 'ethnicity': [('Not_Hispanic_or_Latino', 0.5)]}
        del out[VERSION_INFO_KEY]
        self.assertEqual(out, expected)
