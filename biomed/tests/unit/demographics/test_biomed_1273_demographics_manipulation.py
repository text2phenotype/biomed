import unittest

from biomed.common.biomed_ouput import BiomedOutput
from text2phenotype.common.demographics import Demographics
from text2phenotype.constants.features import DemographicEncounterLabel

from biomed.demographic.demographics_manipulation import FetchedDemographics, get_best_demographics


class TestDemographicsManipulation(unittest.TestCase):
    def test_none_type_demographics_list(self):
        test_input = None
        fetched_demographics = FetchedDemographics(demographics_list=test_input)
        # assert all values are None and fetched demographics object created
        self.assertTrue(isinstance(fetched_demographics,  FetchedDemographics))
        self.assertEqual(fetched_demographics.pat_names, None)
        self.assertEqual(fetched_demographics.demographics, None)
        self.assertEqual(fetched_demographics.dr_names, None)

        best_values = get_best_demographics(fetched_demographics)
        # test to dict values are also equivalent
        self.assertEqual(best_values.to_dict(), Demographics().to_dict())

    def test_regular_demographics_input(self):
        test_input = [BiomedOutput(text='07/22/1994', label=DemographicEncounterLabel.dob.name, lstm_prob=1,
                                   range=[13, 23]),
                      BiomedOutput(text='Adam', label=DemographicEncounterLabel.pat_first.name, lstm_prob=0.51,
                                   range=[101, 105]),
                      BiomedOutput(text='Francesca', label=DemographicEncounterLabel.pat_first.name, lstm_prob=0.75,
                                   range=[120, 129])]

        fetched_demographics = FetchedDemographics(demographics_list=test_input)
        best_values = get_best_demographics(fetched_demographics)
        self.assertEqual(best_values[DemographicEncounterLabel.pat_first.name], [('Francesca', 0.75)])
        self.assertEqual(best_values[DemographicEncounterLabel.dob.name], [('07/22/1994', 1)])

    def test_empty_list_demogrpahics_list(self):
        test_input = []
        fetched_demographics = FetchedDemographics(demographics_list=test_input)
        # assert all values are None and fetched demographics object created
        self.assertTrue(isinstance(fetched_demographics, FetchedDemographics))
        self.assertEqual(fetched_demographics.pat_names, None)
        self.assertEqual(fetched_demographics.demographics.to_dict(), Demographics().to_dict())
        self.assertEqual(fetched_demographics.dr_names, None)

        best_values = get_best_demographics(fetched_demographics)
        # test to dict values are also equivalent
        self.assertEqual(best_values.to_dict(), Demographics().to_dict())
