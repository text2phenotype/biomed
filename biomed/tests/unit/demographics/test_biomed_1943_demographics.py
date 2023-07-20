import unittest

from biomed.biomed_env import BiomedEnv
from biomed.common.aspect_response import AspectResponse
from biomed.common.biomed_ouput import BiomedOutput
from biomed.constants.constants import ModelType
from biomed.reassembler.reassemble_functions import reassemble_demographics
from text2phenotype.constants.features import DemographicEncounterLabel


class TestBiomedReassembledOutput(unittest.TestCase):
    predicted_token_chunk_1 = AspectResponse(
        category_name=DemographicEncounterLabel.get_category_label().persistent_label,
        response_list=[
            BiomedOutput(
                text='Hermione', lstm_prob=.87, range=[0, 8],
                label=DemographicEncounterLabel.pat_first.value.persistent_label),
            BiomedOutput(
                text='Jean', lstm_prob=.7, range=[10, 14],
                label=DemographicEncounterLabel.pat_first.value.persistent_label),
            BiomedOutput(
                text='Granger', lstm_prob=.7, range=[18, 23],
                label=DemographicEncounterLabel.pat_last.value.persistent_label),
            BiomedOutput(
                text='1987-02-13', lstm_prob=.5, range=[50, 60],
                label=DemographicEncounterLabel.dob.value.persistent_label),
            BiomedOutput(
                text='Female', lstm_prob=.9, range=[25, 32],
                label=DemographicEncounterLabel.sex.value.persistent_label),
            BiomedOutput(
                text='Female', lstm_prob=.9, range=[65, 72],
                label=DemographicEncounterLabel.sex.value.persistent_label),
        ]).to_versioned_json(model_type=ModelType.demographic, biomed_version=BiomedEnv.DEFAULT_BIOMED_VERSION.value)

    predicted_token_chunk_2 = AspectResponse(
        category_name=DemographicEncounterLabel.get_category_label().persistent_label,
        response_list=[
            BiomedOutput(
                text='Granger', lstm_prob=.7, range=[0, 7],
                label=DemographicEncounterLabel.pat_last.value.persistent_label),
            BiomedOutput(
                text='Hermione', lstm_prob=.8, range=[10, 18],
                label=DemographicEncounterLabel.pat_first.value.persistent_label),
            BiomedOutput(
                text='Jean', lstm_prob=.7, range=[20, 24],
                label=DemographicEncounterLabel.pat_middle.value.persistent_label),
            BiomedOutput(
                text='Female', lstm_prob=.9, range=[25, 32],
                label=DemographicEncounterLabel.sex.value.persistent_label),
            BiomedOutput(
                text='f3male', lstm_prob=.2, range=[65, 72],
                label=DemographicEncounterLabel.sex.value.persistent_label),
        ]).to_versioned_json(model_type=ModelType.demographic, biomed_version=BiomedEnv.DEFAULT_BIOMED_VERSION.value)

    expected_output_1 = {
        'ssn': [], 'mrn': [], 'sex': [('Female', 1.0)], 'dob': [('02/13/1987', 0.5)], 'pat_first': [('Hermione Jean', 0.87)],
        'pat_last': [('Granger', 0.7)], 'pat_age': [], 'pat_street': [], 'pat_zip': [], 'pat_city': [], 'pat_state': [],
        'pat_phone': [], 'pat_email': [], 'insurance': [], 'facility_name': [], 'dr_first': [], 'dr_last': [],
        'pat_full_name': [('Hermione Jean Granger', 0.7849999999999999)], 'dr_full_names': []}


    expected_output_2 = {
        'ssn': [], 'mrn': [], 'sex': [('Female', 1.0)], 'dob': [], 'pat_first': [('Hermione', 0.8)],
        'pat_last': [('Granger', 0.7)], 'pat_age': [], 'pat_street': [], 'pat_zip': [], 'pat_city': [], 'pat_state': [],
        'pat_phone': [], 'pat_email': [], 'insurance': [], 'facility_name': [], 'dr_first': [], 'dr_last': [],
        'pat_full_name': [('Hermione Jean Granger', 0.7333333333333334)], 'dr_full_names': []}

    expected_output_combined = {
        'ssn': [], 'mrn': [], 'sex': [('Female', 1.0)], 'dob': [('02/13/1987', 0.5)], 'pat_first': [('Hermione Jean', 0.87)],
        'pat_last': [('Granger', 0.7)], 'pat_age': [], 'pat_street': [], 'pat_zip': [], 'pat_city': [], 'pat_state': [],
        'pat_phone': [], 'pat_email': [], 'insurance': [], 'facility_name': [], 'dr_first': [], 'dr_last': [],
        'pat_full_name': [('Hermione Jean Granger', 0.7849999999999999)], 'dr_full_names': []}

    def test_reassemble_demographics_chunk_1(self):
        chunk_mapping = [((0, 500), self.predicted_token_chunk_1)]
        output = reassemble_demographics(chunk_mapping)
        for key in self.expected_output_1:
            self.assertEqual(self.expected_output_1[key], output[key])

    def test_reasssemble_dem_chunk_2(self):
        chunk_mapping = [((0, 500), self.predicted_token_chunk_2)]
        output = reassemble_demographics(chunk_mapping)
        for key in self.expected_output_2:
            self.assertEqual(self.expected_output_2[key], output[key])

    def test_reassemble_dem_chunks(self):
        chunk_mapping = [((0, 500), self.predicted_token_chunk_2), ((502, 650),  self.predicted_token_chunk_1)]
        output = reassemble_demographics(chunk_mapping)
        for key in self.expected_output_combined:
            self.assertEqual(self.expected_output_combined[key], output[key])

