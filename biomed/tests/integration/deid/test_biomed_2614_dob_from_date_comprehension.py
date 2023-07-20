import datetime
import unittest

from text2phenotype.apiclients import FeatureServiceClient
from text2phenotype.constants.features import FeatureType, PHILabel

from biomed.deid.global_redaction_helper_functions import phi_from_dob_match, globally_redact_phi_tokens


class TestPHIFromDOBMatch(unittest.TestCase):
    TEXT = "Casper the Ghost was born on July 21, 2018. Encounter_date: 07/21/2018 07-21-2018 07-21-18"
    FS_CLIENT = FeatureServiceClient()

    def test_text_finding_matches(self):
        annot = self.FS_CLIENT.annotate(text=self.TEXT, features={FeatureType.date_comprehension})

        phi_tokens = phi_from_dob_match(tokens=annot, dob=datetime.datetime(year=2018, month=7, day=21))
        self.assertEqual(len(phi_tokens), 8)
        for entry in phi_tokens:
            self.assertEqual(entry.label, 'patient')
            self.assertEqual(entry.lstm_prob, 1)

    def test_add_matching_dobs_to_phi(self):
        annot = self.FS_CLIENT.annotate(text=self.TEXT, features={FeatureType.date_comprehension})
        phi_out = globally_redact_phi_tokens(
            demographic_json={'dob': [['07/21/2018', .9]]},
            machine_annotation=annot,
            phi_token_json={
                PHILabel.get_category_label().persistent_label:
                    [{'text': 'Casper', 'range': [0, 6], 'label': 'patient', 'score': 0.87}]}
        )
        phi_out.postprocess(text=self.TEXT)
        expected_dict_out = {'PHI': [{'label': 'patient',
          'page': None,
          'range': [0, 6],
          'score': 0.87,
          'text': 'Casper'},
         {'label': 'patient',
          'page': None,
          'range': [29, 43],
          'score': 1.0,
          'text': 'July 21, 2018.'},
         {'label': 'patient',
          'page': None,
          'range': [60, 90],
          'score': 1.0,
          'text': '07/21/2018 07-21-2018 07-21-18'}]}

        self.assertEqual(expected_dict_out, phi_out.to_json())
