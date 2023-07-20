import unittest

from text2phenotype.common.featureset_annotations import MachineAnnotation
from text2phenotype.constants.features.feature_type import FeatureType

from biomed.common.helpers import get_pref_umls_concept_polarity, get_by_pref_tty, get_first, get_longest_pref_text


class TestBiomed1894GetConcept(unittest.TestCase):
    INPUT_ANNOTATIONS = {
        'clinical': {
            '12': [{'Lab': [
                {'code': '15220000', 'codingScheme': 'SNOMEDCT_US', 'cui': 'C0022885',
                 'preferredText': 'Laboratory Procedures', 'tty': None, 'tui': 'T059'}], 'polarity': 'positive'},
                {'SignSymptom': [
                    {'code': '277775005', 'codingScheme': 'SNOMEDCT_US', 'cui': 'C0456984',
                     'preferredText': 'Test Result', 'tty': None, 'tui': 'T033'},
                    {'code': '398447004', 'codingScheme': 'SNOMEDCT_US', 'cui': 'C1175175',
                     'preferredText': 'Severe Acute Respiratory Syndrome', 'tty': None, 'tui': 'T047'}],
                    'polarity': 'positive'}]},
        'drug_rxnorm': {
            '40': [{'Medication': [
                {'code': '1001007', 'codingScheme': 'RXNORM', 'cui': 'C2926735', 'preferredText': 'Duration',
                 'tty': 'BN', 'tui': 'T109'},
                {'code': '1001007', 'codingScheme': 'RXNORM', 'cui': 'C2926735', 'preferredText': 'Duration',
                 'tty': 'BN', 'tui': 'T121'}],
                'attributes': {'medDosage': None, 'medDuration': None, 'medForm': None,
                               'medFrequencyNumber': [], 'medFrequencyUnit': [], 'medRoute': None,
                               'medStatusChange': 'noChange', 'medStrengthNum': [], 'medStrengthUnit': [],
                               'polarity': 'positive'}}]}

    }
    MACHINE_ANNOTATION = MachineAnnotation(json_dict_input=INPUT_ANNOTATIONS)

    def test_get_prefered_concept_first_from_clinical_type(self):
        annot = self.MACHINE_ANNOTATION[FeatureType.clinical, 12]
        self.assert_same_get_first(annot, ['Lab'],
                                   {'code': '15220000', 'codingScheme': 'SNOMEDCT_US', 'cui': 'C0022885',
                                    'preferredText': 'Laboratory Procedures', 'tty': None, 'tui': 'T059'})
        self.assert_same_get_first(annot, ['SignSymptom'],
                                   {'code': '277775005', 'codingScheme': 'SNOMEDCT_US', 'cui': 'C0456984',
                                    'preferredText': 'Test Result', 'tty': None, 'tui': 'T033'})

        self.assert_same_get_first(annot, ['asldjfas'], None, expected_pol=None)
        annot = self.MACHINE_ANNOTATION[FeatureType.drug_rxnorm, 40]

        self.assert_same_get_first(annot, ['Medication'],
                                   {'code': '1001007', 'codingScheme': 'RXNORM', 'cui': 'C2926735',
                                    'preferredText': 'Duration', 'tty': 'BN', 'tui': 'T109'},
                                   expected_pol=None,
                                   exprected_attrib={'medDosage': None, 'medDuration': None, 'medForm': None,
                                                     'medFrequencyNumber': [], 'medFrequencyUnit': [], 'medRoute': None,
                                                     'medStatusChange': 'noChange', 'medStrengthNum': [],
                                                     'medStrengthUnit': [],
                                                     'polarity': 'positive'})
        self.assert_same_get_first(None, ['Medication'], None, None, None)

    def assert_same_get_first(self, annot, sem_type, expected_umls, expected_pol='positive', exprected_attrib=None):
        umls, polarity, attributes = get_pref_umls_concept_polarity(annot, sem_type, get_first)
        self.assertEqual(umls, expected_umls)
        self.assertEqual(polarity, expected_pol)
        self.assertEqual(attributes, exprected_attrib)
