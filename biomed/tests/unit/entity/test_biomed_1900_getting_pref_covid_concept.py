import unittest

from biomed.common.helpers import get_prefered_covid_concept
from text2phenotype.common.featureset_annotations import MachineAnnotation


class TestBiomedGetPrefCOvidConcept(unittest.TestCase):
    def test_without_tokens(self):
        tokens = MachineAnnotation(json_dict_input={'tokens': ['a', 'bug']})
        c, p = get_prefered_covid_concept(tokens, 1)
        self.assertIsNone(c)
        self.assertIsNone(p)

    def test_with_tokens(self):
        tokens = MachineAnnotation(
            json_dict_input={'tokens': ['a', 'bug'],
                             'covid_representation': {1: [
                                 {
                                     "Entity": [
                                         {
                                             "code": "b",
                                             "cui": "a",
                                             "tui": "c",
                                             "tty": None,
                                             "preferredText": "bug",
                                             "codingScheme": "hello"
                                         },
                                     ],
                                     "polarity": "positive"
                                 }
                             ]}})
        c, p = get_prefered_covid_concept(tokens, 1)
        self.assertDictEqual(c, {
            "code": "b",
            "cui": "a",
            "tui": "c",
            "tty": None,
            "preferredText": "bug",
            "codingScheme": "hello"
        })
        self.assertEqual(p, 'positive')
