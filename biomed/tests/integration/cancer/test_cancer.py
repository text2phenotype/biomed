import unittest

from biomed.cancer.cancer import get_oncology_tokens


class TestCancer(unittest.TestCase):
    def test_get_oncology_tokens_None(self):
        self.assertEqual({}, get_oncology_tokens(None))

    def test_get_oncology_tokens_empty_text(self):
        self.assertEqual({}, get_oncology_tokens(""))

    def test_get_oncology_tokens_no_cancer_mention(self):
        self.assertListEqual([], get_oncology_tokens("Patient is healthy and doing great!")['Cancer'])

    def test_get_oncology_tokens_with_cancer_mention(self):
        expected = [{'text': 'metastatic',
                     'range': [34, 44],
                     'label': 'behavior',
                     'polarity': None,
                     'code': '6',
                     'cui': 'C3266877',
                     'tui': 'T080',
                     'vocab': 'SNOMEDCT_US',
                     'preferredText': 'Malignant neoplasms, stated or presumed to be secondary',
                     'page': None},
                    {'text': 'sigmoid colon',
                     'range': [45, 58],
                     'label': 'topography_primary',
                     'polarity': None,
                     'code': '60184004',
                     'cui': 'C0227391',
                     'tui': 'T023',
                     'vocab': 'SNOMEDCT_US',
                     'preferredText': 'C18.7',
                     'page': None},
                    {'text': 'adenocarcinoma',
                     'range': [59, 73],
                     'label': 'morphology',
                     'polarity': None,
                     'code': '443961001',
                     'cui': 'C0001418',
                     'tui': 'T191',
                     'vocab': 'SNOMEDCT_US',
                     'preferredText': '8140/3',
                     'page': None}
                    ]

        cancer_res = get_oncology_tokens("Patient has previous diagnosis of metastatic sigmoid colon adenocarcinoma.")
        tokens = cancer_res['Cancer']
        # ignore scores
        for token in tokens:
            del token['score']

        self.assertListEqual(expected, tokens)
