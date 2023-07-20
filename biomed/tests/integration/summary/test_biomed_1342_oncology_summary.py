import unittest

from biomed.common.helpers import annotation_helper
from text2phenotype.constants.features.label_types import CancerLabel
from biomed.cancer.cancer import get_oncology_tokens
from biomed.diagnosis.diagnosis import diagnosis_sign_symptoms
from biomed.summary.text_to_summary import summary_from_parts
from text2phenotype.tasks.task_enums import TaskOperation


class TestBiomed1342(unittest.TestCase):
    INPUT_TEXT = ' Patient has previous diagnosis of sigmoid colon adenocarcinoma.'
    EXPECTED_CATEGORES = [CancerLabel.get_category_label().persistent_label,
                          'VersionInfo']

    def test_oncology_empty_text(self):
        test_input = ''
        tokens, vectors = annotation_helper(test_input, {TaskOperation.oncology_only})
        res = get_oncology_tokens(tokens=tokens, vectors=vectors, text=test_input)
        self.assertEqual(set(), set(res.keys()))

    def test_oncology_summary_nonsense_text(self):
        test_input = 'sdkhpfoweiulksd [uywe759-8reuj fnb93-0 #$^%@&#*P^()&*%4-089G -9T7 '
        tokens, vectors = annotation_helper(test_input, {TaskOperation.oncology_only})
        res = get_oncology_tokens(tokens=tokens, vectors=vectors, text=test_input)
        self.assertEqual(set(self.EXPECTED_CATEGORES), set(res.keys()))

    def test_oncology_summary_cancer_text(self):
        test_input = ' Patient has previous diagnosis of sigmoid colon adenocarcinoma.'
        tokens, vectors = annotation_helper(test_input, {TaskOperation.oncology_only,
                                                         TaskOperation.disease_sign})
        cancer_res = get_oncology_tokens(tokens=tokens, vectors=vectors, text=test_input)
        problem_res = diagnosis_sign_symptoms(tokens=tokens, vectors=vectors, text=test_input)

        res = summary_from_parts([cancer_res, problem_res], text=test_input)
        #signmoid colon recognized
        sigmoid_colon = False
        if 'sigmoid colon' in res['Cancer'][0]['text']:
            sigmoid_colon = True
        elif ('sigmoid' in res['Cancer'][0]['text'] and
              len(res['DiseaseDisorder']) > 0 and 'colon' in res['DiseaseDisorder'][0]['text']):
            sigmoid_colon = True

        self.assertTrue(sigmoid_colon)
        self.assertEqual(res['Cancer'][0]['label'], 'topography_primary')
