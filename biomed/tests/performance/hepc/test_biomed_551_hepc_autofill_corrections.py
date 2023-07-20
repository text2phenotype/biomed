import unittest

from biomed.summary.text_to_summary import text_to_summary
from feature_service.hep_c.form import autofill_hepc_form

from text2phenotype.common import common

from biomed.tests.samples import RICARDO_HPI_TXT
from biomed.hepc import hepc_filters


class TestBiomed551(unittest.TestCase):

    def test_filter_route_of_infection(self):
        text = common.read_text(RICARDO_HPI_TXT)
        form = autofill_hepc_form(text)
        filtered = hepc_filters.filter_route_of_infection(form)

        for suggest in form['ROUTE_OF_INFECTION']:
            if suggest['suggest'] == 'sexual_partner_has_hepc':
                self.assertIsNotNone(suggest['evidence'])
            else:
                self.assertIsNone(suggest['evidence'])

        for suggest in filtered['ROUTE_OF_INFECTION']:
            self.assertIsNone(suggest['evidence'])

    def test_filter_problem_list(self):
        text = common.read_text(RICARDO_HPI_TXT)
        form = autofill_hepc_form(text)

        summary = text_to_summary(text)

        problem_list_text = [problem['text'] for problem in summary['DiseaseDisorder']]

        # tricky case:
        # Form is not filled by ctakes, which matches HCV antibody test (lab, not problem)
        # HCV is not yet diagnosed
        #
        hcv = [suggest['evidence'] for suggest in form['HCV'] if suggest['evidence']]
        self.assertGreater(len(hcv), 0)

        # clinical summary knows HCV is not yet diagnosed
        for problem in problem_list_text:
            self.assertFalse('HCV' in problem)

        filtered = hepc_filters.filter_problem_list(form, summary)

        # do not filter other diagnosis
        self.assertTrue(form['CIRRHOSIS'], filtered['CIRRHOSIS'])
        self.assertTrue(form['PSYCHIATRIC_DIAGNOSES'], filtered['PSYCHIATRIC_DIAGNOSES'])
        self.assertTrue(form['DIAGNOSIS_OTHER'], filtered['DIAGNOSIS_OTHER'])
        self.assertTrue(form['DIAGNOSIS_LIVER_HISTORY'], filtered['DIAGNOSIS_LIVER_HISTORY'])

        # PROCEDURES : do not change
        self.assertTrue(form['DIAGNOSTIC_PROCEDURES'], filtered['DIAGNOSTIC_PROCEDURES'])

        # LABS : do not change, especially BMI
        self.assertTrue(form['BMI'], filtered['BMI'])
