import unittest

from biomed.deid.utils import demographics_chunk_to_deid
from biomed.common.biomed_summary import FullSummaryResponse
from biomed.deid.global_redaction_helper_functions import redact_text
from biomed.demographic.demographic import get_demographic_tokens
from text2phenotype.constants.features import DemographicEncounterLabel, PHILabel
from text2phenotype.tasks.task_enums import TaskOperation

from biomed.deid import deid
from biomed.common.helpers import annotation_helper


class TestBiomed893(unittest.TestCase):
    def test_redact_txt(self):
        text = """
        Continuity of Care Document 06/16/2017 -- Campos,Ricardo
        Patient	Ricardo Campos
        Date of birth	January 2, 1963
        Sex	Male
        Race	Other Race
        Ethnicity	Hispanic or Latino
        ******* info	Primary Home:
        123 MONTGOMERY BLVD NE
        Las Cruces, NM 88007, US
        Tel: (575) 123-4567
        Patient IDs:	123456789 3.4.1.5.6.172943.1234
        Document Id:	280004 14.5.7912.236.12.2.2
        Document Created:	July 16, 2017, 11:44:46, EST
        Performer (primary care physician)	Miranda Bailey, MD
        Author	Millennium Clinical Document Generator
        ******* info	
        """
        tokens, vectors = annotation_helper(text, {TaskOperation.phi_tokens, TaskOperation.demographics})
        phi_tokens = deid.get_phi_tokens(tokens, vectors=vectors)['PHI']
        dem_tokens = get_demographic_tokens(
            tokens,
            vectors=vectors)[DemographicEncounterLabel.get_category_label().persistent_label]
        phi_tokens.extend(
            demographics_chunk_to_deid(
                dem_tokens
            ).to_json()[PHILabel.get_category_label().persistent_label])

        phi_resp = FullSummaryResponse.from_json({'PHI': phi_tokens})
        out_txt = redact_text(phi_resp, text)

        expected_redaction_text = """
        Continuity of Care Document ********** -- ******,*******
        Patient	******* ******
        Date of birth	******* *, ****
        ***	Male
        Race	Other ****
        Ethnicity	Hispanic or ******
        ******* info	Primary Home:
        *** ********** **** **
        *** ******, ** *****, US
        Tel: (***) ********
        Patient IDs:	********* *********************
        Document Id:	****** ********************
        Document Created:	**** **, ****, 11:44:46, EST
        Performer (primary care physician)	******* ******, MD
        ******	********** Clinical Document Generator
        ******* info	
        """
        self.assertEqual(expected_redaction_text, out_txt)

    def biomed_1030_sample_text(self):
        text = "John Huntington \nHome Address: 123 Huntington Street \nPast Medical History: Notable for Huntington’s disease beginning age 12. \n Home Phone: (555) 867 - 5309."
        tokens, vectors = annotation_helper(text, {TaskOperation.phi_tokens})
        phi_tokens = deid.get_phi_tokens(tokens, vectors=vectors)

        out_txt = redact_text(FullSummaryResponse.from_json(phi_tokens), text)
        expected_redaction_text = "Patient: **** ********** \nHome Address: *** ********** ****** \nPast Medical History: Notable for Huntington’s disease beginning age 12. \n Home Phone: (***) *** - ****."

        self.assertEqual(expected_redaction_text, out_txt)