import unittest

from biomed.reassembler.reassemble_functions import reassemble_demographics
from text2phenotype.common.dates import parse_dates
from text2phenotype.common.demographics import Demographics
from text2phenotype.constants.features import DemographicEncounterLabel
from text2phenotype.tasks.task_enums import TaskOperation

from biomed.common.helpers import annotation_helper
from biomed.demographic.demographic import get_demographic_tokens


class TestBiomed1038(unittest.TestCase):
    def test_Shannon_Fee_Nacors_format(self):
        text = "'Beverly Hills Institute / Alex Foxman, M.D., F.A.C.P.\n9400 Brighton Way, Suite 410 Beverly Hills, CA 90210\n(310) 274-0657 Fax: (310) 274-6083\nApril 8, 2019\nPage 1\nChart Summary\nName: Shannon Casper Fee\nFemale DOB: 07/27/1996\nHome: (650) 388-2901 Work: (650) 326-9515\nIns: Aetna (Box 18923)\n10005\nHome Phone: (650) 388-6538\nWork Phone: (650) 388-2901\nPatient Information\nName: Ms. Shannon Fee\nAddress: 1155 Jones St Apt 401\nSan Francisco, CA 94109\nPatient ID: 87261\nBirth Date: 07/27/1996\nGender: Female\nContact By: Email\nSoc Sec No: 606-87-1827\nResp Prov:\nReferred by: Alex MD Foxman\nEmail:\nHome LOC: Beverly Hills Institute / Alex Foxman,\nM.D., F.A.C.P.\nFax:\nStatus: Active\nMarital Status: Single\nRace: White\nLanguage: English\nMRN:\nEmp. Status:\nSens Chart: No\nExternal ID:\nProblems\nHIP PAIN, RIGHT (ICD-719.45) (ICD10-M25.551)\nHIP REPLACEMENT, RIGHT, HX OF (ICD-143.64) (ICD10-296.641)\nProcedures\nMedications\nImmunizations\nDirectives\nAllergies and Adverse Reactions\nServices Due\nReport run by Lance Roberts\n'"
        tokens, vectors = annotation_helper(text, {TaskOperation.demographics})
        date_matches = parse_dates(text)
        demographic_tokens = get_demographic_tokens(tokens=tokens, vectors=vectors, date_matches=date_matches)
        demographic_out = reassemble_demographics([((0, len(text)), demographic_tokens)])
        expected_demographics_dict = {
            'ssn': [('606-87-1827', 1.0)],
            'mrn': [('10005', 0.965), ('87261', 0.909)],
            'sex': [('Female', 1.0)],
            'dob': [('07/27/1996', 1.0)],
            'pat_first': [('Shannon', 1.0)],
            'pat_last': [('Fee', 0.992)],
            'pat_age': [],
            'pat_street': [(',', 0.489), ('1155 Jones St Apt 401', 1.0)],
            'pat_zip': [('94109', 1.0)],
            'pat_city': [('Hills', 1.0), ('San Francisco', 1.0)],
            'pat_state': [('CA', 1.0)],
            'pat_phone': [],
            'pat_email': [],
            'insurance': [],
            'facility_name': [("'Beverly Hills Institute /", 0.999),
                              ('Beverly Hills Institute /', 0.999)],
            'dr_first': [('Alex', 0.996), ('Lance', 0.333)],
            'dr_last': [('Foxman', 1.0), ('Roberts', 0.967)],
            'pat_full_name': [('Shannon Casper Fee', 0.9283333333333333)],
            'dr_full_names': [],
            'race': [('WHITE', 1.0)],
            'ethnicity': []
        }

        for k, v in expected_demographics_dict.items():
            if v:
                self.assertEqual(v[0][0].lower(), demographic_out[k][0][0].lower())
