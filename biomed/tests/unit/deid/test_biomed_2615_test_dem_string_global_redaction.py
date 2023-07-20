import json
import unittest

from biomed.deid.global_redaction_helper_functions import parse_demographic_string, get_sensitive_demographic_tokens


class TestDEIDGlobalRedaction(unittest.TestCase):
    def test_demographic_text_splitter(self):
        self.assertEqual(parse_demographic_string('Shannon'), {'shannon'})
        self.assertEqual(parse_demographic_string('Shannon Fee'), {'shannon', 'fee'})
        self.assertEqual(parse_demographic_string('780-213-9378'), {'780-213-9378', '7802139378'})
        self.assertEqual(parse_demographic_string('(402)583-1289'), {'(402)583-1289', '(402)5831289'})
        self.assertEqual(parse_demographic_string('shannon.fee'), {'shannon.fee'})

    def test_getting_sensitive_dem_tokens(self):
        demographics = json.loads(
            """{"ssn": [["235-21-0677", 0.991]], "mrn": [], "sex": [["Male", 1.0]], "dob": [["03/19/1968", 0.999]], 
            "pat_first": [["Emilian", 0.937]], "pat_last": [["Elefteratos", 0.834]], "pat_age": [["49", 1.0]], 
            "pat_street": [], "pat_zip": [], "pat_city": [], "pat_state": [], "pat_phone": [["000-000-0000", 0.219]],
             "pat_email": [], "insurance": [], "facility_name": [["ViSolve", 0.779], ["ViSolve Clinic 1", 0.977], 
             ["ViSolve Clinic", 0.554], ["Clinic", 0.308]], "dr_first": [], "dr_last": [["X", 0.729]],
              "pat_full_name": [["Emilian Elefteratos", 0.8855]], "dr_full_names": [], "race": [], "ethnicity": [],
               "VersionInfo": {"product_id": null, "product_version": "2.09", "tags": [], "active_branch": null, 
               "commit_id": null, "docker_image": null, "commit_date": null, "model_versions": 
               {"demographic": "2.00"}}}""")
        tokens_to_delete = get_sensitive_demographic_tokens(demographics)
        self.assertSetEqual(
            tokens_to_delete, {'000-000-0000', '0000000000','235-21-0677', '235210677','elefteratos','emilian'})

