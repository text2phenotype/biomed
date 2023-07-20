import unittest

from biomed.common.biomed_summary import FullSummaryResponse
from biomed.diagnosis.remove_family_history import remove_family_history_from_disease


class TestBiomed2458FamilyHistoryFilter(unittest.TestCase):
    diagnosis_response = {
        "DiseaseDisorder":
            [{"text": "Seasonal Allergies", "range": [6617, 6635], "score": 0.9995100312556253,
              "label": "diagnosis", "page": 4, "polarity": "positive", "code": "21719001", "cui": "C0018621",
              "tui": "T047", "vocab": "SNOMEDCT_US", "preferredText": "Hay fever"},
             {"text": "Elevated Cholesterol", "range": [6637, 6657], "score": 0.9994648192983808,
              "label": "diagnosis", "page": 4, "polarity": "positive", "code": "13644009", "cui": "C0020443",
              "tui": "T047", "vocab": "SNOMEDCT_US", "preferredText": "Hypercholesterolemia"},
             {"text": "GERD", "range": [6659, 6663], "score": 0.9996576409866197, "label": "diagnosis", "page": 4,
              "polarity": "positive", "code": "235595009", "cui": "C0017168", "tui": "T047", "vocab": "SNOMEDCT_US",
              "preferredText": "Gastroesophageal reflux disease"},
             {"text": "Arthritis", "range": [6665, 6674], "score": 0.9996547516947885, "label": "diagnosis", "page": 4,
              "polarity": "positive", "code": "3723001", "cui": "C0003864", "tui": "T047", "vocab": "SNOMEDCT_US",
              "preferredText": "Arthritis"},
             {"text": "Hyperthyroidism", "range": [6675, 6690],
              "score": 0.9996415938439156, "label": "diagnosis", "page": 4,
              "polarity": "positive", "code": "34486009", "cui": "C0020550",
              "tui": "T047", "vocab": "SNOMEDCT_US", "preferredText": "Hyperthyroidism"},
             {"text": "Depression", "range": [6692, 6702], "score": 0.9996123570282837, "label": "diagnosis", "page": 4,
              "polarity": "positive", "code": "35489007", "cui": "C0011581", "tui": "T048", "vocab": "SNOMEDCT_US",
              "preferredText": "Depressive disorder"},
             {"text": "Anxiety", "range": [6704, 6711], "score": 0.9995196983216981, "label": "diagnosis", "page": 4,
              "polarity": "positive", "code": "48694002", "cui": "C0003467", "tui": "T048", "vocab": "SNOMEDCT_US",
              "preferredText": "Anxiety"},
             {"text": "High", "range": [6730, 6734], "score": 0.6477983965911962, "label": "diagnosis", "page": 4,
              "polarity": "positive", "code": "38341003", "cui": "C0020538", "tui": "T047", "vocab": "SNOMEDCT_US",
              "preferredText": "Hypertensive disease"}],
        "SignSymptom":
            [{"text": "knee pain", "range": [7274, 7283], "score": 0.9881266077241765, "label": "signsymptom",
              "page": 4, "polarity": "positive", "code": "30989003", "cui": "C0231749", "tui": "T033",
              "vocab": "SNOMEDCT_US", "preferredText": "Knee pain"},
             {"text": "leg pain", "range": [7295, 7303], "score": 0.9963968633044726, "label": "signsymptom",
              "page": 4, "polarity": "positive", "code": "10601006", "cui": "C0023222", "tui": "T184",
              "vocab": "SNOMEDCT_US", "preferredText": "Pain in lower limb"},
             {"text": "phantom pain", "range": [12152, 12164], "score": 0.6880424072759529, "label": "signsymptom",
              "page": 6, "polarity": "positive", "code": "710110008", "cui": "C3495442", "tui": "T047",
              "vocab": "SNOMEDCT_US", "preferredText": "Phantom pain"}],
        "VersionInfo": [{"product_id": None, "product_version": "2.07", "tags": [], "active_branch": None,
                         "commit_id": None, "docker_image": None, "commit_date": None,
                         "model_versions": {"diagnosis": "2.01"}}]}

    family_history_response = {
        "Family_History":
            [
                {"text": "Elevated Cholesterol", "range": [6637, 6657], "score": 0.8,
                 "label": "family_history", "page": 4, "polarity": "positive", "code": "13644009", "cui": "C0020443",
                 "tui": "T047", "vocab": "SNOMEDCT_US", "preferredText": "Hypercholesterolemia"},
                {"text": "GERD", "range": [6659, 6663], "score": 0.9996576409866197, "label": "diagnosis", "page": 4,
                 "polarity": "positive", "code": "235595009", "cui": "C0017168", "tui": "T047", "vocab": "SNOMEDCT_US",
                 "preferredText": "Gastroesophageal reflux disease"},
                {"text": "Hyperthyroidism", "range": [6675, 6690], "score": 0.9996415938439156, "label": "family_history",
                 "page": 4, "polarity": "positive", "code": "34486009", "cui": "C0020550",
                 "tui": "T047", "vocab": "SNOMEDCT_US", "preferredText": "Hyperthyroidism"},
                {"text": "knee pain", "range": [7274, 7283], "score": 0.9881266077241765, "label": "family_history",
                 "page": 4, "polarity": "positive", "code": "30989003", "cui": "C0231749", "tui": "T033",
                 "vocab": "SNOMEDCT_US", "preferredText": "Knee pain"},
            ]}

    def test_family_history_remove_from_diagnosis(self):
        diagnosis_resp = FullSummaryResponse.from_json(self.diagnosis_response)
        family_hist_resp = FullSummaryResponse.from_json(self.family_history_response)

        new_diag_resp = remove_family_history_from_disease(diagnosis_resp, family_hist_resp)

        self.assertEqual(len(new_diag_resp['SignSymptom'].response_list), len(self.diagnosis_response['SignSymptom']) -1 )
        self.assertEqual(len(new_diag_resp['DiseaseDisorder'].response_list),
                         len(self.diagnosis_response['DiseaseDisorder']) - 3)