import unittest

from biomed.models.model_cache import ModelCache
from text2phenotype.tasks.task_enums import TaskOperation

from biomed.common.helpers import annotation_helper
from biomed.drug.drug import meds_and_allergies
from biomed.lab.labs import summary_lab_value

TEXT_CHEMISTRY_STUDY = ['Chemistry', 'CHEMISTRY',
                        'Chemistry Study', 'CHEMISTRY STUDY',
                        'Chemistry Studies', 'CHEMISTRY STUDIES',
                        'study', 'studies',
                        'no display name',
                        '18719-5']

TEXT_ANNOTATION_COMMENT = ['ANNOTATION COMMENT', 'Annotation Comment', '18719-5']


class TestBiomed163(unittest.TestCase):

    def test_lab_cui_blacklist(self):
        cache = ModelCache()
        cui_rule = cache.cui_rule()
        full_text = ""
        for cui, entry in cui_rule.items():
            if 'lab' not in entry['aspect_list'] and entry['aspect_list']:
                full_text += " ".join(entry['txt'])
        tokens, vectors = annotation_helper(full_text, {TaskOperation.drug, TaskOperation.lab})
        lab_res = summary_lab_value(tokens=tokens, vectors=vectors, text=full_text)
        self.assertEqual(0, len(lab_res.get('Lab', list())))

    def test_loinc_blacklist_do_not_summarize(self):
        for text in ['LOINC', 'LNC'] + TEXT_CHEMISTRY_STUDY + TEXT_ANNOTATION_COMMENT:
            tokens, vectors = annotation_helper(text, {TaskOperation.drug, TaskOperation.lab})
            drug_res = meds_and_allergies(tokens=tokens, vectors=vectors, text=text)
            lab_res = summary_lab_value(tokens=tokens, vectors=vectors, text=text)

            self.assertEqual(0, len(lab_res.get('Lab', list())))
            self.assertEqual(0, len(drug_res.get('Allergy', list())))
            self.assertEqual(0, len(drug_res.get('Medication', list())))
