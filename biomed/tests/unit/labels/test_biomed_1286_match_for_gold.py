import os
import unittest

from text2phenotype.common.feature_data_parsing import is_digit_punctuation
from text2phenotype.constants.features import MedLabel, LabLabel, DuplicateDocumentLabel

from biomed.data_sources.data_source import BiomedDataSource


class TestMatchForGold(unittest.TestCase):
    ANN_FP = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'fixtures/match_for_gold_test.ann')
    def test_from_brat_med(self):
        med_result = BiomedDataSource.get_brat_label(self.ANN_FP, MedLabel)
        self.assertEqual(len(med_result),  6)
        for i in med_result:
            self.assertEqual(i.label, MedLabel.med.value.persistent_label)
        tokens_text = ['Atenolol', 'Lisinopril', 'Omeprazole', 'Atenolol', 'Lisinopril', 'Omeprazole']
        token_ranges = [[0, 8], [9, 19], [20, 30], [31, 39], [40, 50], [51, 61]]

        vectors = BiomedDataSource.match_for_gold(token_ranges=token_ranges, token_text_list=tokens_text,
                                                  brat_res=med_result, label_enum=MedLabel)
        for i in range(6):
            self.assertEqual(vectors[i], [0, 1])

    def test_from_brat_lab(self):
        lab_result = BiomedDataSource.get_brat_label(self.ANN_FP, LabLabel, parse_labs=True)
        actual_readable_results = {i.text: i.label for i in lab_result}
        expected_readable_results = {'pCO2': 'lab', 'mm': 'lab_unit', 'elevation': 'lab_interp', 'ALT': 'lab',
                                     'elevated': 'lab_interp', 'bilirubin': 'lab', 'White': 'lab', 'blood': 'lab',
                                     'cell': 'lab', 'count': 'lab', 'glucose': 'lab', '58mg': 'lab_unit', '200s': 'lab_value',
                                     'DL': 'lab', 'RBC': 'lab', 'mm3': 'lab_unit', 'PT': 'lab', 'PTT': 'lab',
                                     'INR': 'lab', 'AST': 'lab', 'Alk': 'lab', 'Phos': 'lab', 'T': 'lab', 'Bili': 'lab',
                                     'Differential-Neuts': 'lab', 'Lymph': 'lab', 'Mono': 'lab', 'Eos': 'lab',
                                     'Lactic':'lab', 'Acid': 'lab', 'mmol': 'lab_unit', 'L': 'lab_unit', 'LDH': 'lab',
                                     'IU': 'lab_unit', 'Ca++': 'lab', 'mg': 'lab_unit', 'dL': 'lab_unit', 'Mg++': 'lab',
                                     'PO4': 'lab', 'Ca': 'lab', 'Mg-2': 'lab', 'CK': 'lab', 'MB': 'lab',
                                     'not': 'lab_interp', 'done': 'lab_interp', 'TnT': 'lab', 'baseline': 'lab_interp',
                                     'free': 'lab', 'T4': 'lab', 'pO2-34*': 'lab', 'pCO2-28*': 'lab', 'pH-7':'lab',
                                     'calHCO3-14*': 'lab', 'Base': 'lab_interp', 'XS--12': 'lab', 'FUNGAL': 'lab',
                                     'CULTURE': 'lab', 'PLEURAL': 'lab', 'TotProt-3': 'lab', 'Glucose-1': 'lab',
                                     'LD': 'lab', 'GLUCOSE-161*': 'lab', 'LACTATE-2': 'lab', 'NA+-140': 'lab',
                                     'K+-4': 'lab', 'CL--110': 'lab', 'TCO2-20': 'lab', 'HYPOCHROM-NORMAL': 'lab',
                                     'ANISOCYT-NORMAL': 'lab', 'POIKILOCY-NORMAL': 'lab', 'MACROCYT-NORMAL': 'lab',
                                     'MICROCYT-NORMAL': 'lab', 'POLYCHROM-NORMAL': 'lab', 'LDL': 'lab'}

        self.assertEqual(len(actual_readable_results), 106)
        for i in actual_readable_results:
            if is_digit_punctuation(i):
                self.assertEqual(actual_readable_results[i], LabLabel.lab_value.name)
            else:
                self.assertIn(i, expected_readable_results)
                self.assertEqual(actual_readable_results[i], expected_readable_results[i], i)

    def test_duplicate_from_brat(self):
        self.assertEqual(DuplicateDocumentLabel.duplicate,
                         DuplicateDocumentLabel.from_brat(DuplicateDocumentLabel.duplicate.value.persistent_label))

    def test_duplicate_indexes(self):
        duplicate_result = BiomedDataSource.get_brat_label(self.ANN_FP, DuplicateDocumentLabel)
        self.assertEqual(len(duplicate_result), 1)



