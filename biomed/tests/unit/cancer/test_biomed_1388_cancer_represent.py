import unittest
import os

from text2phenotype.common import common
from text2phenotype.common.feature_data_parsing import from_bag_of_words
from text2phenotype.annotations.file_helpers import AnnotationSet

from biomed.cancer.cancer_represent import Qualifier
from biomed.cancer.cancer_represent import Behavior, Grade
from biomed.cancer.cancer_represent import parse_behavior, parse_grade
from biomed.cancer.cancer_represent import select_pref_snomed

class TestBiomed1388(unittest.TestCase):

    def test_behavior_qualifier(self):
        expect = {"code": '6',
                   "cui": 'C3266877',
                   "label": "behavior",
                   "score": 0.86,
                   "preferredText": 'Malignant neoplasms, stated or presumed to be secondary',
                   "range": [3405,3415],
                  'polarity': 'positive',
                   "text": "Metastatic",
                   "tui": ['T080'],
                   "codingScheme": 'SNOMEDCT_US'}

        mention = {"code": None,
                   "cui": None,
                   "label": "behavior",
                   "score": 0.86,
                   "preferredText": None,
                   "range": [3405,3415],
                   "text": "Metastatic",
                   "tui": None,
                   "codingScheme": None}

        text = mention.get('text')

        actual = Qualifier.represent(mention, parse_behavior(text))

        self.assertEqual(expect, actual)

    def test_parse_behavior(self):
        self.assertEqual(Behavior.B0_benign, parse_behavior('benign'))

        self.assertEqual(Behavior.B1_uncertain, parse_behavior('uncertain'))
        self.assertEqual(Behavior.B1_uncertain, parse_behavior('indeterminate'))
        self.assertEqual(Behavior.B1_uncertain, parse_behavior('unknown'))
        self.assertEqual(Behavior.B1_uncertain, parse_behavior('NOS'))

        self.assertEqual(Behavior.B2_in_situ, parse_behavior('in situ'))

        self.assertEqual(Behavior.B3_malignant_primary, parse_behavior('malignant'))
        self.assertEqual(Behavior.B3_malignant_primary, parse_behavior('invasive'))
        self.assertEqual(Behavior.B3_malignant_primary, parse_behavior('invades'))
        self.assertEqual(Behavior.B3_malignant_primary, parse_behavior('Infiltrating'))

        self.assertEqual(Behavior.B6_malignant_secondary, parse_behavior('metastatic'))
        self.assertEqual(Behavior.B6_malignant_secondary, parse_behavior('metastases'))

    def test_parse_grade(self):
        self.assertEqual(Grade.G1_well, parse_grade('G1'))
        self.assertEqual(Grade.G1_well, parse_grade('well differentiated'))
        self.assertEqual(Grade.G1_well, parse_grade('low grade'))
        self.assertEqual(Grade.G1_well, parse_grade('LG'))

        self.assertEqual(Grade.G2_moderate, parse_grade('G2'))
        self.assertEqual(Grade.G2_moderate, parse_grade('II'))
        self.assertEqual(Grade.G2_moderate, parse_grade('G II'))
        self.assertEqual(Grade.G2_moderate, parse_grade('Grade 2'))
        self.assertEqual(Grade.G2_moderate, parse_grade('Intermediate'))
        self.assertEqual(Grade.G2_moderate, parse_grade('Moderate'))

        self.assertEqual(Grade.G3_poor, parse_grade('G3'))
        self.assertEqual(Grade.G3_poor, parse_grade('Grade 3'))
        self.assertEqual(Grade.G3_poor, parse_grade('GIII'))
        self.assertEqual(Grade.G3_poor, parse_grade('high grade'))
        self.assertEqual(Grade.G3_poor, parse_grade('HIGH GRADE'))
        self.assertEqual(Grade.G3_poor, parse_grade('high-grade'))
        self.assertEqual(Grade.G3_poor, parse_grade('poorly differentiated'))
        self.assertEqual(Grade.G3_poor, parse_grade('poorly-differentiated'))
        self.assertEqual(Grade.G3_poor, parse_grade('Biopsy shows poorly differentiated mass'))
        self.assertEqual(Grade.G3_poor, parse_grade('Group 3'))
        self.assertEqual(Grade.G3_poor, parse_grade('HG'))

        self.assertEqual(Grade.G4_undifferentiated, parse_grade('G4'))
        self.assertEqual(Grade.G4_undifferentiated, parse_grade('Grade IV'))
        self.assertEqual(Grade.G4_undifferentiated, parse_grade('undifferentiated'))
        self.assertEqual(Grade.G4_undifferentiated, parse_grade('anaplastic'))

        unknown = ['Grade 5', 'grade', 'differentiation', 'differentiated', 'PNO', 'Nottingham', 'Nottingham I', 'Nib']
        for actual in unknown:
            self.assertEqual(Grade.G9_unknown, parse_grade(actual))

    def test_select_pref_snomed(self):
        lnc   = {"code": "MTHU008683", "tui": "T023", "tty": "LS", "codingScheme": "LNC", "cui": "C0024109", "preferredText": "C34.9"}
        nci   = {"code": "TCGA", "tui": "T023", "tty": "SY", "codingScheme": "NCI", "cui": "C0024109", "preferredText": "C34.9"}
        rcd   = {"code": "XM0PM", "tui": "T023", "tty": "PT", "codingScheme": "RCD", "cui": "C0024109", "preferredText": "C34.9"}
        sno_pt= {"code": "181216001", "tui": "T023", "tty": "PT", "codingScheme": "SNOMEDCT_US", "cui": "C1278908", "preferredText": "C34.9"}
        sno_fn= {"code": "181216001", "tui": "T023", "tty": "FN", "codingScheme": "SNOMEDCT_US", "cui": "C1278908", "preferredText": "C34.9"}
        sno_is= {"code": "39607008", "tui": "T023", "tty": "IS", "codingScheme": "SNOMEDCT_US", "cui": "C0024109", "preferredText": "C34.9"}

        actual = select_pref_snomed([lnc, nci, rcd, sno_fn, sno_pt, sno_is])

        self.assertEqual(sno_pt, actual)

        actual = select_pref_snomed([lnc, nci, rcd, sno_fn, sno_is])

        self.assertEqual(sno_fn, actual)

    @unittest.skip
    def get_corpus(self, annotation_dir_root='/mnt/s3/BIOMED-1000-cancer'):
        """
        (optional) re-process corpus
        aws s3 sync s3://biomed-data/despina.siolas/BIOMED-1000-cancer BIOMED-1000-cancer
        """
        corpus = list()
        for root, dirs, files in os.walk(annotation_dir_root, topdown=False):
            for name in dirs:
                for f in common.get_file_list(os.path.join(root, name), '.ann'):
                    corpus.append(f)
        return corpus

    @unittest.skip
    def test_get_corpus_term_frequency(self):
        tf_topography = list()
        tf_morphology = list()
        tf_behavior = list()
        tf_grade = list()
        tf_stage = list()

        for f in self.get_corpus():
            text = common.read_text(f)
            for _, annot in AnnotationSet().from_file_content(text):

                if 'topography' in annot.aspect:
                    tf_topography.append(annot.text)

                if 'morphology' in annot.aspect:
                    tf_morphology.append(annot.text)

                if str(annot.aspect).startswith('b'):
                    tf_behavior.append(annot.text)

                if 'grade' == annot.aspect:
                    tf_grade.append(annot.text)

                if 'stage' == annot.aspect:
                    tf_stage.append(annot.text)

        common.write_json(from_bag_of_words(tf_topography), './test_biomed_1388_cancer_represent.topography.json')
        common.write_json(from_bag_of_words(tf_morphology), './test_biomed_1388_cancer_represent.morphology.json')
        common.write_json(from_bag_of_words(tf_stage), './test_biomed_1388_cancer_represent.stage.json')
        common.write_json(from_bag_of_words(tf_grade), './test_biomed_1388_cancer_represent.grade.json')
        common.write_json(from_bag_of_words(tf_behavior), './test_biomed_1388_cancer_represent.behavior.json')


if __name__ == '__main__':
    unittest.main()
