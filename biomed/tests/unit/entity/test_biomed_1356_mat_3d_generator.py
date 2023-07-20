import unittest
from math import ceil

from text2phenotype.common.featureset_annotations import MachineAnnotation, Vectorization

from text2phenotype.constants.features.feature_type import FeatureType

from biomed.biomed_env import BiomedEnv
from biomed.common.matrix_construction import get_3d_matrix
from biomed.common.mat_3d_generator import Mat3dGenerator
from biomed.meta.ensembler import Ensembler
from biomed.constants.constants import  get_version_model_folders
from biomed.models.model_metadata import (
    ModelMetadata,
    ModelType,
)


class TestBiomed1356(unittest.TestCase):
    feature_types = [FeatureType.header]
    window_size = 5
    num_tokens = 30
    vectors = Vectorization(
        json_input_dict={'person': {}, 'morphology_code_regex': {}, 'tnm_code': {}, 'clinical_medgen': {},
                         'latinizer': {}, 'allergy_regex': {}, 'aspect_line': {}, 'pathology_report': {},
                         'aspect_enforce': {}, 'tumor_grade_code': {},
                         'tumor_grade_terms': {},
                         'header': {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
                                    '10': 10,
                                    '11': 11, '12': 12, '13': 13, '14': 14, '15': 15, '16': 16, '17': 17, '18': 18,
                                    '19': 19,
                                    '20': 20, '21': 21, '22': 22, '23': 23, '24': 24, '25': 25, '26': 26, '27': 27,
                                    '28': 28,
                                    '29': 29}, 'drug_rxnorm': {}, 'lab_hepc': {}, 'clinical': {},
                         'aspect': {},
                         'speech_bin': {}, 'header_aspect': {}, 'polarity': {}, 'severity': {}, 'zipcode': {},
                         'sectionizer': {},
                         'history': {}, 'len': {}, 'speech': {}, 'tf_cities': {}, 'form': {},
                         'tf_ccda': {},
                         'tf_states': {},
                         'tf_i2b2': {}, 'tf_mtsample': {}, 'tf_mrconso': {}, 'tf_npi_city': {}, 'tf_npi_address': {},
                         'tf_npi_phone': {}, 'tf_npi_first_name': {}, 'tf_npi_last_name': {},
                         'tf_patient_first_name': {},
                         'tf_patient_last_name': {}, 'tf_finding': {}, 'case': {}, 'tf_procedure': {}, 'smoking': {},
                         'regex_dates': {}, 'clinical_code_icd9': {}, 'problem': {}, 'tf_disorder': {},
                         'clinical_code_icd10': {},
                         'clinical_general': {}, 'clinical_snomed': {}, 'date_comprehension': {},
                         'diagnosis': {},
                         'word2vec_mimic': {}, 'topography': {}, 'lab_loinc': {}, 'topography_code': {},
                         'morphology': {}, 'morphology_code': {}, 'topography_code_regex': {}, 'defaults': {
                'clinical_code_icd10': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'clinical_general': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'clinical_snomed': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'regex_dates': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'status': [0, 0, 0], 'date_comprehension': [0, 0, 0], 'family_history': [0, 0, 0], 'problem': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'tumor_grade_code': [0, 0], 'tf_npi_address': [0], 'tumor_grade_terms': [0, 0, 0, 0], 'diagnosis': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'pathology_report': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'pathology_quickpicks': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'lab_loinc': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'lab_hepc_attributes': [0, 0], 'transfer': [0], 'smoking_keywords': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'imaging_regex': [0, 0, 0, 0, 0, 0, 0, 0], 'smoking_regex': [0, 0, 0, 0, 0, 0, 0], 'laterality': [0], 'word2vec_mimic': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'topography': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'lab_unit_probable': [0], 'clinical': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'lab_value_phrases': [0, 0, 0, 0, 0, 0], 'topography_code': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'clinical_snomed_binary': [0, 0], 'morphology': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'clinical_general_binary': [0, 0], 'morphology_code': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'clinical_code_icd9_binary': [0, 0], 'tf_i2b2': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'tf_mtsample': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'topography_code_regex': [0, 0, 0], 'clinical_code_icd10_binary': [0, 0], 'tf_npi_city': [0], 'morphology_code_regex': [0, 0], 'tf_mrconso': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'tnm_code': [0, 0, 0, 0, 0, 0, 0, 0], 'clinical_medgen_binary': [0, 0], 'tf_ccda': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                'finding_regex': [0, 0, 0, 0, 0], 'clinical_medgen': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'drug_rxnorm_binary': [0, 0], 'tf_npi_last_name': [0], 'smoking': [0, 0, 0, 0, 0], 'latinizer': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'tf_patient_last_name': [0], 'loinc_section_attributes': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'lab_hepc_binary': [0, 0], 'loinc_section_doc_types': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'header': [0], 'lab_loinc_binary': [0, 0], 'tf_npi_phone': [0], 'tf_procedure': [0], 'tf_cities': [0], 'aspect_enforce': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'latinizer_binary': [0], 'drug_rxnorm': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'aspect_line': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'clinical_binary': [0, 0], 'tf_states': [0],
                'allergy_regex': [0, 0], 'page_break': [0, 0], 'blood_pressure': [0], 'frequency': [0, 0, 0, 0], 'lab_hepc': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'topography_binary': [0, 0], 'zipcode': [0], 'aspect': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'topography_code_binary': [0, 0], 'tf_patient_first_name': [0], 'morphology_binary': [0, 0], 'tf_disorder': [0], 'tf_npi_first_name': [0], 'morphology_code_binary': [0, 0], 'person': [0], 'units_of_measure': [0, 0, 0, 0, 0, 0, 0, 0], 'history': [0], 'spacing': [0, 0, 0], 'polarity': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'vital_signs': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'severity': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'contact_info': [0, 0, 0, 0, 0, 0, 0], 'hospital_personnel': [0, 0, 0, 0, 0, 0, 0], 'patient': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'sectionizer': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'gender': [0, 0, 0, 0, 0, 0, 0], 'form': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'len': [0, 0, 0], 'speech': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'age': [0, 0, 0, 0], 'speech_bin': [0, 0, 0, 0, 0, 0, 0, 0, 0], 'phi_indicator_words': [0, 0, 0, 0, 0, 0], 'header_aspect': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'address': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'case': [0, 0, 0], 'clinical_code_icd9': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'hospital': [0, 0], 'loinc_section': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'grammar': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'loinc_title': [0], 'analyte': [0, 0, 0, 0, 0, 0, 0, 0, 0], 'regex_covid': [0, 0, 0, 0, 0], 'tf_finding': [0], 'url': [0, 0, 0, 0, 0], 'covid_device_regex': [0], 'family': [0], 'covide_device_hint': [0, 0, 0, 0], 'time_qualifier': [0, 0, 0], 'icd': [0, 0, 0, 0, 0], 'social_history': [0, 0, 0], 'personal_history': [0], 'predisposal': [0], 'pain': [0, 0, 0], 'covid_representation': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'clinical_vocab': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'clinical_tui': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'clinical_tty': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'clinical_sem_type': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'clinical_snomed_vocab': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'clinical_snomed_tui': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'clinical_snomed_tty': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'clinical_snomed_sem_type': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'clinical_general_vocab': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'clinical_general_tui': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'clinical_general_tty': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'clinical_general_sem_type': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'clinical_medgen_vocab': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'clinical_medgen_tui': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'clinical_medgen_tty': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'clinical_medgen_sem_type': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'drug_rxnorm_vocab': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'drug_rxnorm_tui': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'drug_rxnorm_tty': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'lab_hepc_vocab': [0, 0, 0, 0], 'lab_hepc_tty': [0, 0, 0, 0], 'lab_loinc_vocab': [0, 0, 0, 0], 'lab_loinc_tty': [0, 0, 0, 0], 'diagnosis_vocab': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'diagnosis_tui': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'diagnosis_tty': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'diagnosis_sem_type': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'problem_vocab': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'problem_tui': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'problem_tty': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'problem_sem_type': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
}})

    model_metadata = ModelMetadata(features=feature_types, model_type=ModelType.drug, window_size=window_size)

    def test_mat_generator_equal_to_full_function(self):
        batch_size = BiomedEnv.BIOMED_MAX_DOC_WORD_COUNT.value
        full_out = get_3d_matrix(vectors=self.vectors,
                                 num_tokens=self.num_tokens,
                                 max_window_size=self.window_size,
                                 feature_types=self.feature_types)

        out = Mat3dGenerator(vectors=self.vectors, num_tokens=self.num_tokens,
                             features=self.feature_types, max_window_size=self.window_size)
        start = 0
        for i in range(ceil(self.num_tokens / batch_size)):
            actual = out[i]
            self.assertTrue((full_out[0][start:min(start + BiomedEnv.BIOMED_MAX_DOC_WORD_COUNT.value,
                                                   self.num_tokens - self.window_size + 1)] == actual).all())
            start += BiomedEnv.BIOMED_MAX_DOC_WORD_COUNT.value

    def test_mat_generator_equal_to_full_function_min_batch_size(self):
        batch_size = 1
        full_out = get_3d_matrix(vectors=self.vectors,
                                 num_tokens=self.num_tokens,
                                 max_window_size=self.window_size,
                                 feature_types=self.feature_types)

        out = Mat3dGenerator(vectors=self.vectors, num_tokens=self.num_tokens,
                             features=self.feature_types, max_window_size=self.window_size, batch_size=batch_size)
        start = 0
        for i in range(self.num_tokens):
            actual = out[i]
            self.assertTrue((full_out[0][start:min(start + batch_size,
                                                   self.num_tokens - self.window_size + 1)] == actual).all())
            start += batch_size

    def test_ensembler_generator_nonsense_text(self):
        for i in range(3):
            ensembler = Ensembler(model_type=ModelType.drug,
                                  model_file_list=[get_version_model_folders(ModelType.drug)[1]])
            tokens = MachineAnnotation(json_dict_input={'token': [str(i) for i in range(self.num_tokens)],
                                                        'range': [[i, i + 1] for i in range(self.num_tokens)]})
            mat_3d_gen = Mat3dGenerator(vectors=self.vectors,
                                        num_tokens=len(tokens['token']),
                                        max_window_size=20,
                                        min_window_size=1,
                                        features=ensembler.feature_list,
                                        include_all=True)

            out_full = ensembler.predict(tokens, use_generator=False, ensembler=ensembler, mat_3d=mat_3d_gen,
                                         vectors=self.vectors)

            output_generator = ensembler.predict(tokens, use_generator=True, ensembler=ensembler, vectors=self.vectors)

            self.assertTrue(((out_full.predicted_probs - output_generator.predicted_probs) < .0000001).all())

    def test_ensembler_generator_short_nonsense(self):
        # ensure it works for small values and values > token count
        for j in [5, 10, 15, BiomedEnv.BIOMED_MAX_DOC_WORD_COUNT.value + 5,
                  BiomedEnv.BIOMED_MAX_DOC_WORD_COUNT.value + 10, BiomedEnv.BIOMED_MAX_DOC_WORD_COUNT.value + 15]:
            for i in range(3):
                ensembler = Ensembler(model_type=ModelType.drug,
                                      model_file_list=[get_version_model_folders(ModelType.drug)[1]])
                tokens = MachineAnnotation(json_dict_input={'token': [str(i) for i in range(j)],
                                                            'range': [[i, i + 1] for i in range(j)]})

                out_full = ensembler.predict(tokens, vectors=self.vectors, use_generator=False, ensembler=ensembler)

                output_generator = ensembler.predict(tokens, use_generator=True, ensembler=ensembler,
                                                     vectors=self.vectors)

                self.assertTrue(((out_full.predicted_probs - output_generator.predicted_probs) < .0000001).all(),
                                f'{i} {j}')

    def test_ensembler_generator_nonsense_text_mult_models(self):
        ensembler = Ensembler(model_type=ModelType.drug)
        tokens = MachineAnnotation(json_dict_input={'token': [str(i) for i in range(self.num_tokens)],
                                                    'range': [[i, i + 1] for i in range(self.num_tokens)]})
        mat_3d_gen = Mat3dGenerator(vectors=self.vectors,
                                    num_tokens=len(tokens['token']),
                                    max_window_size=ensembler.max_window_size,
                                    min_window_size=1,
                                    features=ensembler.feature_list,
                                    include_all=True)

        out_full = ensembler.predict(tokens, use_generator=False, ensembler=ensembler, mat_3d=mat_3d_gen,
                                     vectors=self.vectors)

        output_generator = ensembler.predict(tokens, use_generator=True, ensembler=ensembler, vectors=self.vectors)

        self.assertTrue((abs(out_full.predicted_probs - output_generator.predicted_probs) < .000001).all())
