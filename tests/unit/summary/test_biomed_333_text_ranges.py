import argparse
import os
import requests
import json
from builtins import *
import unittest
from tests.samples import MTSAMPLES_DIR


def create_parser():
    """
    Creates parser of arguments
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-service_url', default='http://198.199.106.70:8080/ctakes/rest')
    return parser


class TestBiomed333(unittest.TestCase):
    """
    Text ranges test.
    Warning: the test can take a lot of time
    """

    # properties with ranges
    propsWithRange = {
        'text',
        'medFrequencyNumber',
        'medFrequencyUnit',
        'medStrengthNum',
        'medStrengthUnit',
        'medDosage',
        'labValue',
        'labValueUnit'
    }

    # all possible endpoints
    endpoints = {
        'cpt_default_clinical': '/cpt/default_clinical',
        'cpt_temporal_module': '/cpt/temporal_module',
        'cvx_default_clinical': '/cvx/default_clinical',
        'cvx_lab_value': '/cvx/lab_value',
        'general_default_clinical': '/general/default_clinical',
        'general_lab_value': '/general/lab_value',
        'general_pos_tagger': '/general/pos_tagger',
        'general_temporal_module': '/general/temporal_module',
        'hcc_default_clinical': '/hcc/default_clinical',
        'healthcare_default_clinical': '/healthcare/default_clinical',
        'hepc_default_clinical': '/hepc/default_clinical',
        'hepc_drug_ner': '/hepc/drug_ner',
        'hepc_lab_value': '/hepc/lab_value',
        'hepc_temporal_module': '/hepc/temporal_module',
        'icd10_default_clinical': '/icd10/default_clinical',
        'icd9_default_clinical': '/icd9/default_clinical',
        'loinc_lab_value': '/loinc/lab_value',
        'loinc_temporal_module': '/loinc/temporal_module',
        'medgen_default_clinical': '/medgen/default_clinical',
        'ndfrt_default_clinical': '/ndfrt/default_clinical',
        'ndfrt_drug_ner': '/ndfrt/drug_ner',
        'ndfrt_smoking_status': '/ndfrt/smoking_status',
        'original_default_clinical': '/original/default_clinical',
        'rxnorm_default_clinical': '/rxnorm/default_clinical',
        'rxnorm_drug_ner': '/rxnorm/drug_ner',
        'rxnorm_temporal_module': '/rxnorm/temporal_module',
        'shrine-icd10_default_clinical': '/shrine-icd10/default_clinical',
        'shrine-icd9_default_clinical': '/shrine-icd9/default_clinical',
        'shrine-loinc_lab_value': '/shrine-loinc/lab_value',
        'shrine-rxnorm_drug_ner': '/shrine-rxnorm/drug_ner',
        'snomedct_default_clinical': '/snomedct/default_clinical',
        'snomedct_smoking_status': '/snomedct/smoking_status',
        'snomedct_temporal_module': '/snomedct/temporal_module'
    }

    def check_range(self, range_data, input_text):
        range_text = range_data[0]
        begin = range_data[1]
        end = range_data[2]
        self.assertEqual(
            range_text.upper(),
            input_text[begin:end],
            'Wrong text range {0} -> {1} != {2}'.format(range_data, range_text.upper(), input_text[begin:end])
        )

    def check_response(self, actual_json, input_text):
        """
        Recursive checking of JSON
        :param actual_json: JSON
        :param input_text: source text
        :return:
        """
        if isinstance(actual_json, dict):
            for prop in actual_json:
                if (prop in self.propsWithRange) and (len(actual_json[prop]) == 3):
                    self.check_range(actual_json[prop], input_text)
                else:
                    self.check_response(actual_json[prop], input_text)
        elif isinstance(actual_json, list):
            for idx in range(0, len(actual_json)):
                self.check_response(actual_json[idx], input_text)

    def test_text_ranges(self):
        params = create_parser().parse_args()

        headers = {
            'Content-type': 'application/x-www-form-urlencoded; charset=utf-8'
        }

        for endpoint_name in self.endpoints:
            url = params.service_url + self.endpoints.get(endpoint_name)
            for dirpath, dirnames, filenames in os.walk(MTSAMPLES_DIR):
                filenames.sort()
                for filename in filenames:
                    if not filename.endswith('.txt'):
                        continue

                    input_file = os.path.join(MTSAMPLES_DIR, filename)
                    with open(input_file, 'r') as input_text_file:
                        input_text = filename + ' ' + input_text_file.read()

                        data = {
                            'inputText': input_text
                        }
                        response = requests.post(url, data, headers=headers)
                        self.assertEqual(response.status_code, 200, msg='Wrong pipeline')
                        self.check_response(json.loads(response.text), input_text.upper())
