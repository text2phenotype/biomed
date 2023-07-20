from collections import defaultdict
import os
from typing import List
import unittest

from feature_service.feature_service_env import FeatureServiceEnv
from feature_service.nlp import nlp_cache

from text2phenotype.common.common import get_file_list, read_text
from text2phenotype.entity.brat import BratReader


class TestI2b2CtakesSmokingRecall(unittest.TestCase):
    SMOKING_SAMPLE_DIRS = [os.path.join(FeatureServiceEnv.DATA_ROOT.value, d) for d in
                        ['testing-RiskFactors-Complete',
                         'training-RiskFactors-Complete-Set1',
                         'training-RiskFactors-Complete-Set2']]

    def test(self):
        result_map = defaultdict(lambda: defaultdict(int))

        for sample_dir in self.SMOKING_SAMPLE_DIRS:
            ann_dir = os.path.join(sample_dir, 'ann')
            txt_dir = os.path.join(sample_dir, 'txt')

            ann_files = sorted(get_file_list(ann_dir, 'ann'))
            txt_files = sorted(get_file_list(txt_dir, 'txt'))

            self.__analyze(txt_files, ann_files, result_map)

        self.__report(result_map)

    @staticmethod
    def __analyze(txt_files: List[str], ann_files: List[str], result_map):

        for txt_file, ann_file in zip(txt_files, ann_files):
            ann_text = read_text(ann_file)
            reader = BratReader(ann_text)
            exp_status = {annotation.aspect for annotation in reader.annotations.values()}
            if len(exp_status) != 1:
                raise Exception(exp_status)
            exp_status = exp_status.pop()

            record_text = read_text(txt_file)
            response = nlp_cache.smoking(record_text)
            obs_status = response['smokingStatus']

            result_map[obs_status][exp_status] += 1

        return result_map

    @staticmethod
    def __report(result_map):
        status_to_aspect_map = {'CURRENT_SMOKER': 'current',
                                'PAST_SMOKER': 'past',
                                'NON_SMOKER': 'never',
                                'UNKNOWN': 'unknown'}

        obs_statuses = set()
        for obs_map in result_map.values():
            obs_statuses.update(obs_map.keys())
        obs_statuses = sorted(list(obs_statuses))

        print('\t\t' + '\t'.join(obs_statuses))
        obs_count_map = defaultdict(int)
        total = 0
        incorrect = 0
        for expected, obs_map in result_map.items():
            row = [expected]

            exp_status = status_to_aspect_map[expected]

            row_count = 0
            for obs_status in obs_statuses:
                obs_count = obs_map[obs_status]
                obs_count_map[obs_status] += obs_count
                total += obs_count

                row.append(str(obs_count))
                row_count += obs_count

                if obs_status != exp_status:
                    incorrect += obs_count

            row.append(str(row_count))
            print('\t'.join(row))

        print('TOTAL\t' + '\t'.join(str(obs_count_map[s]) for s in obs_statuses))

        correct_count = total - incorrect
        print(f'Total correct: {correct_count} of {total} ({100 * correct_count/total}%)')


if __name__ == '__main__':
    unittest.main()
