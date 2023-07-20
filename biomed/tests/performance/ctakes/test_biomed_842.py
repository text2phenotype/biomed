import threading
import os
import unittest

from text2phenotype.apiclients.biomed import BioMedClient
from text2phenotype.apiclients.feature_service import FeatureServiceClient
from text2phenotype.common.common import read_text

from biomed.deid.deid import get_phi_tokens
from biomed.biomed_env import BiomedEnv


class TestBiomed842(unittest.TestCase):
    def setUp(self) -> None:
        self.feature_service_client = FeatureServiceClient()
        self.NOTE_DIR = os.path.join(BiomedEnv.DATA_ROOT.value, 'mimic', 'biomed_674_records')

    @unittest.skip
    def test_andy_fail_records(self):
        record_files = {
            'Discharge summary/54/6/txt/54604_126129_34391.txt',
            'Discharge summary/41/0/txt/410_122592_47743.txt',
            'Discharge summary/58/4/txt/58417_156284_39868.txt',
            'Discharge summary/58/4/txt/58451_161476_36352.txt',
            'Discharge summary/58/4/txt/58483_124917_54053.txt',
            'Discharge summary/58/4/txt/58484_128086_40470.txt',
            'Discharge summary/58/4/txt/58466_144091_36183.txt',
            'Discharge summary/58/4/txt/58433_168831_53403.txt',
            'Discharge summary/58/4/txt/58449_136219_54740.txt',
            'Discharge summary/58/4/txt/5847_146952_16117.txt',
            'Discharge summary/58/4/txt/58416_116117_33870.txt',
            'Discharge summary/58/4/txt/58441_133224_25829.txt',
            'Discharge summary/58/8/txt/58857_148116_37975.txt',
            'Discharge summary/58/4/txt/58430_190383_578.txt',
            'Discharge summary/58/4/txt/58433_150152_53404.txt',
            'Discharge summary/58/4/txt/58452_176067_40427.txt',
            'Discharge summary/58/4/txt/58451_114843_36353.txt',
            'Discharge summary/58/4/txt/58414_142885_36360.txt',
            'Discharge summary/58/4/txt/58433_129499_53386.txt',
            'Discharge summary/32/4/txt/32401_187361_33181.txt',
            'Discharge summary/84/4/txt/84461_150956_37898.txt',
            'Discharge summary/58/2/txt/58223_148538_47152.txt',
            'Discharge summary/58/2/txt/58242_139697_507.txt',
            'Discharge summary/58/1/txt/58163_141861_38292.txt',
            'Discharge summary/26/9/txt/26981_197448_16331.txt',
            'Discharge summary/94/2/txt/9425_126145_46406.txt',
            'Discharge summary/16/6/txt/16662_151637_15005.txt',
            'Discharge summary/69/1/txt/69104_176991_40091.txt',
            'Discharge summary/58/3/txt/5839_178560_23921.txt',
            'Discharge summary/58/3/txt/58319_167333_2153.txt',
            'Discharge summary/15/7/txt/15716_114110_48903.txt',
            'Discharge summary/58/4/txt/5841_126020_30118.txt',
            'Discharge summary/58/4/txt/58438_112782_29447.txt',
            'Discharge summary/58/4/txt/58433_100415_53406.txt'
        }

        tasks = [self.__autocode] * 15
        data = [os.path.join(self.NOTE_DIR, record_file) for record_file in record_files]
        threads = []
        for task in tasks:
            thread = threading.Thread(target=task, args=(data,))
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

    def __autocode(self, record_files):
        total_len = 0
        record_count = 0
        for record_file in record_files:
            print(f'Processing {record_file}...')

            txt = read_text(record_file)
            total_len += len(txt)
            self.feature_service_client.annotate(txt)
            get_phi_tokens(txt)

            BioMedClient().get_phi_tokens(txt)

            record_count += 1
            if not record_count % 100:
                print(f'{record_count} records processed...')

        print(f'{record_count} records processed.')
        print(f'Avg. document length: {int(total_len / record_count)} characters')


if __name__ == '__main__':
    unittest.main()
