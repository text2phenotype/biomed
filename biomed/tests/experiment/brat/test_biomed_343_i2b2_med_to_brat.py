import os
import unittest

from text2phenotype.common import common
from text2phenotype.common.log import operations_logger
from text2phenotype.entity.brat import Annotation, BratReader

from biomed.biomed_env import BiomedEnv


class I2B2MedsToBrat(unittest.TestCase):
    MED_ROOT = os.path.join(BiomedEnv.DATA_ROOT.value, 'I2B2', '2009 Medication Challenge', 'gold')
    
    def test_convert_to_brat(self):
        json_dir = os.path.join(self.MED_ROOT, 'expert_label')
        json_files = common.get_file_list(json_dir, 'json')
        operations_logger.info(f"Queued {len(json_files)} files from {json_dir} for conversion.")
        
        out_dir = os.path.join(json_dir, common.version_text('BRAT'))
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        
        reader = BratReader()
        for json_file in json_files:
            operations_logger.info(f"Processing file {json_file}...")
            reader.annotations.clear()
            
            fbase = os.path.splitext(os.path.basename(json_file))[0]
            text = self.__get_record_text(fbase).replace('\n', ' ')

            contents = common.read_json(json_file)
            for content in contents:
                medication = content['annotated_med']
                span = content['range']
                span[1] = span[0] + len(medication)
                
                text_med = text[span[0]:span[1]]
                
                annotation = Annotation({'aspect': 'medication',
                                         'text': text_med,
                                         'spans': (tuple(span),)})
                
                reader.add_annotation(annotation)
                
            ann_file = os.path.join(out_dir, f'{fbase}.ann')
            common.write_text(reader.to_brat(), ann_file)
    
    def __get_record_text(self, file_base: str) -> str:
        file_name = f'{file_base}.txt'
        files = (os.path.join(self.MED_ROOT, 'gold_raw_text', 'test', file_name), 
                 os.path.join(self.MED_ROOT, 'gold_raw_text', 'train', file_name))
        
        for fname in files:
            if os.path.isfile(fname):
                return common.read_text(fname)
        
        raise Exception(f'Did not find raw text file for {file_base}.')
