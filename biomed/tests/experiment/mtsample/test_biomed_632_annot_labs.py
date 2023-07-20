import os
import unittest
from typing import Dict, Set
from collections import defaultdict
from importlib import reload

from text2phenotype.common import common
from text2phenotype.common.log import operations_logger
from text2phenotype.entity.brat import BratReader

from feature_service.nlp import autocode

from biomed.biomed_env import BiomedEnv


class TestBiomed632(unittest.TestCase):
    @unittest.SkipTest
    def test_autocode(self):
        labs = self.__extract_labs()
        
        orig_nlp_host = os.environ.get('NLP_HOST', '')

        NLP_CURRENT = 'https://prod-demo1-nlp2clone.text2phenotype.com/nlp/rest'
        NLP_SANDBOX = 'https://dev-sandbox-nlp.text2phenotype.com/nlp/rest'

        try:
            old_uncoded = self.__autocode(labs, "UncodedCurrentNLPHost.tsv", NLP_CURRENT,
                                          autocode.PipelineURL.hepc_lab_value.value)

            new_uncoded = self.__autocode(labs, "UncodedSandboxNLPHost.tsv", NLP_SANDBOX,
                                          autocode.PipelineURL.loinc.value)

        finally:
            os.environ['NLP_HOST'] = orig_nlp_host
        
        self.__compare_uncoded(old_uncoded, new_uncoded)
        
    def __extract_labs(self) -> Dict[str, int]:
        """Get the set of unique labs and corresponding number of occurrences."""
        samples_dir = os.path.join(BiomedEnv.DATA_ROOT.value, 'brat', 'annotations',
                                   'angelica.given@yahoo.com', 'work', 'mtsamples_txt_clean')
        
        file_count = 0
        labs = defaultdict(int)
        for ann_file in common.get_file_list(os.path.join(samples_dir), '.ann'):
            source = common.read_text(ann_file)
            if not source:
                continue
            
            file_count += 1
            
            reader = BratReader(source)
            for annotation in reader.annotations.values():
                if annotation.aspect != 'lab':
                    continue
                
                labs[annotation.text] += 1
        
        operations_logger.info(f"Extracted {len(labs)} unique labs from {file_count} files.")
        
        return labs
    
    def __autocode(self, labs: Dict[str, int], uncoded_file: str, nlp_host: str, nlp_url=autocode.PipelineURL.hepc_lab_value.value) -> Set[str]:
        """Try to autocode labs.
        @return: The set of labs that did not code.
        """
        os.environ['NLP_HOST'] = nlp_host
        reload(autocode)
        operations_logger.info(f"Processing labs with host: {nlp_host}")
        
        uncoded_labs = set()
        coded_labs = set()
        
        for lab in labs.keys():
            response = autocode.autocode(lab, nlp_url)
            if not response['labValues']:
                uncoded_labs.add(lab)
            else:
                coded_labs.add(lab)
        
        operations_logger.info(f"{len(coded_labs)} ({100 * len(coded_labs) / len(labs)}%) were properly resolved.")
        
        uncoded_text = ""
        for lab in uncoded_labs:
            uncoded_text += f"{lab}\t{labs[lab]}\n"
        
        common.write_text(uncoded_text, uncoded_file)
        operations_logger.info(f"Wrote {len(uncoded_labs)} uncoded labs to {uncoded_file}.")
        
        return uncoded_labs
    
    def __compare_uncoded(self, old_uncoded, new_uncoded):
        """Determine what labs are common/unique to each NLP server."""
        old_file = "OldOnlyUncoded.txt"
        old_unique = old_uncoded - new_uncoded
        common.write_text('\n'.join(old_unique), old_file)
        operations_logger.info(f"Wrote {len(old_unique)} labs unique to the old NLP server to {old_file}.")
        
        new_file = "NewOnlyUncoded.txt"
        new_unique = new_uncoded - old_uncoded
        common.write_text('\n'.join(new_unique), new_file)
        operations_logger.info(f"Wrote {len(new_unique)} labs unique to the updated NLP server to {new_file}.")
        
        common_file = "CommonUncoded.txt"
        common_labs = old_uncoded & new_uncoded
        common.write_text('\n'.join(common_labs), common_file)
        operations_logger.info(f"Wrote {len(common_labs)} labs uncoded on both servers to {common_file}.")


if __name__ == '__main__':
    unittest.main()
