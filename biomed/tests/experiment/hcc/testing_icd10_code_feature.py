import os

from feature_service.feature_set.annotation import annotate_text
from text2phenotype.apiclients import FeatureServiceClient
from text2phenotype.common import common
from biomed.tests.experiment.hcc.hcc_optima_results_output import JOB_IDS, get_completed_doc_ids_from_job_id, \
    sync_relevant_output_files, SOURCE_DIR, processed_dcs
from text2phenotype.constants.features import FeatureType

if __name__ == '__main__':
    for job_id in JOB_IDS:
        fs_client = FeatureServiceClient()
        completed_doc_ids =  get_completed_doc_ids_from_job_id(job_id=job_id)
        dict_output = {}
        if not completed_doc_ids:
            continue
        for doc_id in completed_doc_ids:
            contin = sync_relevant_output_files(doc_id=doc_id, job_id=job_id, doc_suffixes_to_include=['.extracted_text.txt'], include_chunks=True)

            if contin:
                icd_10_cods = set()
                for chunk_text_fp in common.get_file_list(os.path.join(SOURCE_DIR, job_id, processed_dcs, doc_id), '.txt', True):
                    if 'chunk' in chunk_text_fp:
                        txt = common.read_text(chunk_text_fp)
                        icd_10_code_annot = annotate_text(
                            text=txt,
                            feature_types={FeatureType.clinical_code_icd10})

                        for idx in icd_10_code_annot[FeatureType.clinical_code_icd10].token_indexes:
                            icd_10_cods.add(icd_10_code_annot.tokens[idx])

                print(icd_10_cods)
                dict_output[doc_id] = list(icd_10_cods)

        common.write_json(dict_output, os.path.join(SOURCE_DIR, 'ICD_10_CLINICAL_CODE', f'{job_id}.json'))
