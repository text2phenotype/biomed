from collections import defaultdict
import os
import random
import shutil
import unittest

from biomed.document_section.mapper import mimic_to_text2phenotype_mapping
from biomed.document_section.model import DocumentTypeAnnotation
from biomed.tests.performance.mimic import test_biomed_1576_make_doc_classifier_records

from text2phenotype.common.common import get_file_list, read_json, read_text, write_json
from text2phenotype.common.featureset_annotations import MachineAnnotation
from text2phenotype.common.log import operations_logger
from text2phenotype.constants.environment import Environment
from text2phenotype.constants.features.label_types import NewDocumentTypeLabel


class TestPartition(unittest.TestCase):
    def test(self):
        machine_annotations_files = self.__get_annotation_files()
        subject_ids = self.__get_target_subjects()

        self.__move_machine_annotations(subject_ids, machine_annotations_files)

    @staticmethod
    def __move_machine_annotations(subject_ids, machine_annotations_files):
        out_dir = os.path.join(Environment.DATA_ROOT.value, 'target_jsons')
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
        os.makedirs(out_dir)

        for subject_id in subject_ids:
            if subject_id not in machine_annotations_files:
                continue

            sub_dir = random.choice(['a', 'b', 'c', 'd'])
            out_file = os.path.join(out_dir, sub_dir, f'{subject_id}.json')

            annotation = DocumentTypeAnnotation(
                MachineAnnotation(json_dict_input=read_json(machine_annotations_files[subject_id])))

            write_json(annotation.to_dict(), out_file)

    @staticmethod
    def __get_subject_id(file_name):
        return os.path.splitext(os.path.basename(file_name))[0]

    def __get_target_subjects(self):
        target_types = {
            #NewDocumentTypeLabel.dischargesummary.name,
                        NewDocumentTypeLabel.historyandphysical.name,
                        #NewDocumentTypeLabel.progressnote.name,
                        NewDocumentTypeLabel.consultnote.name,
                        #NewDocumentTypeLabel.na.name
        }

        doc_files = get_file_list(os.path.join(Environment.DATA_ROOT.value,
                                               test_biomed_1576_make_doc_classifier_records.TestBiomed1576.TYPE_DIR),
                                  '.txt')

        processed_count = 0
        doc_type_counts = defaultdict(int)
        subject_ids = set()
        for doc_file in doc_files:
            subject_id = self.__get_subject_id(doc_file)

            processed_count += 1

            doc_type = read_text(doc_file)

            subject_doc_type_counts = defaultdict(int)
            keep_subject = False
            for line in doc_type.splitlines():
                _, _, dtype = line.split(',')
                dtype = dtype.strip()

                if dtype not in mimic_to_text2phenotype_mapping:
                    dtype = NewDocumentTypeLabel.na.name
                else:
                    dtype = mimic_to_text2phenotype_mapping[dtype].name

                if dtype in target_types:
                    keep_subject = True

                subject_doc_type_counts[dtype] += 1

            if keep_subject:
                subject_ids.add(subject_id)

                for t, c in subject_doc_type_counts.items():
                    doc_type_counts[t] += c

        operations_logger.info(f"Processed {processed_count} records; kept {len(subject_ids)}")
        operations_logger.info(f'Document types: {doc_type_counts}')

        return subject_ids

    def __get_annotation_files(self):
        annotation_files = get_file_list(os.path.join(Environment.DATA_ROOT.value, 'mimic'), '.json', recurse=True)

        return {self.__get_subject_id(annotation_file): annotation_file for annotation_file in annotation_files}


if __name__ == '__main__':
    unittest.main()
