from collections import defaultdict
import os
import unittest

from biomed.document_section.mapper import mimic_to_text2phenotype_mapping
from biomed.tests.performance.mimic import test_biomed_1576_make_doc_classifier_records

from text2phenotype.common.common import get_file_list, read_json, read_text, write_text
from text2phenotype.common.log import operations_logger
from text2phenotype.constants.environment import Environment
from text2phenotype.constants.features.label_types import NewDocumentTypeLabel


class TestMakeAnnFiles(unittest.TestCase):
    def test(self):
        machine_annotations = self.__merge_annotations(self.__read_machine_annotations())
        doc_ids = self.__expand_docs(self.__encode_docs(self.__get_docs()), machine_annotations)

        self.__make_anns(machine_annotations, doc_ids)

    def __make_anns(self, machine_annotations, doc_ids):
        out_dir = os.path.join(Environment.DATA_ROOT.value, 'ann')
        os.makedirs(out_dir, exist_ok=True)

        for subject_id, docs in doc_ids.items():
            annotations = machine_annotations[subject_id]
            ranges = sorted((int(i), r) for i, r in annotations['range'].items())
            tokens = annotations['token']

            entries = []
            for i in range(len(docs)):
                r = ranges[i]
                entry = [f'T{i}', f'{docs[i].value.persistent_label} {r[1][0]} {r[1][1]}', tokens[str(ranges[i][0])]]

                entries.append('\t'.join(entry))

            write_text('\n'.join(entries), os.path.join(out_dir, f'{subject_id}.ann'))

    @staticmethod
    def __get_subject_id(file_name):
        return os.path.splitext(os.path.basename(file_name))[0]

    @staticmethod
    def __get_loinc_code(annotation):
        return annotation[0]['umlsConcept'][0]['cui']

    def __get_docs(self):
        doc_files = get_file_list(os.path.join(Environment.DATA_ROOT.value,
                                               test_biomed_1576_make_doc_classifier_records.TestBiomed1576.TYPE_DIR),
                                  '.txt')

        doc_types = defaultdict(list)
        doc_type_counts = defaultdict(int)
        for doc_file in doc_files:
            doc_type = read_text(doc_file)

            subject_id = self.__get_subject_id(doc_file)

            for line in doc_type.splitlines():
                start, stop, dtype = line.split(',')

                doc_types[subject_id].append(((int(start), int(stop)), dtype))

                doc_type_counts[dtype] += 1

        operations_logger.info(f'Document types: {doc_type_counts}')

        return doc_types

    def __encode_docs(self, docs_map):
        encoded_map = {}
        for subject_id, doc_list in docs_map.items():
            docs = []

            for span, doc_type in doc_list:
                doc_type = doc_type.strip()

                label = NewDocumentTypeLabel.na \
                    if doc_type not in mimic_to_text2phenotype_mapping \
                    else mimic_to_text2phenotype_mapping[doc_type]

                docs.append((span, label))

            encoded_map[subject_id] = docs

        return encoded_map

    def __read_machine_annotations(self):
        annotation_files = get_file_list(os.path.join(Environment.DATA_ROOT.value, 'mimic'), '.json', recurse=True)

        operations_logger.info(f"Processing {len(annotation_files)} annotation files...")

        target_annotations = {'token', 'range', 'loinc_section'}
        annotation_map = {}
        for annotation_file in annotation_files:
            subject_id = self.__get_subject_id(annotation_file)

            annotations = read_json(annotation_file)
            keys = list(annotations.keys())
            for key in keys:
                if key not in target_annotations:
                    del annotations[key]

            section_indices = list(x for x in annotations['loinc_section'].keys())
            if not section_indices:
                continue

            for key in ['token', 'range']:
                key_annotations = annotations[key]
                annotations[key] = {i: key_annotations[int(i)] for i in section_indices}

            annotation_map[subject_id] = annotations
            if not len(annotation_map) % 100:
                operations_logger.info(f'Processed {len(annotation_map)} files...')

        return annotation_map

    @staticmethod
    def __expand_docs(doc_id_map, section_map):
        expanded_map = {}

        for subject_id, docs in doc_id_map.items():
            if not docs:
                continue

            if subject_id not in section_map:
                continue

            doc_index = 0
            (doc_start, doc_end), doc_type = docs[doc_index]

            sections = section_map[subject_id]['range'].values()
            expanded = []
            for sec_start, _ in sorted(sections):
                while sec_start < doc_start or sec_start > doc_end:
                    doc_index += 1
                    (doc_start, doc_end), doc_type = docs[doc_index]

                expanded.append(doc_type)

            if len(sections) != len(expanded):
                raise Exception(len(sections), len(expanded))

            expanded_map[subject_id] = expanded

        return expanded_map

    @classmethod
    def __merge_annotations(cls, annotations):
        merged = {}

        for subject_id, annotation in annotations.items():
            sections = annotation['loinc_section']
            ranges = annotation['range']
            tokens = annotation['token']

            section_list = sorted([(int(index), section) for index, section in sections.items()])

            if not section_list:
                continue

            index = section_list[0][0]
            last_index = str(index)
            new_tokens = {last_index: tokens[str(str(index))]}
            curr_range = ranges[str(index)]
            new_ranges = {last_index: (curr_range[0], curr_range[1])}
            new_sections = {last_index: section_list[0][1]}
            for i in range(1, len(section_list)):
                index, section = section_list[i]
                curr_token = tokens[str(index)]
                curr_range = ranges[str(index)]

                loinc_code = cls.__get_loinc_code(section)

                same_code = loinc_code == cls.__get_loinc_code(section_list[i - 1][1])
                adj_index = index - 1 == section_list[i - 1][0]
                if same_code and adj_index:
                    new_tokens[str(last_index)] += ' ' + curr_token
                    new_ranges[str(last_index)] = (new_ranges[str(last_index)][0], curr_range[1])
                else:
                    new_sections[str(index)] = section
                    new_ranges[str(index)] = (curr_range[0], curr_range[1])
                    new_tokens[str(index)] = curr_token
                    last_index = index

            merged[subject_id] = {'loinc_section': new_sections, 'range': new_ranges, 'token': new_tokens}

        return merged


if __name__ == '__main__':
    unittest.main()
