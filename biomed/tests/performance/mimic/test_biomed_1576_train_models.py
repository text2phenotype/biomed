from abc import ABC, abstractmethod
from collections import defaultdict
import os
import unittest

from biomed.document_section.decoder import DocumentSectionizer
from biomed.document_section.mapper import mimic_to_text2phenotype_mapping
from biomed.tests.performance.mimic import test_biomed_1576_make_doc_classifier_records

from feature_service.features.loinc import sections

from text2phenotype.common.common import read_text, get_file_list, read_json
from text2phenotype.common.featureset_annotations import MachineAnnotation, DocumentTypeAnnotation
from text2phenotype.common.log import operations_logger
from text2phenotype.constants.environment import Environment
from text2phenotype.constants.features.feature_type import FeatureType
from text2phenotype.constants.features.label_types import NewDocumentTypeLabel


class _TrainHarness(ABC):
    TEST_DIR = 'a/mimic/patient_full/txt'

    def test_train_test(self):
        annotations = self.__get_annotations()

        section_map = self._encode_annotations(annotations)
        doc_map = self.__encode_docs(self.__get_docs())

        section_train, doc_train, section_test, doc_test = self.__partition(section_map, doc_map)

        self._train(section_train, doc_train)

        model = self._load_model()

        self._test(model, section_test, doc_test)

    @abstractmethod
    def _train(self, section_list, doc_list):
        pass

    @abstractmethod
    def _predict(self, model, sections):
        pass

    def _test(self, model: DocumentSectionizer, sections_list, docs_list):
        counts = defaultdict(lambda: defaultdict(int))
        for sections, docs in zip(sections_list, docs_list):
            predicted = self._predict(model, sections)

            for obs, exp in zip(predicted, docs):
                counts[exp][obs] += 1

        self.__show_report(counts)

    @abstractmethod
    def _load_model(self) -> DocumentSectionizer:
        pass

    @abstractmethod
    def _encode_annotations(self, annotations):
        pass

    @staticmethod
    def __encode_docs(docs_map):
        doc_type_counts = defaultdict(int)

        encoded_map = {}
        for subject_id, doc_list in docs_map.items():
            doc_ints = []

            for span, doc_type in doc_list:
                doc_type = doc_type.strip()

                index = NewDocumentTypeLabel.na.value.column_index \
                    if doc_type not in mimic_to_text2phenotype_mapping \
                    else mimic_to_text2phenotype_mapping[doc_type].value.column_index

                doc_ints.append((span, index))
                doc_type_counts[index] += 1

            encoded_map[subject_id] = doc_ints

        operations_logger.info(f'Document types: {doc_type_counts}')

        return encoded_map

    def __partition(self, section_map, doc_map):
        section_train, doc_train, section_test, doc_test = [], [], [], []

        for subject_id, sections in section_map.items():
            if not sections:
                continue

            docs = doc_map[subject_id]

            if os.path.exists(os.path.join(Environment.DATA_ROOT.value, 'target_jsons', self.TEST_DIR, f'{subject_id}.json')):
                section_test.append(sections)
                doc_test.append(docs)
            else:
                section_train.append(sections)
                doc_train.append(docs)

        operations_logger.info(f'{len(doc_train)} training; {len(doc_test)} testing')

        return section_train, doc_train, section_test, doc_test

    @staticmethod
    def __get_subject_id(file_name):
        return os.path.splitext(os.path.basename(file_name))[0]

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

    def __get_annotations(self):
        annotation_files = get_file_list(os.path.join(Environment.DATA_ROOT.value, 'target_jsons'), '.json', recurse=True)

        operations_logger.info(f"Processing {len(annotation_files)} annotation files...")

        annotation_map = {}
        for annotation_file in annotation_files:
            subject_id = self.__get_subject_id(annotation_file)

            annotations = DocumentTypeAnnotation(MachineAnnotation(json_dict_input=read_json(annotation_file)))
            annotation_map[subject_id] = annotations
            if not len(annotation_map) % 500:
                operations_logger.info(f'Processed {len(annotation_map)} files...')

        return annotation_map

    @staticmethod
    def __show_report(counts):
        # counts -> mapping of exp -> obs
        class_tp_counts = defaultdict(int)
        class_obs_counts = defaultdict(int)
        class_exp_counts = defaultdict(int)
        for exp_label in NewDocumentTypeLabel:
            row = [exp_label.value.persistent_label]
            class_counts = counts[exp_label.value.column_index]

            class_exp_counts[exp_label.value.column_index] = sum(class_counts.values())
            class_tp_counts[exp_label.value.column_index] = class_counts[exp_label.value.column_index]

            for obs_label in NewDocumentTypeLabel:
                class_obs_counts[exp_label.value.column_index] += \
                    counts[obs_label.value.column_index][exp_label.value.column_index]

                row.append(str(class_counts[obs_label.value.column_index]))

            print('\t'.join(row))

        tp_count = sum(class_tp_counts.values())
        header_count = sum(class_obs_counts.values())

        print('')
        print(f'{header_count} total headers, {tp_count} predicted correctly')

        print('')
        print('\t'.join(['DocType', 'Precision', 'Recall', 'f1', 'Support']))
        for exp_label in NewDocumentTypeLabel:
            tp_count = class_tp_counts[exp_label.value.column_index]
            support = class_exp_counts[exp_label.value.column_index]
            obs_count = class_obs_counts[exp_label.value.column_index]

            precision = (tp_count / obs_count) if obs_count else 0
            recall = (tp_count / support) if support else 0
            pr = precision + recall
            f1 = 2 * (precision * recall) / (precision + recall) if pr else 0
            row = [exp_label.value.persistent_label,
                   f'{precision:5.2f}',
                   f'{recall:5.2f}',
                   f'{f1:5.2f}',
                   str(support)]

            print('\t'.join(row))

    @staticmethod
    def _get_loinc_code(annotation):
        return annotation[0]['umlsConcept'][0]['cui']

    @staticmethod
    def __decompose_docs(section_codes, doc_codes):
        start, end = 0, 0
        while start < len(doc_codes):
            while end < len(doc_codes) and doc_codes[start] == doc_codes[end]:
                end += 1

            yield section_codes[start:end], doc_codes[start:end]

            start = end

    @staticmethod
    def _expand_docs(doc_list, section_list):
        expanded_list = []

        for docs, sections in zip(doc_list, section_list):
            doc_index = 0
            (doc_start, doc_end), doc_type = docs[doc_index]

            expanded = []
            for sec_type, (sec_start, _) in sections:
                while sec_start < doc_start or sec_start > doc_end:
                    doc_index += 1
                    (doc_start, doc_end), doc_type = docs[doc_index]

                expanded.append(doc_type)

            if len(sections) != len(expanded):
                raise Exception(len(sections), len(expanded))

            expanded_list.append(expanded)

        return expanded_list


# TODO: model w/page breaks
class TestTrainRichenHMM(unittest.TestCase, _TrainHarness):
    MODEL_FILE = 'richen_hmm'

    def _train(self, section_list, doc_list):
        model = self.__get_model()
        doc_list = self._expand_docs(doc_list, section_list)
        section_list = self.__clean_sections(section_list)

        model.train(section_list, doc_list)

    def _test(self, model: DocumentSectionizer, sections_list, docs_list):
        expanded = self._expand_docs(docs_list, sections_list)

        doc_type_counts = defaultdict(int)
        for doc_list in expanded:
            for doc_type in doc_list:
                doc_type_counts[doc_type] += 1

        operations_logger.info(f'Document types: {doc_type_counts}')

        super()._test(model, self.__clean_sections(sections_list), expanded)

    def _predict(self, model, sections):
        return model.recover(sections)

    def _load_model(self) -> DocumentSectionizer:
        model = self.__get_model()
        model.load()
        return model

    def __get_model(self) -> DocumentSectionizer:
        return DocumentSectionizer(NewDocumentTypeLabel, model_file_name=self.MODEL_FILE)

    @staticmethod
    def __clean_sections(section_lists):
        cleaned = []

        for section_list in section_lists:
            cleaned.append([section[0] for section in section_list])

        return cleaned

    def _encode_annotations(self, annotations):
        loinc_code_map = {code: i for i, (code, _) in enumerate(sections)}

        encoded_map = defaultdict(list)
        for subject_id, doc_annotation in annotations.items():
            ranges = doc_annotation['range']
            for index, annotation in doc_annotation[FeatureType.loinc_section].items():
                # remove lower case
                if annotation[0]['text'][0].islower():
                    continue

                code = self._get_loinc_code(annotation)
                if code not in loinc_code_map:
                    continue

                encoded_map[subject_id].append((loinc_code_map[code], ranges[int(index)]))

        return encoded_map


if __name__ == '__main__':
    unittest.main()
