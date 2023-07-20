from collections import defaultdict
import os
from typing import Dict, Iterable, Set, List, Tuple
import unittest

from feature_service.tests.experiment.i2b2.surrogates import MimicSurrogateInjector

from text2phenotype.common.common import get_file_list, read_text, write_text
from text2phenotype.common.log import operations_logger
from text2phenotype.entity.brat import BratReader, Annotation


def get_unique_file_names(root_dir: str, sub_dirs: Iterable[str]) -> Set[str]:
    file_names = get_file_names(root_dir, sub_dirs)

    return set(os.path.basename(f) for f in file_names)


def get_file_names(root_dir: str, sub_dirs: Iterable[str]) -> Set[str]:
    file_names = set()

    for sub_dir in sub_dirs:
        file_names.update(f for f in get_file_list(os.path.join(root_dir, sub_dir), 'txt'))

    return file_names


class TestBiomed1452(unittest.TestCase):
    ROOT_DIR = '/mnt/google-drive/mimic'
    REF_FILE_DIR = '/Users/mike.banos/Documents/jira/BIOMED-1452/'
    NLP2013_ROOT_DIR = os.path.join(ROOT_DIR,
                                      'shareclef-ehealth-2013-natural-language-processing-and-information-retrieval-for-clinical-care-1.0')
    NLP2013_TXT_DIRS = {'ALLREPORTS', 'ALLREPORTS 2', 'Gold_SN2011', 'Gold_SN2012'}
    ATTR2014_ROOT_DIR = os.path.join(ROOT_DIR,
                                     'shareclef-ehealth-evaluation-lab-2014-task-2-disorder-attributes-in-clinical-reports-1.0')
    ATTR2014_TXT_DIRS = {'2014ShAReCLEFeHealthTasks2_training_10Jan2014/2014ShAReCLEFeHealthTask2_training_corpus',
                         'ShAReCLEFTask2TestTemplatesDefaultValuesWithCorpus_CORRECTED/TestCorpus',
                         'ShAReCLEFeHealth2014Task2Batch1/corpus'}
    ATTR2014_ANN_DIRS = {'2014ShAReCLEFeHealthTasks2_training_10Jan2014/2014ShAReCLEFeHealthTask2_training_pipedelimited',
                         'ShAReCLEFeHealth2014_test_data_gold', 'ShAReCLEFeHealth2014Task2Batch1/templates'}

    def test_compare_cohort_overlap(self):
        # see what is unique to the 2013 cohort
        nlp2013_files = get_unique_file_names(self.NLP2013_ROOT_DIR, self.NLP2013_TXT_DIRS)
        operations_logger.info(f'Found {len(nlp2013_files)} unique 2103 cohort files.')

        attr2014_files = get_unique_file_names(self.ATTR2014_ROOT_DIR, self.ATTR2014_TXT_DIRS)
        operations_logger.info(f'Found {len(attr2014_files)} unique 2104 cohort files.')

        differences = nlp2013_files - attr2014_files
        operations_logger.info(f'Found {len(differences)} files unique to 2013 cohort.')

    def test_summarize_note_types(self):
        attr2014_files = get_unique_file_names(self.ATTR2014_ROOT_DIR, self.ATTR2014_TXT_DIRS)
        operations_logger.info(f'Found {len(attr2014_files)} unique 2104 cohort files.')

        report_type_counts = defaultdict(int)
        for f in attr2014_files:
            report_type_counts[self.__get_doc_type(f)] += 1

        print(report_type_counts)

    @staticmethod
    def __get_doc_type(file_name: str) -> str:
        return os.path.splitext(file_name)[0].split('-')[-1]

    def test_make_brat_annotations(self):
        text_map = self.__get_content_map(self.ATTR2014_ROOT_DIR, self.ATTR2014_TXT_DIRS)
        operations_logger.info(f'Found {len(text_map)} unique texts.')

        annotation_map = self.__parse_annotations(
            self.__get_content_map(self.ATTR2014_ROOT_DIR, self.ATTR2014_ANN_DIRS))
        operations_logger.info(f'Found {len(annotation_map)} unique annotations.')

        text_map, annotation_map = self.__inject_surrogates(text_map, annotation_map)

        brat_annotations = self.__make_brat_annotations(annotation_map, text_map)

        txt_dir = os.path.join(self.ATTR2014_ROOT_DIR, 'txt')
        os.makedirs(txt_dir, exist_ok=True)
        for fname, record_txt in text_map.items():
            if fname not in brat_annotations:
                continue

            doc_type = self.__get_doc_type(fname)
            os.makedirs(os.path.join(txt_dir, doc_type), exist_ok=True)

            write_text(record_txt, os.path.join(txt_dir, doc_type, fname))

        ann_dir = os.path.join(self.ATTR2014_ROOT_DIR, 'ann')
        os.makedirs(ann_dir, exist_ok=True)
        for fname, annotations in brat_annotations.items():
            if fname not in text_map:
                continue

            doc_type = self.__get_doc_type(fname)
            os.makedirs(os.path.join(ann_dir, doc_type), exist_ok=True)

            write_text(annotations, os.path.join(ann_dir, doc_type, fname[:-3] + 'ann'))

    @classmethod
    def __inject_surrogates(cls, text_map, annotation_map):
        injector = MimicSurrogateInjector()

        for fname, record_txt in text_map.items():
            if fname not in annotation_map:
                continue

            annotation_locs = annotation_map[fname]
            annotation_texts = [cls.__get_annotation_text(record_txt, annotation_loc)
                                for annotation_loc in annotation_locs]

            surrogate_text = injector.inject(record_txt)
            if len(injector.substitution_map):
                annotation_map[fname] = cls.__get_updated_annotation_indices(annotation_locs, injector.substitution_map)
                text_map[fname] = surrogate_text

            for annotation_text, annotation_loc in zip(annotation_texts, annotation_map[fname]):
                assert cls.__get_annotation_text(surrogate_text, annotation_loc) == annotation_text

        return text_map, annotation_map

    @staticmethod
    def __get_updated_annotation_indices(annotation_locs: List[List[Tuple[int, int]]],
                                         substitution_map: Dict[Tuple[int, int], str]) -> List[List[Tuple[int, int]]]:
        for indices, sub_text in substitution_map.items():
            sub_start = indices[0]
            sub_offset = len(sub_text) - (indices[1] - sub_start)

            offset_locations = []
            for annotation_loc in annotation_locs:
                offset_location = []
                for r in annotation_loc:
                    if r[0] >= sub_start:
                        offset_location.append((r[0] + sub_offset, r[1] + sub_offset))
                    else:
                        offset_location.append(r)

                offset_locations.append(offset_location)

            annotation_locs = offset_locations

        return annotation_locs

    @staticmethod
    def __get_annotation_text(record_text: str, ranges: List[Tuple[int, int]]):
        return ' '.join(record_text[r[0]:r[1]] for r in ranges)

    @classmethod
    def __make_brat_annotations(cls, annotation_map: Dict[str, List[List[Tuple[int, int]]]],
                                text_map: Dict[str, str]) -> Dict[str, str]:
        brat_annotations = dict()

        for txt_file, range_list in annotation_map.items():
            record_text = text_map[txt_file]
            reader = BratReader()

            for ranges in range_list:
                disorder = cls.__get_annotation_text(record_text, ranges)

                annotation = Annotation()
                annotation.aspect = 'diseasesign'
                annotation.text = disorder
                annotation.spans = tuple(ranges)

                reader.add_annotation(annotation)

            brat_annotations[txt_file] = reader.to_brat()

        return brat_annotations

    @staticmethod
    def __parse_ranges(txt_range: str) -> List[Tuple[int]]:
        return [tuple([int(x) for x in r.split('-')]) for r in txt_range.split(',')]

    def __get_sign_symptom_cuis(self) -> Set[str]:
        cuis = set()

        with open(os.path.join(self.REF_FILE_DIR, 'sign_symptom_cuis.txt')) as fh:
            for line in fh:
                cuis.add(line.strip())

        operations_logger.info(f"Found total of {len(cuis)} sign/symptom CUIs.")

        return cuis

    def __parse_annotations(self, annotation_map: Dict[str, str]) -> Dict[str, List[Tuple[int]]]:
        ss_cuis = self.__get_sign_symptom_cuis()
        parsed = defaultdict(list)

        ann_count = 0
        skipped_count = 0
        for anno_str in annotation_map.values():
            for anno_line in anno_str.split('\n'):
                if not anno_line:
                    continue

                # notes: leaving this as variables so it's clear what each field is in case i need to know later.
                #        the last field is there due to poor formatting.
                #        did not have examples for internal 2 unknowns, but "generic" is the only outstanding
                txt_file, txt_range, cui, neg_indicator, neg_range, subject, subject_range, uncertain_indicator, \
                    uncertain_range, course_indicator, course_range, severity_indicator, severity_range, \
                    conditional_indicator, conditional_range, _, _, loc_indicator, loc_range, doctime_indicator, \
                    temporal_indicator, temporal_range, _ = anno_line.split('|')

                if neg_indicator == 'yes' or uncertain_indicator == 'yes' or subject != 'patient' or \
                        conditional_indicator == 'true' or cui in ss_cuis:
                    skipped_count += 1
                    continue

                ann_count += 1

                ranges = self.__parse_ranges(txt_range)

                parsed[txt_file].append(ranges)

        operations_logger.info(f'Kept {ann_count} annotations; skipped {skipped_count}')

        return parsed

    @staticmethod
    def __get_content_map(root_dir: str, txt_dirs: Iterable[str]) -> Dict[str, str]:
        text_map = dict()

        for f in get_file_names(root_dir, txt_dirs):
            fbase = os.path.basename(f)
            t = read_text(f)

            if fbase in text_map:
                if t != text_map[fbase]:
                    raise Exception(f'Found 2 different versions of {fbase}!')
            else:
                text_map[fbase] = t

        return text_map


if __name__ == '__main__':
    unittest.main()
