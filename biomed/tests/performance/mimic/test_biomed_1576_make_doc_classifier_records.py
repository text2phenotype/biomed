from collections import defaultdict
import csv
import os
import shutil
import unittest

from feature_service.tests.experiment.i2b2.surrogates import read_names
from feature_service.tests.experiment.mimic import test_biomed_674_inject_surrogates

from text2phenotype.common.log import operations_logger
from text2phenotype.common.common import write_text
from text2phenotype.constants.common import OCR_PAGE_SPLITTING_KEY


class TestBiomed1576(test_biomed_674_inject_surrogates.TestBiomed647):
    TXT_DIR = 'txt'
    TYPE_DIR = 'types'

    # @unittest.skip
    def test_surrogate(self):
        operations_logger.info("Getting valid surrogates...")
        self._get_valid_surrogates()

        # file needs to be synched down from s3://biomed-raw-data/resources
        self._names_by_gender = read_names('us-likelihood-of-gender-by-name-in-2014.csv')

        self._read_patient_dates()
        self._filter_neonates()
        self._get_year_adj()
        self.__process_patient_notes()

    def __process_patient_notes(self):
        """Process patient notes, surrogating PHI, adjusting dates, and writing new outputs."""
        # file needs to be synched down from s3://biomed-raw-data/resources/mimic
        note_file = 'NOTEEVENTS.csv'

        note_count = 0
        notes_by_patient = defaultdict(list)
        with open(note_file, 'r') as note_fh:
            for note in csv.reader(note_fh, doublequote=False, escapechar='\\', quoting=csv.QUOTE_ALL):
                subject_id = int(note[0])
                if subject_id not in self._patients:
                    continue

                note_id = note[2]
                category = note[3]

                notes_by_patient[subject_id].append((note_id, category, note[4]))

                note_count += 1
                if not note_count % 5000:
                    operations_logger.info(f'Read {note_count} note events...')

        operations_logger.info(f'Read {note_count} total note events.')

        self.__process_notes(notes_by_patient)

    def __process_notes(self, notes_by_patient):
        self.__init_note_dirs()

        page_delim = OCR_PAGE_SPLITTING_KEY[0]
        note_delim = f'\n{page_delim}\n'
        max_lines = 100

        person_cache = defaultdict(list)
        patient_count = 0
        patient_cache = dict()
        for subject_id, note_infos in notes_by_patient.items():
            self.__last_doc = None
            self.__last_doc_name_field = None

            patient = self._get_patient(subject_id, patient_cache, person_cache)

            # this should sort notes in a relatively liner temporal fashion, but probably some random shuffling
            note_infos = sorted(note_infos)

            substitutions = dict()
            surrogated_notes = []
            doc_indices = []
            offset = 0
            for _, doc_type, note in note_infos:
                surrogated_text = self.__surrogate_note(note, substitutions, subject_id, patient)

                line_count = 0
                char_count = 0
                for c in surrogated_text:
                    char_count += 1

                    if c == '\n':
                        line_count += 1
                        if not line_count % max_lines:
                            surrogated_text = surrogated_text[:char_count] + page_delim + surrogated_text[char_count:]
                            char_count += 1

                note_len = len(surrogated_text)
                doc_indices.append((offset, offset + note_len, doc_type))
                offset += note_len + len(note_delim)

                surrogated_notes.append(surrogated_text)

            full_text = note_delim.join(surrogated_notes)
            for span, note in zip(doc_indices, surrogated_notes):
                if note != full_text[span[0]:span[1]]:
                    raise Exception()

            patient_count += 1
            if not patient_count % 500:
                operations_logger.info(f'Processed {patient_count} patients...')

            self.__write_notes(subject_id, full_text, doc_indices)

    def __write_notes(self, subject_id, full_text, doc_indices):
        write_text(full_text, os.path.join(self.TXT_DIR, f"{subject_id}.txt"))

        indices_text = '\n'.join([','.join([str(i) for i in index]) for index in doc_indices])
        write_text(indices_text, os.path.join(self.TYPE_DIR, f"{subject_id}.txt"))

    def __init_note_dirs(self):
        for d in [self.TXT_DIR, self.TYPE_DIR]:
            self.__remove_dir(d)
            os.makedirs(d)

    @staticmethod
    def __remove_dir(d):
        if os.path.exists(d):
            shutil.rmtree(d)

    def __surrogate_note(self, note: str, substitutions: dict, patient_id: int, patient) -> str:
        """Inject surrogate data into a note.
        @param note: The note to process.
        @param substitutions: The aggregate mapping of PHI indicator to previously used substitution string.
        @param patient_id: The patient id.
        @return: The surrogated note.
        """
        while True:
            phi_match = self._PHI_PATTERN.search(note)
            if not phi_match:
                break

            phi_text = phi_match[0]
            if phi_match not in substitutions:
                substitutions[phi_text] = self._get_phi_sub(phi_match, patient_id, patient)

            substitution, demo_type, phi_type = substitutions[phi_text]
            sub_text = ' '.join(substitution)
            note = note.replace(phi_text, sub_text, 1)

        return note


if __name__ == '__main__':
    unittest.main()
