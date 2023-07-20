from collections import defaultdict
import csv
import unittest
import os

from feature_service.tests.experiment.i2b2.surrogates import read_names
from feature_service.tests.experiment.mimic.test_biomed_674_inject_surrogates import TestBiomed647
from feature_service.nlp import autocode
from feature_service.aspect import chunker

from text2phenotype.common.data_source import DataSource

from text2phenotype.common.log import operations_logger


class TestBiomed1576(TestBiomed647):
    def test(self):
        operations_logger.info("Getting valid surrogates...")
        #self._get_valid_surrogates()

        # file needs to be synched down from s3://biomed-raw-data/resources
        #self._names_by_gender = read_names('us-likelihood-of-gender-by-name-in-2014.csv')

        #self._read_patient_dates()
        #self._filter_neonates()
        #self._get_year_adj()
        self.__process_patient_notes()

    def __process_patient_notes(self):
        """Process patient notes, surrogating PHI, adjusting dates, and writing new outputs."""
        # file needs to be synched down from s3://biomed-raw-data/resources/mimic
        note_file = 'NOTEEVENTS.csv'

        curr_dir = os.getcwd()

        #sync down the noteevents csv file
        ds = DataSource()

        ds.sync_down(source_path='s3://biomed-raw-data/resources/mimic', dest_path=curr_dir)

        note_file_path = os.path.join(curr_dir, note_file)

        note_count = 0
        notes_by_patient = defaultdict(list)
        with open(note_file_path, 'r') as note_fh:
            for note in csv.reader(note_fh, doublequote=False, escapechar='\\', quoting=csv.QUOTE_ALL):
                subject_id = int(note[0])

                #print(subject_id)
                #print(subject_id)
                #if subject_id not in self._patients:
                #    continue
                note_id = note[2]

                category = note[3]

                #print(category)

                notes_by_patient[subject_id].append((note_id, category, note[4]))

                note_count += 1

                #if note_count > 1000:
                #    break

                if not note_count % 5000:
                    operations_logger.info(f'Read {note_count} note events...')

        operations_logger.info(f'Read {note_count} total note events.')

        self.__generate_training_data(notes_by_patient, use_chunker=True)

    def __generate_training_data(self, notes_by_patient, use_chunker):
        #no surrogate

        doc_type_list = []
        section_header_list = []
        save_dir = os.path.join(os.getcwd(), 'data')
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        chunker_object = chunker.Chunker()
        counter = 0
        for subject_id, note_infos in notes_by_patient.items():
            #counter += 1
            #if counter > 10:
            #    break
            note_infos = sorted(note_infos) #sort by note_id

            full_text = ''

            save_file_path = os.path.join(save_dir, str(subject_id) + '.txt')

            for _, doc_type, note in note_infos:
                full_text += ' ' + note

                if not use_chunker:
                    section_result = autocode.loinc_section(note)
                    #or can use chunker/feature.sectionizer to see if there are better sections headers returns than loinc sections
                    content = section_result.get('result', [])
                    if content:
                        doc_type_temp_list = []
                        header_temp_list = []
                        #loinc_code_list = []
                        for item in content:
                            header = item.get('text', [])
                            if header:
                                header_text = header[0]
                                if header_text.islower():
                                    continue
                                doc_type_temp_list.append(doc_type) #keep the original string
                                # reverse look up the loinc code from feature_service/docont
                                header_temp_list.append(header_text) #keep the original header

                else:
                    doc_type_temp_list = []
                    header_temp_list = []
                    chunker_result = chunker_object.predict_aspect_emb_by_section_no_enforce(note)
                    for item in chunker_result:
                        if item['header']:
                            if item['text'].islower():
                                continue
                            header_temp_list.append(item['text'])
                            doc_type_temp_list.append(doc_type)
                doc_type_list.append(doc_type_temp_list)
                section_header_list.append(header_temp_list)
                with open(os.path.join(save_dir, 'doc_type_list_test_1.txt'), 'w') as fb:
                    fb.write(str(doc_type_list))
                with open(os.path.join(save_dir, 'section_list_test_1.txt'), 'w') as fb:
                    fb.write(str(section_header_list))

            #if os.path.exists(save_file_path):
            #    continue

            with open(save_file_path, 'w') as fb:
                fb.write(full_text)
                #sync up to S3 or save it locally
                #create training file, label and also the text file.

    def __process_notes(self, notes_by_patient):
        person_cache = defaultdict(list)
        note_count = 0
        patient_cache = dict()
        for subject_id, note_infos in notes_by_patient.items():
            self.__last_doc = None
            self.__last_doc_name_field = None

            patient = self._get_patient(subject_id, patient_cache, person_cache)

            # this should sort notes in a relatively liner temporal fashion, but probably some random shuffling
            note_infos = sorted(note_infos)

            substitutions = dict()
            surrogated_notes = []
            doc_type_list = []
            for _, doc_type, note in note_infos:
                surrogated_notes.append(self.__surrogate_note(note, substitutions, subject_id, patient))

                # TODO: do the annotation of sections
                # see text2phenotype.annotations.file_helpers for annotation classes and I/O support
                # annotate sectionizer with ctakes pipeline, add endpoints in feature_service/nlp/nlp_cache
                # collect the section headers and its corresponding document type as labels

                sections = autocode.loinc_section(note)
                #based on the structure of the response, parse the output to get the list of sections
                #and their loinc codes

                doc_type_list.append(doc_type)

                #get the list of titles
                #information that needed - 1. the doc type of that text. 2 aggregation of the actual text. 3 time stamp or doc_id.
                # 4. sections that are annotated by loinc section headers. output the text to a file
                #section headers are annotated by

                note_count += 1
                if not note_count % 5000:
                    operations_logger.info(f'Processed {note_count} notes...')

            full_text = '\n\n\n'.join(surrogated_notes)
            raise Exception(full_text)

            # TODO: write outputs (full_text + annotations)

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
