import bisect
import datetime
import os
import shutil
from math import inf
from typing import List, Dict, Tuple, Set, Union, Type

import numpy

from biomed.data_manipulation.data_migration import SandsFilePaths, JobMeta
from text2phenotype.annotations.file_helpers import Annotation
from text2phenotype.common import common
from text2phenotype.common.feature_data_parsing import min_max_range
from text2phenotype.common.data_source import DataSource, DataSourceContext
from text2phenotype.common.featureset_annotations import MachineAnnotation
from text2phenotype.common.log import operations_logger
from text2phenotype.constants.features import LabelEnum, LabLabel, DuplicateDocumentLabel


class BiomedDataSource(DataSource):
    BASE_TEXT_PATH = 'active_charts'

    def get_matched_annotated_files(self, label_enum: LabelEnum,
                                    context: DataSourceContext = None) -> Tuple[list, list]:
        _ = self.get_feature_set_annotation_files(context=context)
        ann_files = self.get_ann_files(label_enum=label_enum, context=context)
        fs_files = [None] * len(ann_files)
        for i in range(len(ann_files)):
            fs_files[i] = self.get_feature_set_from_ann_file(ann_files[i], context)
            if fs_files[i] is None:
                operations_logger.debug(f'No feature set annotation found for annotation file {ann_files[i]}')
                ann_files[i] = None
        real_ann_files = [ann_file for ann_file in ann_files if ann_file is not None]
        real_fs_files = [fs_file for fs_file in fs_files if fs_file is not None]
        operations_logger.info(f'Total Number of Matched annotated Files: {len(real_ann_files)}')
        if len(real_ann_files) == 0:
            msg = "No matched feature annotations and human annotations were found; check your data source?"
            operations_logger.error(msg)
            raise ValueError(msg)
        return real_ann_files, real_fs_files

    def get_original_raw_text_from_ann_file(self, ann_file: str, context: DataSourceContext):
        for ann_dir in self.ann_dirs_from_context(context):
            ann_file = ann_file.replace(f'/{ann_dir}/', '/')
        return ann_file.replace(self.HUMAN_ANNOTATION_SUFFIX, self.TEXT_SUFFIX)

    def get_feature_set_from_ann_file(self, ann_file: str, context: DataSourceContext):
        out_path = None
        for ann_dir in self.ann_dirs_from_context(context):
            if ann_dir in ann_file:
                fs_file = ann_file.replace(f'/{ann_dir}/', f'/{self.feature_set_version}/').replace(
                    self.HUMAN_ANNOTATION_SUFFIX,
                    self.MACHINE_ANNOTATION_SUFFIX)
                for subdir in self.get_subdir_from_context(context):
                    updated_path = self.update_file_path(fs_file, subdir)
                    if os.path.isfile(updated_path):
                        out_path = updated_path
                        break

                    for b in [True, False]:
                        updated_path_async = self.async_file_from_assumed_path(updated_path, b)
                        if os.path.isfile(updated_path_async):
                            out_path = updated_path_async
                            break

                operations_logger.debug(f'Failed to find fs file within subdirectories: {fs_file}')
                if os.path.isfile(fs_file):
                    out_path = fs_file
                if out_path is not None:
                    break
        # check out_path token count
        if out_path is not None and os.path.isfile(out_path) and len(common.read_json(out_path)['token']) <= self.max_token_count:
            return out_path

    def async_file_from_assumed_path(self, path, remove_extracted: bool = True):
        doc_dir, doc_fn = os.path.split(path)
        doc_uuid = doc_fn.split('.')[0]
        intermediate_dirs = os.path.join('processed', 'documents', doc_uuid)
        full_fp = os.path.join(doc_dir, intermediate_dirs, doc_fn)
        ext_text = '.extracted_text.'
        if remove_extracted:
            return full_fp.replace(ext_text, '.')

        if ext_text not in full_fp:
            full_fp = full_fp.replace(self.MACHINE_ANNOTATION_SUFFIX, '') + ext_text[:-1] + \
                      self.MACHINE_ANNOTATION_SUFFIX

        return full_fp

    def get_subdir_from_context(self, context: DataSourceContext):
        if context == DataSourceContext.testing:
            return self.testing_fs_subfolders
        elif context == DataSourceContext.validation:
            return self.validation_fs_subfolders
        return self.feature_set_subfolders

    def get_ann_results(self, ann_file: str, label_enum: LabelEnum):
        if label_enum == LabLabel:
            ann_result = self.get_brat_lab_name_value_unit(ann_file)
        else:
            ann_result = self.get_brat_label(ann_file, label_enum)

        return ann_result

    @classmethod
    def get_duplicate_token_idx(cls, ann_file, machine_annotations) -> Set[int]:
        res = cls.get_brat_label(ann_file, DuplicateDocumentLabel)
        annotations = cls.match_for_gold(token_ranges=machine_annotations.range,
                                         token_text_list=machine_annotations.tokens,
                                         brat_res=res,
                                         label_enum=DuplicateDocumentLabel)
        duplicate_tokens = {token for token in annotations if
                            annotations[token][DuplicateDocumentLabel.duplicate.value.column_index] == 1}
        return duplicate_tokens

    @classmethod
    def token_true_label_list(cls, ann_filepath: str, tokens: MachineAnnotation, label_enum) -> List[int]:
        # get the brat result
        brat_res = cls.get_brat_label(ann_filepath, label_enum)
        matched_vectors = cls.match_for_gold(tokens.range, tokens.tokens, brat_res, label_enum=label_enum)

        test_results = [0] * len(tokens)
        for i in matched_vectors:
            label_vector = matched_vectors[i]
            test_results[i] = numpy.argmax(label_vector)
        return test_results

    @staticmethod
    def tokenize(text, split_points: str = '.,:)(;/|\\!'):
        split_list = text.split()
        for split_char in split_points:
            curr_num_tokens = len(split_list)
            for i in range(curr_num_tokens):
                curr_entry = split_list.pop(0)
                split_list.extend(curr_entry.split(split_char))
        return split_list

    @staticmethod
    def match_for_gold(
            token_ranges: List[List[int]],
            token_text_list: List[str],
            brat_res: List[Annotation],
            label_enum: Union[LabelEnum, Type[LabelEnum]],
            binary_classifier: bool = False
    ) -> Dict[int, list]:
        """
        this function takes in featureset annotation and a ground truth annotation from BRAT
        and populate the correct labeling to featureset annotation to construct training matrix and label matrix
        """
        output = {}
        entry_idx = None

        for j in range(len(brat_res)):
            search = min_max_range(brat_res[j].text_range)
            index = bisect.bisect_left(token_ranges, list(search))

            # handle the case where the last annotation didn't capture the start of the token
            if index == len(token_ranges):
                index -= 1

            while index == 0 or (index < len(token_ranges) and search[1] > token_ranges[index - 1][0]):

                if index and search[0] <= token_ranges[index - 1][1] and (
                        token_text_list[index - 1] in brat_res[j].text or
                        sum([brat_text in token_text_list[index - 1] for brat_text in
                             brat_res[j].text.split()]) > 0):
                    # may need to change how featureset construct training label then
                    entry_idx = index - 1

                elif index < len(token_text_list) and search[1] >= token_ranges[index][0] and (
                        token_text_list[index] in brat_res[j].text or
                        sum([brat_text in token_text_list[index] for brat_text in brat_res[j].text.split()]) > 0):
                    entry_idx = index

                else:
                    operations_logger.debug(f"the highlighted text, {brat_res[j].text} does not match the document"
                                            f" text {token_text_list[min(index, len(token_text_list) - 1)]}")
                if entry_idx is not None:
                    if binary_classifier and label_enum[brat_res[j].label].value.column_index > 0:
                        output[entry_idx] = [0, 1]
                    elif entry_idx not in output:
                        temp_vect = [0] * len(label_enum)
                        temp_vect[label_enum.from_brat(brat_res[j].label.lower()).value.column_index] = 1
                        output[entry_idx] = temp_vect
                    else:
                        old_label = label_enum.get_from_int(numpy.argmax(output[entry_idx]))
                        if old_label.value.order == label_enum.from_brat(brat_res[j].label.lower()).value.order:
                            operations_logger.debug('multiple annotations for the same type on the same token')
                        # if multiple annotations on token due to improper tokenizations pick the  one with lowest order
                        else:
                            operations_logger.debug(f'multiple annotations of different types within same label class'
                                                    f' on the same token {brat_res[j].text}, token idx : {entry_idx}')
                            if old_label.value.order > label_enum.from_brat(brat_res[j].label.lower()).value.order:
                                temp_vect = [0] * len(label_enum)
                                temp_vect[label_enum.from_brat(brat_res[j].label.lower()).value.column_index] = 1
                                output[entry_idx] = temp_vect
                index += 1
                entry_idx = None
        return output

    def get_app_dest_for_uuids(self, uuid_set: set) -> dict:
        """
        :param uuid_set: set of document UUIDs that we are trying to find app destinations for
        :return: a dictionary of UUID: app_destination, also ensures that the document text and everything is synced
        WARNING::: Note that currently you need to delete the pod before running this for multiple environments bc the
         files get intermingled
        """
        output_uuid_dest_map = dict()
        job_ids = set()
        for uuid in list(uuid_set):
            # ignore legacy files for syncing data
            if '-' in uuid:
                continue
            # use the class that understands where to find SANDS files to sync down the document's folder
            # (self.source_bucket/processed/documents/uuid)
            doc_info = SandsFilePaths(uuid, parent_dir=self.parent_dir)
            self.sync_down(doc_info.doc_folder_path, os.path.join(self.parent_dir, doc_info.doc_folder_path))
            job_id = doc_info.job_id
            # if a job id is found append it to the set of jobs that we will be downloading metadata for
            if job_id:
                job_ids.add(job_id)
        operations_logger.info(f"Found {len(job_ids)} relevant job_ids")

        for job_id in list(job_ids):
            # use the job mta class to sync down the job metadata (s3://self.source_bucket/processed/jobs/job_id)
            job_meta = JobMeta(job_id=job_id, parent_dir=self.parent_dir)
            self.sync_down(job_meta.job_meta_path, os.path.join(self.parent_dir, job_meta.job_meta_path))
            app_dest = job_meta.app_dest
            # if there is no app destination, use app destination  = 'default'
            if app_dest is None:
                app_dest = "default"
            # if an empty string iss returned, usse app destination = 'API'
            elif len(app_dest) == 0:
                app_dest = 'API'
            # for all of the doc ids that were included in a job, add the mapping of doc_id: app_destination
            doc_ids = job_meta.doc_ids
            for i in doc_ids:
                # only incldue uuids that were annotated by humans (in the uuid set)
                if i in uuid_set:
                    output_uuid_dest_map[i] = app_dest
        for uuid in list(uuid_set):
            # if any uuids still don't have an app_destination
            # ie: are legacy enough that the job_id was stored differently, set app_desstination = 'default'
            if uuid not in output_uuid_dest_map:
                output_uuid_dest_map[uuid] = 'default'
        operations_logger.info(f'{len(output_uuid_dest_map)} annotation files with destination mappings')
        return output_uuid_dest_map

    def sync_all_data(self):
        """
        :return: This function syncs all annotation files for a given annotator to the train1 bucket at
         /annotations/annotator/todays_date/self.source_bucket/uuid.ann and syncs all text files that have an annotation
         to /active_charts/self.source_bucket/path at base of source bucket
        """
        new_text_path = os.path.join(self.parent_dir, self.BASE_TEXT_PATH, self.source_bucket)
        operations_logger.info(f"making new text path: {new_text_path}")
        os.makedirs(new_text_path, exist_ok=True)
        # date stamp the annotations to version them
        today_date = datetime.date.today().isoformat()

        for ann_dir in self.ann_dirs:
            ann_files = self.get_ann_files(ann_dir=ann_dir)
            base_new_ann_dir = os.path.join(ann_dir, today_date)
            # use the fact that files are stored in SANDS as annotations/annotator/uuid.ann
            uuids = {os.path.split(ann_fp)[1].split('.')[0] for ann_fp in ann_files}
            app_dest_mapping = self.get_app_dest_for_uuids(uuids)


            if len(app_dest_mapping) != len(uuids):
                operations_logger.warning('The number of uuids from annotations != size of app destination mapping')

            for doc_uuid in app_dest_mapping:
                # copy_text_files
                orig_text_path = SandsFilePaths(doc_uuid, parent_dir=self.parent_dir).text_path

                if os.path.isfile(orig_text_path):
                    ntp = orig_text_path.replace(self.parent_dir.rstrip('/'), new_text_path.rstrip('/'))
                    os.makedirs(os.path.split(ntp)[0], exist_ok=True)
                    shutil.copy(orig_text_path, ntp)
                    operations_logger.info(f'Copied {orig_text_path} to {ntp}')
                    current_ann_fp = os.path.join(self.parent_dir, ann_dir, f'{doc_uuid}.ann')
                    # copy ann file
                    if not os.path.isfile(current_ann_fp):
                        operations_logger.info(f"No ann file found at {current_ann_fp}")
                    new_ann_dir = os.path.join(base_new_ann_dir, app_dest_mapping[doc_uuid], self.source_bucket)
                    ann_path = current_ann_fp.replace(ann_dir, new_ann_dir)
                    os.makedirs(os.path.split(ann_path)[0], exist_ok=True)
                    shutil.copy(current_ann_fp, ann_path)
                    operations_logger.debug(f'Copied {current_ann_fp} to {current_ann_fp.replace(ann_dir, new_ann_dir)}')

                # if the text file did not get synced down, it is a legacy file and the text will need
                # to be copied over separately, but should already exist within active-charts,
                # so we still want to copy the ann file over
                elif '-' in doc_uuid:
                    if not os.path.isfile(current_ann_fp):
                        operations_logger.info(f"No ann file found at {current_ann_fp}")
                    new_ann_dir = os.path.join(base_new_ann_dir, app_dest_mapping.get(doc_uuid, 'legacy'),
                                               self.source_bucket)
                    ann_path = current_ann_fp.replace(ann_dir, new_ann_dir)
                    os.makedirs(os.path.split(ann_path)[0], exist_ok=True)
                    shutil.copy(current_ann_fp, ann_path)
                    operations_logger.info(f'Copied {current_ann_fp} to {current_ann_fp.replace(ann_dir, new_ann_dir)}')
            # sync up the annotations/annotator/date  and active_charts/self.source_bucket
            self.sync_up(
                os.path.join(self.parent_dir, ann_dir, today_date),
                os.path.join(ann_dir, today_date))
            self.sync_up(
                os.path.join(self.parent_dir, self.BASE_TEXT_PATH, self.source_bucket),
                os.path.join(self.BASE_TEXT_PATH, self.source_bucket))
