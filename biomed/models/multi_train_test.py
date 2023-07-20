import os
from abc import ABC, abstractmethod
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold

from biomed.constants.model_constants import MODEL_TYPE_2_CONSTANTS
from text2phenotype.common import common
from text2phenotype.common.log import operations_logger
from text2phenotype.constants.common import deserialize_enum

from biomed import RESULTS_PATH
from biomed.constants.constants import ModelType
from biomed.data_sources.data_source import BiomedDataSource
from biomed.meta.ensemble_model_metadata import EnsembleModelMetadata
from biomed.models.model_metadata import ModelMetadata
from biomed.models.testing_reports import (
    MinusPartialAnnotation,
    CuiReport,
    RemovingAdjacentConfusion,
    WeightedReport,
    ConfusionPrecisionMisclassReport,
)
from biomed.train_test.job_metadata import JobMetadata
from biomed.train_test.train_test import TrainTestJob
from biomed.common.report_helpers import score_files_to_melt_df, plot_model_perf_by_class_label


class MultiTrainTest(ABC):
    def __init__(
            self,
            input_dict: Dict):
        self.input_dict = input_dict
        self.job_count = self.get_job_count()
        self.data_source_list = self.get_data_sources()
        self.model_metadata_list = self.get_model_metadatas()
        self.job_metadata_list = self.get_job_metadatas()
        self.ensemble_metadata_list = self.get_ensemble_metadatas()

    @abstractmethod
    def get_job_count(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_data_sources(self) -> List[BiomedDataSource]:
        raise NotImplementedError

    @abstractmethod
    def get_job_metadatas(self) -> List[JobMetadata]:
        raise NotImplementedError

    @abstractmethod
    def get_model_metadatas(self) -> List[ModelMetadata]:
        raise NotImplementedError

    @abstractmethod
    def get_ensemble_metadatas(self) -> List[EnsembleModelMetadata]:
        raise NotImplementedError

    @abstractmethod
    def summarization_job(self):
        raise NotImplementedError

    @property
    def job_id(self):
        return self.input_dict['job_id']

    @property
    def model_type(self):
        return deserialize_enum(self.input_dict['model_type'], ModelType)

    @property
    def model_constants(self):
        return MODEL_TYPE_2_CONSTANTS[self.model_type]

    @property
    def label_enum(self):
        return self.model_constants.label_class

    def get_reports_clean(self):
        reports = [
            CuiReport(label_enum=self.label_enum), MinusPartialAnnotation(label_enum=self.label_enum),
            RemovingAdjacentConfusion(label_enum=self.label_enum), WeightedReport(label_enum=self.label_enum)]
        return reports

    def get_results_files(self):
        all_files = []
        for job_meta in self.job_metadata_list:
            folder = os.path.join(RESULTS_PATH, job_meta.job_id)
            all_files.extend(common.get_file_list(folder, '.txt'))
        return all_files

    def run_composite_jobs(self):
        for idx in range(self.job_count):
            operations_logger.info(
                f'running job {idx + 1}/{self.job_count} with data_source: {self.data_source_list[idx].to_json()}')
            job = TrainTestJob(
                data_source=self.data_source_list[idx],
                model_metadata=self.model_metadata_list[idx],
                ensemble_metadata=self.ensemble_metadata_list[idx],
                job_metadata=self.job_metadata_list[idx])
            job.run()

    def run(self):
        self.run_composite_jobs()
        self.summarization_job()
        operations_logger.info(f">>> Finished composite job '{self.job_id}'")


class KFoldValidation(MultiTrainTest):
    @staticmethod
    def aggregate_confusion_matrices(confusion_matrix_files) -> pd.DataFrame:
        whole_confusion_matrix = ConfusionPrecisionMisclassReport.parse_confusion_matrix_text_to_df(
            common.read_text(confusion_matrix_files[0]))
        for i in range(1, len(confusion_matrix_files)):
            whole_confusion_matrix += ConfusionPrecisionMisclassReport.parse_confusion_matrix_text_to_df(
                common.read_text(confusion_matrix_files[i]))
        return whole_confusion_matrix

    def __create_reports(self, all_files):
        for report in self.get_reports_clean():
            report_files = [file for file in all_files if report.REPORT_SUFFIX in file and 'confusion' in file]
            report.confusion_matrix = self.aggregate_confusion_matrices(report_files)
            report.write(job_id=self.input_dict['job_id'])

    def get_job_count(self):
        return self.input_dict['k_folds']

    def get_job_metadatas(self) -> List[JobMetadata]:
        job_metadatas = []
        for i in range(self.job_count):
            job_meta = JobMetadata.from_dict(self.input_dict)
            job_meta.job_id = os.path.join(job_meta.job_id, f'fold_{str(i)}')
            job_meta.train = True
            job_meta.test = True
            job_metadatas.append(job_meta)
        return job_metadatas

    def get_data_sources(self) -> List[BiomedDataSource]:
        kf = KFold(self.input_dict['k_folds'], True)
        data_sources = list()
        subfolders = self.input_dict.get('k_fold_subfolders')
        numpy_subfolders = np.array(subfolders)

        for train_index, test_index in kf.split(subfolders):
            updated_ds: BiomedDataSource = BiomedDataSource(**self.input_dict)
            updated_ds.feature_set_subfolders = list(numpy_subfolders[train_index])
            updated_ds.testing_fs_subfolders = list(numpy_subfolders[test_index])
            updated_ds.validation_fs_subfolders = updated_ds.testing_fs_subfolders
            data_sources.append(updated_ds)
        return data_sources

    def get_ensemble_metadatas(self) -> List[EnsembleModelMetadata]:
        return [None] * self.job_count

    def get_model_metadatas(self) -> List[ModelMetadata]:
        model_meta = ModelMetadata(**self.input_dict)
        return [model_meta] * self.job_count

    def summarization_job(self):
        # sync up results to test area
        all_files = self.get_results_files()
        self.__create_reports(all_files)
        self.data_source_list[0].sync_up(
            os.path.join(RESULTS_PATH, self.job_id),
            os.path.join('models', 'multi', self.job_id))


class MultiEnsembleTest(MultiTrainTest):
    def __init__(self, input_dict):
        # Get the list of models and whether or not we iterate over threshold pairs
        # need this to exist before super.init, because need number of jobs
        # TODO(mjp): add method to iterate through list of possible voter methods (listed or all)
        self.job_ensemble_metadata_list = self.get_voting_metadata(input_dict["testing_model_mapping"],
                                                                   input_dict.get("voting_method"),
                                                                   input_dict.get("voting_model_folder"))
        super().__init__(input_dict)

    @staticmethod
    def get_voting_metadata(testing_model_mapping: Dict[str, List[str]], voting_method: str, voting_model_folder=None)\
            -> List[Dict[str, Any]]:
        """
        Map the voting parameters to all models
        For each ensemble name, take the voting parameters and append them to the ensemble metadata to be used

        # TODO(mjp): add feature to iterate through voting methods

        :param testing_model_mapping: dict of key ensemble name and value list of model folders to use
        :param voting_method: the voting method to use, will deserialize to VotingMethodEnum
        :param voting_model_folder: the folder name that contains the correct voter for the voting method, if required
        :return: list of dicts for each of the model ensembles
        """
        out = []
        for model_name in testing_model_mapping:
            out.append({
                "model_file_list": testing_model_mapping[model_name],
                "model_name": model_name,
                "voting_method": voting_method,
                "voting_model_folder": voting_model_folder,
            })
        return out

    @staticmethod
    def get_ordered_model_threshold_pairs(
            testing_model_mapping: Dict[str, List[str]],
            use_threshold_pairs=False,
            use_threshold=False
    ):
        """
        Create list of dicts for ensembles to iterate over, not using threshold by default
        Option for iterating over pairs of threshold and not threshold for each ensemble

        OBSOLETE: `use_threshold` is no longer the target parameter for voting models
        TODO(mjp): update this method to iterate through different voters

        :param testing_model_mapping: dictionary with the ensemble name and the list of associated models to run
        :param use_threshold_pairs: Option flag for creating threshold and no threshold results pairs
        :param use_threshold: default value to use for Ensembler.use_threshold if not using pairs
        :return:
        """
        # sanity check that the model mapping dict contains what we want
        # throw error when we iterate over a string rather than list of model names
        assert isinstance(list(testing_model_mapping.keys())[0], str)
        assert isinstance(list(testing_model_mapping.values())[0], list)

        use_threshold_switch_list = [True, False] if use_threshold_pairs else [use_threshold]

        out = []
        for threshold_bool in use_threshold_switch_list:
            for model_name in testing_model_mapping:
                out.append({
                    "model_name": model_name,
                    "use_threshold": threshold_bool,
                    "model_file_list": testing_model_mapping[model_name]
                })
        return out

    def get_job_count(self):
        return len(self.job_ensemble_metadata_list)

    def get_model_metadatas(self) -> List[ModelMetadata]:
        return [None] * self.job_count

    def get_data_sources(self) -> List[BiomedDataSource]:
        base_data_source = BiomedDataSource(**self.input_dict)
        operations_logger.info(
            f'Created base data source from {self.input_dict} with datasource: {base_data_source.to_json()}')
        return [base_data_source] * self.job_count

    def get_job_metadatas(self) -> List[JobMetadata]:
        out = []
        for mapping in self.job_ensemble_metadata_list:
            base_job_meta = JobMetadata.from_dict(self.input_dict)
            base_job_meta.test_ensemble = True
            job_name = os.path.join(self.job_id, mapping['model_name'])
            base_job_meta.job_id = job_name
            out.append(base_job_meta)
        return out

    def get_ensemble_metadatas(self) -> List[EnsembleModelMetadata]:
        out = []
        for mapping in self.job_ensemble_metadata_list:
            ensemble_meta = EnsembleModelMetadata(
                model_type=self.model_type,
                model_file_list=mapping['model_file_list'],
                voting_method=mapping["voting_method"],
                voting_model_folder=mapping["voting_model_folder"]
            )
            out.append(ensemble_meta)
        return out

    def summarization_job(self):
        self.create_reports()
        self.data_source_list[0].sync_up(
            os.path.join(RESULTS_PATH, self.job_id),
            os.path.join('models', 'multi', self.job_id))

    def get_model_file_name_report(self, report):
        report_filenames = []
        model_file_names = []

        for job_meta in self.job_metadata_list:
            folder = os.path.join(RESULTS_PATH, job_meta.job_id)
            all_files = common.get_file_list(folder, report.REPORT_SUFFIX)
            report_files = [file for file in all_files if os.path.split(file)[1] == f'report_{report.REPORT_SUFFIX}']
            if len(report_files) != 1:
                operations_logger.info(f"Number of report files != 1, report files found: {report_files}")
            else:
                report_filenames.append(report_files[0])
                model_file_names.append(job_meta.job_id.split('/')[-1])
        return report_filenames, model_file_names

    def create_reports(self):
        for report in self.get_reports_clean():
            operations_logger.info(f"Creating {report}")
            report_files, model_names = self.get_model_file_name_report(report)
            score_melt_df = score_files_to_melt_df(report_files, model_names)
            labels = set(score_melt_df['class_label'])
            for label in labels:
                fig = plot_model_perf_by_class_label(score_melt_df, label)
                label = label.replace('/', '_')
                plt.savefig(os.path.join(RESULTS_PATH, self.job_id, f"{report.REPORT_SUFFIX}_{label}.png"))
