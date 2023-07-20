import os
import json
from typing import Optional
from tensorflow.keras.backend import clear_session

from text2phenotype.common.log import operations_logger

from biomed.biomed_env import BiomedEnv
from biomed.models.get_model import MODEL_CLASS_2_ENUM_CLASS, model_folder_exists, get_model_folder_path
from biomed import RESULTS_PATH
from biomed.train_test.job_metadata import JobMetadata
from biomed.models.model_base import ModelBase
from biomed.models.model_cache import find_model_file
from biomed.models.voter_model import VoterModel, DataMissingLabelsException
from biomed.models.model_metadata import ModelMetadata
from biomed.meta.ensembler import Ensembler
from biomed.meta.ensemble_model_metadata import EnsembleModelMetadata
from biomed.data_sources.data_source import BiomedDataSource


class TrainTestJob:
    def __init__(self, model_metadata: ModelMetadata,
                 job_metadata: JobMetadata,
                 data_source: BiomedDataSource,
                 ensemble_metadata: EnsembleModelMetadata = None):
        self.model_metadata: ModelMetadata = model_metadata
        self.job_metadata: JobMetadata = job_metadata
        self.data_source: BiomedDataSource = data_source
        tid = self.job_metadata.job_id
        self.ensemble_metadata: EnsembleModelMetadata = ensemble_metadata
        operations_logger.info('Saving job metadata', tid=self.job_metadata.job_id)
        self.job_metadata.save()
        self.data_source.save(self.job_metadata.job_id, RESULTS_PATH)

    @property
    def model_type(self):
        if self.model_metadata:
            return self.model_metadata.model_type
        elif self.ensemble_metadata:
            return self.ensemble_metadata.model_type

    def __sync_data(self):
        operations_logger.info('Running Data Sync Job')
        self.data_source.sync_all_data()

    def __train_ensemble_voter(self):
        self.ensemble_metadata.save(job_id=self.job_metadata.job_id)
        self.job_metadata.full_output = True
        ensembler = self.__get_ensembler()
        operations_logger.info('All model files have been synced, beginning ensembling')

        # enforce storing all probabilities locally, skip if we already have them (to save time)
        if not os.path.isfile(os.path.join(RESULTS_PATH, ensembler.job_metadata.job_id, "full_info.pkl.gz")):
            ensembler.job_metadata.full_output = True
            ensembler.test()

        voter = VoterModel(
            model_metadata=self.model_metadata,
            model_type=self.ensemble_metadata.model_type,
            data_source=self.data_source,
            job_metadata=self.job_metadata,
        )
        operations_logger.info('Beginning voter model training')

        try:
            voter.train()
        except DataMissingLabelsException as e:
            operations_logger.error(e.args[0])
            # sync the data we collected for debugging
            operations_logger.info("Error in ensemble voter training, syncing current state to S3")
            self.data_source.sync_up(os.path.join(RESULTS_PATH, self.job_metadata.job_id),
                                     os.path.join('models', 'train', self.job_metadata.job_id))
            self.data_source.sync_up(os.path.join(RESULTS_PATH, self.job_metadata.job_id),
                                     os.path.join('models', 'test', self.job_metadata.job_id))
            operations_logger.info(">>> Ensemble voter model attempted training, found missing labels in dataset.")
            return

        operations_logger.info('Model training complete, syncing data')
        self.data_source.sync_up(os.path.join(RESULTS_PATH, self.job_metadata.job_id),
                                 os.path.join('models', 'train', self.job_metadata.job_id))
        voter.test()
        operations_logger.info('Model testing complete, syncing data')
        self.data_source.sync_up(os.path.join(RESULTS_PATH, self.job_metadata.job_id),
                                 os.path.join('models', 'test', self.job_metadata.job_id))
        operations_logger.info(">>> Ensemble voter model training complete.")

    def __test_ensemble(self):
        self.ensemble_metadata.save(job_id=self.job_metadata.job_id)
        ensembler = self.__get_ensembler()
        operations_logger.info('All model files have been synced, beginning ensembling')

        ensembler.test()
        operations_logger.info(f"Ensembler testing completed, syncing results files from: "
                               f"{os.path.join(RESULTS_PATH, self.job_metadata.job_id)} to "
                               f"{os.path.join('models', 'test', self.job_metadata.job_id)}")
        self.data_source.sync_up(os.path.join(RESULTS_PATH, self.job_metadata.job_id),
                                 os.path.join('models', 'test', self.job_metadata.job_id))
        operations_logger.info(">>> Ensemble Test job completed.")

    def __active_learning(self):

        self.ensemble_metadata.save(job_id=self.job_metadata.job_id)
        operations_logger.info('All model files have been synced, beginning ensembling')

        ensembler = self.__get_ensembler()

        ensembler.active_learning(url_base=self.job_metadata.url_base,
                                  dir_part_to_replace=self.job_metadata.dir_to_replace)
        operations_logger.info(f"Active Learning List construction completed, syncing results files from: "
                               f"{os.path.join(RESULTS_PATH, self.job_metadata.job_id)} to "
                               f"{os.path.join('models', 'active', self.job_metadata.job_id)}")
        self.data_source.sync_up(os.path.join(RESULTS_PATH, self.job_metadata.job_id),
                                 os.path.join('models', 'active', self.job_metadata.job_id))

    def __train(self):
        model = self.__get_model()
        if not model:
            operations_logger.info('No model described, job complete')
            return

        operations_logger.info('Beginning model training')
        model.train()
        operations_logger.info('Model training complete, syncing data')
        self.data_source.sync_up(os.path.join(RESULTS_PATH, self.job_metadata.job_id),
                                 os.path.join('models', 'train', self.job_metadata.job_id))
        operations_logger.info('Data syncing complete')
        operations_logger.info(">>> Train job completed.")

    def __update_model(self):
        model = self.__get_model()

        if not model:
            operations_logger.info('No model described, job complete')
            return

        operations_logger.info(f'Beginning to Update the Model: {self.model_metadata.model_file_name}')
        model.update_model()
        operations_logger.info(f'Model Update Complete, Saved Altered Model to'
                               f' {model.model_metadata.model_file_name}')
        self.data_source.sync_up(os.path.join(RESULTS_PATH, self.job_metadata.job_id),
                                 os.path.join('models', 'train', self.job_metadata.job_id))
        operations_logger.info('Data syncing complete')

    def __test(self):
        model = self.__get_model()

        if not model:
            operations_logger.info('No model described, job complete')
            return
        operations_logger.info('Beginning model testing')
        model.test()
        # save testing model metadata
        self.model_metadata.save_test()
        operations_logger.info('Model testing complete, syncing data')
        self.data_source.sync_up(os.path.join(RESULTS_PATH, self.job_metadata.job_id),
                                 os.path.join('models', 'test', self.job_metadata.job_id))
        operations_logger.info('Data syncing complete')
        operations_logger.info(">>> Test job completed.")

    def __get_model(self) -> Optional[ModelBase]:
        if not self.model_type:
            return None
        if ((self.job_metadata.test and not self.job_metadata.train)
                or self.job_metadata.update_model
                or self.job_metadata.test_ensemble):
            self.__sync_model_files(model_folder_name=self.model_metadata.model_file_name)

        model_class = MODEL_CLASS_2_ENUM_CLASS[self.model_metadata.model_class]
        operations_logger.info(f"Using Model {model_class}")
        result = None

        if self.job_metadata.train:

            result = model_class(
                model_metadata=self.model_metadata,
                data_source=self.data_source,
                job_metadata=self.job_metadata,
                model_type=self.model_type)
        elif self.job_metadata.test or self.job_metadata.update_model:
            result = model_class(
                model_folder_name=self.model_metadata.model_file_name,
                data_source=self.data_source,
                job_metadata=self.job_metadata,
                model_type=self.model_type)
        return result

    def __get_ensembler(self):
        if not self.ensemble_metadata:
            return None
        for model_folder_name in self.ensemble_metadata.model_file_list:
            operations_logger.info(f"loading model: {model_folder_name}")
            model_type_folder, model_name = os.path.split(model_folder_name)
            model_type_folder = model_type_folder or self.model_type.name
            operations_logger.info(f"Locally looking for {model_folder_name}: type={model_type_folder} name={model_name}")
            try:
                meta = find_model_file(model_type_folder, model_name, suffix='.metadata.json')
            except ValueError as e:
                operations_logger.info(e.args[0])
                operations_logger.info(f"Couldn't find model locally, loading model from s3: {model_name}")
                self.__sync_model_files(model_name)

        # if we have a voting_model_folder listed, make sure it exists locally
        voting_model_folder = self.ensemble_metadata.voting_model_folder
        if voting_model_folder:
            operations_logger.info(f"loading VOTER model: {voting_model_folder}")
            operations_logger.info(
                f"Locally looking for {voting_model_folder}: type={self.model_type} name={voting_model_folder}")
            if not model_folder_exists(voting_model_folder, model_type=self.model_type):
                operations_logger.info(f"Couldn't find model locally, loading model from s3: {voting_model_folder}")
                self.__sync_model_files(voting_model_folder)
            else:
                operations_logger.info(
                    f"Found model file: {get_model_folder_path(voting_model_folder, self.model_type)}")

        if (self.job_metadata.test_ensemble
                or self.job_metadata.train_ensemble_voter
                or self.job_metadata.active_learning):
            result = Ensembler(ensemble_metadata=self.ensemble_metadata,
                               data_source=self.data_source,
                               job_metadata=self.job_metadata)
            return result

    def __sync_model_files(self, model_folder_name: str = None):
        if not model_folder_name:
            # assumes no model type prefix, which should be correct
            model_name = os.path.dirname(self.model_metadata.model_file_name)
            model_type_folder = self.model_type.name
            operations_logger.info(f"No model_folder_name passed, using: {model_name}")
        else:
            model_name = model_folder_name
            model_type_folder = self.model_type.name
            operations_logger.info(f"Looking for {model_folder_name}: type={model_type_folder} name={model_name}")

        s3_model_path = os.path.join('models', 'train', model_name)
        dest_path = os.path.join(
            BiomedEnv.BIOMED_NON_SHARED_MODEL_PATH.value,
            "resources/files",
            model_type_folder, model_name)
        operations_logger.info(f'using Base dir: {BiomedEnv.BIOMED_NON_SHARED_MODEL_PATH.value}')
        if not os.path.exists(dest_path) or len(os.listdir(dest_path)) == 0:
            operations_logger.info(f"Syncing down {s3_model_path} to {dest_path}")
            self.data_source.sync_down(s3_model_path, dest_path)
            if os.path.isdir(dest_path) and len(os.listdir(dest_path)) == 0:
                operations_logger.warning(
                    "Empty destination: no files were synced! Make sure you have the correct source bucket specified")

    def __write_failed_file(self, failed_files, failed_dir):
        failed_files_path = os.path.join(failed_dir, 'feature_annotation_failures.txt')
        with open(failed_files_path, 'w') as failed_file:
            failed_file.writelines(failed_files)
        self.data_source.sync_up(failed_dir, os.path.join('annotation_reports', self.job_metadata.job_id))

    def run(self):
        clear_session()
        if self.job_metadata.sync_data:
            self.__sync_data()
        if self.job_metadata.train_ensemble_voter:
            self.__train_ensemble_voter()
        if self.job_metadata.test_ensemble:
            self.__test_ensemble()
        if self.job_metadata.active_learning:
            self.__active_learning()
        if self.job_metadata.train:
            self.__train()
        if self.job_metadata.update_model:
            self.__update_model()
        if self.job_metadata.test:
            self.__test()


def generate_escaped_metadata(model_metadata: ModelMetadata, job_metadata: JobMetadata, data_source: BiomedDataSource):
    result = {**model_metadata.to_json(), **job_metadata.to_json(), **data_source.to_json()}
    converted_string = json.dumps(result).replace('"', r'\"')
    converted_string = f'"{converted_string}"'
    return converted_string
