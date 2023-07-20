import datetime
import shutil
import os
import random
import time
from typing import Tuple, Set, Dict, Iterable

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

from text2phenotype.common.log import operations_logger
from text2phenotype.common import common
from text2phenotype.constants.features import LabelEnum
from text2phenotype.constants.features import FeatureType
from biomed import RESULTS_PATH
from biomed.common.predict_results import PredictResults
from biomed.models.data_counts import DataCounter
from biomed.models.prediction_engine import PredictionEngine
from biomed.models.model_cache import ModelCache
from biomed.constants.constants import ModelType
from biomed.data_sources.data_source import BiomedDataSource
from biomed.models.model_metadata import ModelMetadata
from biomed.train_test.job_metadata import JobMetadata
from biomed.models.testing_reports import WeightedReport


class DataMissingLabelsException(Exception):
    """Raise exception when trying to train a model misses some label classes"""
    pass


class VoterModel(PredictionEngine):
    """
    Class for training and predicting with a trained voter model
    """
    # for keeping consistent splits between train() and test() calls
    RANDOM_SEED = 1234
    TRAIN_TEST_SPLIT = 0.3

    def __init__(
            self,
            model_metadata: ModelMetadata,
            data_source: BiomedDataSource = None,
            job_metadata: JobMetadata = None,
            model_type: ModelType = None,
            binary_classifier: bool = False
    ):
        self.model_metadata: ModelMetadata = model_metadata
        self.model_metadata.model_type = model_type or model_metadata.model_type
        super().__init__(data_source, job_metadata, self.model_metadata.model_type, binary_classifier)

        # This is fragile, because the model_file_name is set with the given extension the first
        # time the parameter is called, and changing this file_ext wont change the model_file_name
        # if it's already set
        self.model_metadata.file_ext = ".joblib"  # force saving as joblib format
        self.results_job_path = os.path.join(RESULTS_PATH, self.job_metadata.job_id)

        self.model_cache: ModelCache = ModelCache()

        if self.job_metadata.random_seed:
            self.RANDOM_SEED = self.job_metadata.random_seed

    @property
    def testing_features(self) -> Set[FeatureType]:
        # we aren't using specific FeatureType features, so return the empty set
        return set()

    @property
    def window_size(self):
        # only looking at one token at a time
        return 1

    @property
    def label_enum(self) -> LabelEnum:
        return self.model_constants.label_class

    def _train_test_split_ix(self, n_samples: int, pct_train: float = 0.7):
        """
        Create list of sample indexes to use for train vs test
        Uses self.RANDOM_SEED locally to generate identical results over repeated calls
        :param n_samples: how many samples we expect
        :param pct_train: the percent of samples that will be train
        :return: Tuple[np.ndarray, np.ndarray]
        """
        rnd = random.Random(self.RANDOM_SEED)
        n_train = int(n_samples * pct_train)
        train = rnd.sample(range(n_samples), k=n_train)
        test = list(set(range(n_samples)) - set(train))

        return train, test

    def predict(self, x_probs) -> PredictResults:
        """
        Generate prediction from the ensemble_encoding probabilities
        This method will shape the encoding array appropriately for the model prediction
        :param x_probs: np.ndarray, shape (n_models, n_tokens, n_classes)
        :return: PredictResults
        """
        # TODO: this is a problem, since we arent receiving MachineAnnotations, only probabilities
        # so maybe we don't inherit from PredictionEngine?
        # load the cached model
        model = self.model_cache.model_sklearn(
            self.model_metadata.model_type,
            self.job_metadata.job_id,
            self.model_metadata.model_file_path)
        x_probs_2d = self.reshape_3d_to_2d(x_probs)
        y_prob = model.predict_proba(x_probs_2d)
        y_pred = y_prob.argmax(axis=1)
        results = PredictResults(predicted_probs=y_prob, predicted_cat=y_pred)
        return results

    def train(self):
        """
        Train the model

        :return: sklearn BaseEstimator, or whatever the model is. maybe coefficients, i dunno
        """
        # if there are any reports, move them to a subfolder
        try:
            prior_voting_method = f"{self._get_prior_voting_method()}_reports"
        except FileNotFoundError:
            operations_logger.warning("No 'ensemble_metadata.json' found in job_id folder, continuing")
            prior_voting_method = "prior_reports"
        now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self._move_existing_reports(f"{prior_voting_method}_{now_str}")

        x_probs, y_true = self.get_probabilities(self.job_metadata.job_id)

        model = self._train(x_probs, y_true)

        self.save(model)
        return model

    def _train(self, x_probs, y_true):
        """
        This method should be overridden by inheritance to return the target voter model type

        :param x_probs: np.ndarray
            3D array of the pre-ensembled probabilities, shape (n_models, n_tokens, n_classes)
        :param y_true: np.ndarray
            sparse encoding of category classes
        :return:
        """
        n_models, n_tokens, n_classes = x_probs.shape
        # put token as highest order dim, reshape to 2D
        train_ix, _ = self._train_test_split_ix(n_tokens)
        x_train = self.reshape_3d_to_2d(x_probs[:, train_ix, :])
        y_train = y_true[train_ix]

        # do sanity check to make sure our training set has all labels, thus avoiding mode collapse
        train_data_classes = np.unique(y_train)

        expected_labels = np.array(sorted([member.value.column_index for member in self.label_enum]))
        # expected_labels = np.array(self.label_enum.get_column_indices())
        if not np.array_equal(train_data_classes, expected_labels):
            label_diff = set(expected_labels) - set(train_data_classes)
            raise DataMissingLabelsException(
                "Training data is missing at least one label class! "
                f"expected: {expected_labels}, got: {train_data_classes}. "
                f"label_diff: {label_diff}. Try a different dataset?")

        # NOTE(mjp): can abstract out the different types of models here. For now, just hard code them
        estimator_params = {
            "n_estimators": 100,
            "bootstrap": True,
        }
        model = RandomForestClassifier(**estimator_params)

        operations_logger.info("Starting voter training...")
        train_start_time = time.time()
        # ######### fit the model
        model.fit(x_train, y_train)
        train_duration = time.time() - train_start_time
        operations_logger.info(
            f"Training model {self.job_metadata.job_id} took {train_duration:.3f} sec over {n_tokens} tokens")
        operations_logger.info(f"Estimator params: n_classes={model.n_classes_}, n_features={model.n_features_}")
        return model

    def test(self):
        """
        Take a subset of the raw ensemble probabilities and evaluate the newly trained model on them
        Writes reports to the results/job_id folder

        NOTE(mjp): This is a different method than what is used in PredictionEngine because it has no concept of
            MachineAnnotation files, only a set of features (aka the ensemble probabiliies). So we override the base method
        :return:
        """
        x_probs, y_true = self.get_probabilities(self.job_metadata.job_id)
        n_models, n_tokens, n_classes = x_probs.shape

        _, test_ix = self._train_test_split_ix(n_tokens)
        x_test = x_probs[:, test_ix, :]
        y_test = y_true[test_ix]

        results = self.predict(x_test)

        voted_cat = results.predicted_category
        voted_prob = results.predicted_probs

        confusion_matrix_reports = [WeightedReport(self.label_enum)]
        for report in confusion_matrix_reports:
            report.add_document(
                expected_category=y_test,
                predicted_results_cat=voted_cat,
                predicted_results_prob=voted_prob,
                tokens=None,
                duplicate_token_idx=None,
            )

        os.makedirs(os.path.join(RESULTS_PATH, self.job_metadata.job_id), exist_ok=True)
        for report in confusion_matrix_reports:
            report.write(job_id=self.job_metadata.job_id)
        operations_logger.info(f"Finished testing model {self.model_metadata.model_file_name}")

    def save(self, model, **kwargs):
        """
        Save the model and associated metadata
        :param model: sklearn estimator
        :param kwargs: passthrough
        :return:
        """
        estimator_filename = os.path.join(RESULTS_PATH, self.model_metadata.model_file_name)
        operations_logger.debug(
            f"Saving model: {self.model_metadata.model_file_name}, Writing model file"
            f" to RESULTS_PATH: {RESULTS_PATH}", tid=self.job_metadata.job_id)
        joblib.dump(model, estimator_filename)

        self.model_metadata.save()
        return self.model_metadata.model_file_path

    def _get_prior_voting_method(self):
        ensemble_meta_path = os.path.join(self.results_job_path, "ensemble_metadata.json")
        ensemble_meta_dict = common.read_json(ensemble_meta_path)
        voting_method = ensemble_meta_dict.get("_voting_method")
        if not voting_method:
            # try the full name, just in case
            voting_method = ensemble_meta_dict.get("voting_method")
        voting_method = voting_method.split(".")[-1]
        return voting_method

    def _move_existing_reports(self, target_dir: str):
        """
        The original job_id folder in RESULTS may have the txt and csv files from
        the original data extraction reports. They may still be useful, but may be overwritten
        in testing the new voter output.
        Move these files somewhere safe for posterity, when saved back to S3
        :param target_dir: str
            relative name for folder inside job_id to move reports to
        :return:
        """
        if not os.path.isdir(self.results_job_path):
            raise FileNotFoundError(f"No associated job_id found in results/ folder: {self.results_job_path}")
        target_exts = [".txt", ".csv"]
        mv_files = []
        for ext in target_exts:
            mv_files += common.get_file_list(self.results_job_path, ext)
        target_path = os.path.join(self.results_job_path, target_dir)
        if mv_files:
            os.makedirs(target_path, exist_ok=True)
            for f in mv_files:
                shutil.move(f, f.replace(self.results_job_path, target_path))

    @staticmethod
    def reshape_3d_to_2d(arr: np.ndarray):
        """
        Reshape the probability array from (n_models, n_tokens, n_classes) to (n_tokens, n_models * n_classes)
        :param arr: np.ndarray
        :return: np.ndarray
        """
        n_models, n_tokens, n_classes = arr.shape
        return arr.swapaxes(0, 1).reshape((n_tokens, n_models * n_classes))

    @staticmethod
    def reshape_2d_to_3d(arr: np.ndarray, shape: Tuple[int, int, int]):
        """
        Reshape the probability array back to the ensemble form (n_models, n_tokens, n_classes)
        :param arr: np.ndarray
        :param shape: tuple
            The target shape of the ensemble format, expecting (n_models, n_tokens, n_classes)
            As used in np.reshape, pass a -1 for the token value if it isnt clear how many tokens exist
        :return: np.ndarray
        """
        # shape is expecting the output format, but isnt taking into account the swap. so do the swap
        shape = (shape[1], shape[0], shape[2])
        return arr.reshape(shape).swapaxes(0, 1)

    @staticmethod
    def _concat_ensemble_results(results: Dict[str, Dict[str, Iterable]], key: str, axis=0) -> np.ndarray:
        """
        Read in a dictionary keyed by document_names, with subdicts containing the result types (see `key`)

        :param results: Dict[str, Dict[str, np.ndarray]]
        :param key: which array to concatenate for each doc, eg 'prob', 'predicted', 'raw_prob', 'labels',
        :param axis: which axis to concatenate along, default=0
        :return:
        """
        return np.concatenate([doc[key] for doc in results.values()], axis=axis)

    def get_probabilities(self, job_id):
        """
        Get the prevoting ensemble probabilities from a job that has the file saved in RESULTS_PATH
        :param job_id: the folder location of the extracted probabilities from the ensemble
            Used to look for the file containing the prevoting probabilities
        :return:
        """
        # assumes the data already exists in results/job_id
        expected_full_results_path = os.path.join(RESULTS_PATH, job_id, "full_info.pkl.gz")
        if not os.path.isfile(expected_full_results_path):
            raise FileNotFoundError(f"Full prediction datafile not found, file does not exist: {expected_full_results_path}")

        # format is results['filename']['data_key']
        results = pd.read_pickle(expected_full_results_path)
        y_true = self._concat_ensemble_results(results, 'labels')
        ensemble_probs = self._concat_ensemble_results(results, 'raw_prob', axis=1)
        return ensemble_probs, y_true
