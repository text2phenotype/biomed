from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import chain
from math import ceil
import os
import string
from threading import Lock
from typing import (
    List,
    Dict, Set)
import gc

import numpy as np
import tensorflow as tf

from text2phenotype.common.data_source import DataSource
from text2phenotype.common.featureset_annotations import MachineAnnotation, Vectorization
from text2phenotype.apm.metrics import text2phenotype_capture_span
from text2phenotype.common import common
from text2phenotype.common.log import operations_logger
from text2phenotype.constants.features import FeatureType

from biomed.common.biomed_ouput import BiomedOutput
from biomed.constants.model_constants import ModelClass
from biomed.models.get_model import get_model_from_model_folder
from biomed.models.prediction_engine import PredictionEngine
from biomed.models.model_cache import ModelCache
from biomed.biomed_env import BiomedEnv
from biomed.common.mat_3d_generator import Mat3dGenerator
from biomed.common.predict_results import PredictResults
from biomed.meta.ensemble_model_metadata import EnsembleModelMetadata
from biomed.meta.voting_methods import vote
from biomed.models.model_metadata import ModelMetadata
from biomed.train_test.job_metadata import JobMetadata

PUNCT_SET = set(string.punctuation)


class Ensembler(PredictionEngine):
    """
    Create and evaluate predictions from data sources, using a collection of Models.
    EnsembleModelMetadata expects a `model_file_list`, which will try to load the listed files
        from the LOCAL_FILES folder (biomed/resources/files/)
    """

    def __init__(self,
                 ensemble_metadata: EnsembleModelMetadata = None,
                 data_source: DataSource = None,
                 job_metadata: JobMetadata = None,
                 **kwargs):
        """
        Class for combining the predictions of several different models and producing formatted output with token ids, predicted classes and probabilities
        Implements data formatting (3Dmatrix for LSTM) and multithreading of predictions
        :param ensemble_metadata: EnsembleModelMetadata object, if not provided, will use kwargs
        :param data_source: DataSource object, if not provided, will use kwargs
        :param job_metadata: JobMetadata object, if not provided, will use kwargs
        :param kwargs: Arguments for initializing any of the above objects
        """

        self.ensemble_metadata = ensemble_metadata or EnsembleModelMetadata(**kwargs)

        ensemble_type = self.ensemble_metadata.model_type
        data_source = data_source or DataSource(**kwargs)
        job_metadata = job_metadata or JobMetadata.from_dict(kwargs)

        # initializing PredictionEngine to use models for prediction
        super().__init__(model_type=ensemble_type, data_source=data_source, job_metadata=job_metadata)

        self.model_list = []
        self._feature_size = None
        self._feature_list = None

        feature_list = set()
        ws_set = set()
        for model_folder_name in self.ensemble_metadata.model_file_list:
            model = get_model_from_model_folder(model_folder_name, base_model_type=self.model_type)
            self.model_list.append(model)

            feature_list = feature_list.union(set(model.model_metadata.features))
            ws_set.add(model.model_metadata.window_size)
            # if any model has a real feature service client, use that for the ensembler
            if model.feature_service_client and not self.feature_service_client:
                self.feature_service_client = model.feature_service_client
        self.max_window_size = max(ws_set)
        self.min_window_size = min(ws_set)
        self.feature_list = feature_list
        self.model_metadata = ModelMetadata(
            features=self.feature_list,
            model_type=self.ensemble_metadata.model_type)
        self.model_cache = ModelCache()  # the Ensemble is responsible for caching the voter

    @property
    def window_size(self):
        return self.max_window_size

    def model_list_by_class(self, target_class_enum: ModelClass):
        """
        Return a sublist of the models in self.model_list that match the target ModelClass
        :param target_class_enum: ModelClass, the target model class we want to list
        :return: list
            list subset of models from self.model_list matching the target class
        """
        return [model for model in self.model_list if model.model_metadata.model_class == target_class_enum]

    @text2phenotype_capture_span()
    def multithread_bert_predictions(
            self,
            tokens: MachineAnnotation,
            num_classes: int,
            use_tf_session: bool = True,
            tid: str = None):
        """
        :param tokens: Machine annotation object containing all features required for all models being ensembled
        :param num_classes: integer number of classes
        :param use_tf_session: use tf.Graph & tf.Session in cached ModelWrapper
        :param tid: str, host id, used for logging
        :return: Dictionary of {model_folder_name: PredictResults objects}
        Note that this is optimized for lstm memory optimization purposes, if it is feasible to simply run model.predict
         from tokens + vectors than that would be the preferred path
        """
        num_tokens = len(tokens["token"])
        predictions = {
            mf.model_metadata.model_file_name: np.zeros((num_tokens, num_classes))
            for mf in self.model_list_by_class(ModelClass.bert)
        }
        # not really using batches, cause the input lists are just tokens and not full 3d matrices
        batch_size = BiomedEnv.BIOMED_MAX_DOC_WORD_COUNT.value  # explicit batch, rather than using Mat3DGenerator
        n_batches = max(ceil(num_tokens / batch_size), 1)

        operations_logger.info(f'Beginning {self.model_type.name} Model Predictions, '
                               f'Batch Size = {batch_size}, ')

        thread_result_predictions = dict()  # keyed by model name, hold the batch prediction results

        prediction_lock = Lock()
        thread_list = []
        for model in self.model_list_by_class(ModelClass.bert):
            # BaseBert.get_model_worker specifies the prediction_function,
            # which will take parameters from executor.submit below
            thread_list.append(
                model.get_model_worker(
                    lock=prediction_lock, results=thread_result_predictions, use_tf_session=use_tf_session, tid=tid))

        with ThreadPoolExecutor(max_workers=BiomedEnv.MAX_THREAD_COUNT.value) as executor:
            futures = []
            for thread in thread_list:
                futures.append(executor.submit(thread.run,
                                               tokens=tokens.tokens, ))
            for future in as_completed(futures):
                future.result()

        # combine batches
        # for each batch (really there is only one batch)
        for model_obj in self.model_list_by_class(ModelClass.bert):
            # get model fn and window size
            model_fn = model_obj.model_metadata.model_file_name
            predictions[model_fn] = thread_result_predictions.get(model_fn)

        # attempt to clear the session memory
        tf.keras.backend.clear_session()
        return {
            model_name: PredictResults(predicted_probs=y_prob)
            for model_name, y_prob in predictions.items()
        }

    @text2phenotype_capture_span()
    def multithread_lstm_predictions(
            self,
            tokens: MachineAnnotation,
            vectors: Vectorization,
            num_classes: int,
            use_tf_session: bool = True,
            tid: str = None):
        """
        :param tokens: Machine annotation object containing all features required for all models being ensembled
        :param vectors: Vectorization object containing all vectorized features required for all model being ensembled
        :param num_classes: integer number of classes
        :param use_tf_session: use tf.Graph & tf.Session in cached ModelWrapper
        :param tid: str, host id, used for logging
        :return: Dictionary of {model_folder_name: PredictResults objects}
        Note that this is optimized for lstm memory optimization purposes, if it is feasible to simply run model.predict
         from tokens + vectors than that would be the preferred path
        """
        mat_3d_gen = Mat3dGenerator(vectors=vectors,
                                    num_tokens=len(tokens['token']),
                                    max_window_size=self.max_window_size,
                                    min_window_size=self.min_window_size,
                                    features=self.feature_list,
                                    tid=tid, include_all=True)

        num_tokens = mat_3d_gen.num_tokens
        # create dictionary to hold all prediction output
        predictions = {
            mf.model_metadata.model_file_name: np.zeros((mat_3d_gen.num_tokens, num_classes))
            for mf in self.model_list_by_class(ModelClass.lstm_base)
        }
        end_vals = {}
        operations_logger.info(f'Beginning {self.model_type.name} Model Predictions, '
                               f'Batch Size = {mat_3d_gen.batch_size}, '
                               f'Number of batches: {len(mat_3d_gen)}')
        # get predictions for each model on max_mat_3d_gen shape matrix at a time
        for i in range(len(mat_3d_gen)):
            start = i * mat_3d_gen.batch_size
            stop = min((i + 1) * mat_3d_gen.batch_size, num_tokens)
            temp_mat = mat_3d_gen[i]

            operations_logger.debug(f'Getting Individual Model Predictions Batch Number {i}/{len(mat_3d_gen)}',
                                    tid=tid)
            tmp_predictions = {}

            prediction_lock = Lock()
            thread_list = []

            for model in self.model_list_by_class(ModelClass.lstm_base):
                thread_list.append(
                    model.get_model_worker(
                        lock=prediction_lock, results=tmp_predictions, use_tf_session=use_tf_session, tid=tid))

            with ThreadPoolExecutor(max_workers=BiomedEnv.MAX_THREAD_COUNT.value) as executor:
                futures = []
                for thread in thread_list:
                    futures.append(executor.submit(thread.run,
                                                   mat_3d=temp_mat,
                                                   feature_col_mapping=mat_3d_gen.feature_col_mapping))
                for future in as_completed(futures):
                    future.result()
            # process output which will be dict of model file name to 3d matrix, batch size x window size x num_classes
            self.combine_batches(full_predictions=predictions,
                                 tmp_3d_predictions=tmp_predictions,
                                 end_vals=end_vals,
                                 start_idx=start,
                                 stop_idx=stop,
                                 mat_3d_gen=mat_3d_gen,
                                 curr_batch_num=i,
                                 tid=tid)

        # attempt to clear the session memory
        tf.keras.backend.clear_session()
        operations_logger.debug('Individual Model Predictions Completed', tid=tid)
        return {key: PredictResults(predicted_probs=value) for key, value in predictions.items()}

    def combine_batches(self,
                        full_predictions: Dict[str, np.ndarray],
                        tmp_3d_predictions: Dict[str, np.ndarray],
                        end_vals: Dict[str, np.ndarray],
                        start_idx,
                        stop_idx,
                        mat_3d_gen,
                        curr_batch_num: int,
                        tid: str = None):
        """
        purpose of this function is to combine all prediction windows for all tokens across models
        first handles issues of differing window sizes/token numbers
        then combines probabilities across timesteps with voting
        Note: num_tokens here may be better described as num_timesteps, as it is related to the number of steps the model
        took to see all tokens
        :param full_predictions: Dictionary of model_name: [num_tokens, num_label_classes], holds combined predictions
        :param tmp_3d_predictions: dictionary of model name: [num_tokens, window_size, num_label_classes]
        :param end_vals: dictionary of model_name: [window_size, window_size, num_label_classes]
        is the tail end of the previous batches 3-d matrix
        :param start_idx: token index of batch start
        :param stop_idx: batch end
        :param mat_3d_gen: matrix 3d generator used to get predictions (used for num tokens, window size]
        :param curr_batch_num: which batch we are on
        :param tid:
        :return: None, updates full_predictions in place
        """
        # loop through models
        for model_obj in self.model_list_by_class(ModelClass.lstm_base):
            # get model fn and window size
            model_fn = model_obj.model_metadata.model_file_name
            wind_size = model_obj.model_metadata.window_size
            # combine the end matrix from previous batch with the newest  3d matrix
            pred_mat = self.combine_results(end_vals.get(model_fn), tmp_3d_predictions.get(model_fn))
            # start at batch start_index - length of our end vals df
            tmp_start = start_idx - len(end_vals.get(model_fn, []))
            tmp_stop = stop_idx
            # calculate the number of expected batches for a model based on tokens and window size
            model_spec_step_count = max(ceil((mat_3d_gen.num_tokens - wind_size + 1) / mat_3d_gen.batch_size) - 1, 0)
            # if we are at the end of the document update the predicted matrix and update stop to be the number of
            # tokens
            if curr_batch_num == model_spec_step_count:
                tmp_stop = mat_3d_gen.num_tokens
                if mat_3d_gen.min_window_size != wind_size:
                    # if the number of tokens left is less than the window size, only the first entry should exist
                    # TODO: This seems to cut off any batches with tokens < window size, is that expected?
                    if mat_3d_gen.num_tokens < wind_size:
                        pred_mat = pred_mat[0:1]
                    # otherwise, either round down or round up what is included in pred_mat
                    elif model_spec_step_count == 0:
                        pred_mat = pred_mat[:mat_3d_gen.num_tokens - (curr_batch_num
                                                                      * mat_3d_gen.batch_size) - wind_size + 1]
                    elif (mat_3d_gen.num_tokens - wind_size + 1) % mat_3d_gen.batch_size == 0:
                        pred_mat = pred_mat[:mat_3d_gen.num_tokens - (curr_batch_num
                                                                      * mat_3d_gen.batch_size) + wind_size]
                    else:
                        pred_mat = pred_mat[:mat_3d_gen.num_tokens - (curr_batch_num * mat_3d_gen.batch_size) + 1]
            # if we are past the end of the document dont process anything, this can happen when models within the same
            # ensembler have different window sizes
            elif curr_batch_num > model_spec_step_count:
                continue
            # if we are not at the end of the document hold the last n=window_size predictions so that we can use them
            # to vote on the next batch without losing context
            else:
                end_vals[model_fn] = tmp_3d_predictions[model_fn][-wind_size:, :, :]

            # vote takes the 3d matrix and turns it into num tokens x num classes
            voted_out = model_obj.vote_helper(pred_mat, tmp_stop - tmp_start, tid=tid)
            if mat_3d_gen.num_tokens < wind_size:
                voted_out = voted_out[:stop_idx]
            # handle binary classifier case
            if voted_out.shape[1] == 2:
                full_predictions[model_fn][start_idx: tmp_stop, :2] = voted_out[-(tmp_stop - start_idx):]
            else:
                full_predictions[model_fn][start_idx: tmp_stop] = voted_out[-(tmp_stop - start_idx):]

    @staticmethod
    def combine_results(old, new):
        # append the new matrix to the end of the old matrix if the old exists
        if old is not None:
            if new is not None:
                # NOTE: this will malloc a new array, so can be an expensive call
                mat = np.append(old, new, axis=0)
            else:
                mat = old
        elif new is not None:
            mat = new
        else:
            raise ValueError('Either the new or old matrix must be provided')
        return mat

    def train(self):
        pass

    @text2phenotype_capture_span()
    def predict(self, tokens: MachineAnnotation, vectors: Vectorization, use_generator: bool = False,
                **kwargs) -> PredictResults:
        """ wrapper for meta_classifier """
        tid = kwargs.get('tid')
        operations_logger.debug('Beginning Ensembler Predict', tid=tid)
        res = self.meta_classifier(tokens, vectors=vectors, use_generator=use_generator, **kwargs)

        operations_logger.debug('Ensembler Predictions Completed Successfully', tid=tid)
        return res

    def get_ensemble_encoding_dimensions(self, tokens: MachineAnnotation, classes: List = None):
        y = len(tokens)
        num_models = len(self.model_list)
        max_window_size = self.model_list[0].model_metadata.window_size
        for model in self.model_list[1:]:
            if model.model_metadata.window_size > max_window_size:
                max_window_size = model.model_metadata.window_size
        # if max_window_size > y:
        #     y = max_window_size
        if classes and isinstance(classes[0], list):
            num_classes = len(set(chain.from_iterable(*classes)))
        else:
            num_classes = len(self.label_enum.__members__)
        return num_models, y, num_classes

    @text2phenotype_capture_span()
    def meta_classifier(
            self, tokens: MachineAnnotation, vectors: Vectorization, tid: str = None,
            **kwargs) -> PredictResults:
        """
        Gets predictions from each model in the ensemble and collapses them into a PredictResult
        object with one prediction and probability vector for each token
        :param tokens: MachineAnnotation, contains the annotation object with the original tokens and features
        :param vectors: Vectorization, contains the numericized features to push into a model
        :param tid:
        :param kwargs:
        :return: PredictResults object
        """
        num_models, num_tokens, num_classes = self.get_ensemble_encoding_dimensions(tokens)
        # predictions is dictionary of {model_file_name: PredictResults}

        # for each model type, collect the predictions
        predictions = dict()
        # default workers to using tf sessions for now
        # each ensemble model will need to define this kwarg for the model's needs
        use_tf_session = kwargs.get('use_tf_session', True)
        if len(self.model_list_by_class(ModelClass.bert)):
            bert_predictions = self.multithread_bert_predictions(
                tokens=tokens,
                num_classes=num_classes,
                use_tf_session=use_tf_session,
                tid=tid)
            predictions.update(bert_predictions)

        if len(self.model_list_by_class(ModelClass.lstm_base)):
            lstm_predictions = self.multithread_lstm_predictions(
                tokens=tokens,
                vectors=vectors,
                num_classes=num_classes,
                use_tf_session=use_tf_session,
                tid=tid)
            predictions.update(lstm_predictions)

        # attempt to clear the session memory
        tf.keras.backend.clear_session()
        gc.collect()

        # sanity check that we have at least one model
        if len(predictions.keys()) == 0:
            raise ValueError("No models were passed in that could be ensembled.")

        out = vote(
            prediction_dict=predictions,
            num_classes=num_classes,
            num_tokens=num_tokens,
            voting_method=self.ensemble_metadata.voting_method,
            include_raw_probs=self.job_metadata.full_output,
            threshold_categories=self.ensemble_metadata.threshold_categories,
            threshold=self.ensemble_metadata.threshold,
            weights=self.ensemble_metadata.model_weights,
            model_type=self.ensemble_metadata.model_type,
            voting_model_folder=self.ensemble_metadata.voting_model_folder,
            model_cache=self.model_cache,
        )

        out.tokens = tokens['token']
        out.ranges = tokens['range']
        return out

    @property
    def feature_list(self):
        return self._feature_list

    @feature_list.setter
    def feature_list(self, value: list):
        self._feature_list = sorted(value)

    @property
    def testing_features(self) -> Set[FeatureType]:
        fs = set(self.feature_list)
        # add concept mapping features
        if self.concept_feature_mapping:
            for concept_feature in self.concept_feature_mapping:
                fs.add(concept_feature)
        return fs

    @property
    def feature_col_size(self):
        # NOTE(SF): fixing bug in ensemble testing with addition of data counter, tbd how to do this in the future
        return 2

    def ensemble_text_to_ann(self):
        matched_annotated_files = self.data_source.get_active_learning_jsons()
        for file in matched_annotated_files:
            res = self.predict(common.read_json(file))
            ensemble_result: List[BiomedOutput] = res.token_dict_list(label_enum=self.label_enum)
            id_num = 0
            ann_file_name = file.replace('.json', '.ann').replace(self.model_type.name, '')
            for ma_dir in self.data_source.active_feature_set_dirs:
                ann_file_name = ann_file_name.replace(ma_dir, self.data_source.active_ann_jira)

            if not os.path.isdir(os.path.dirname(ann_file_name)):
                os.makedirs(os.path.dirname(ann_file_name))
            operations_logger.info(f'writing ann to {ann_file_name}', tid=self.job_metadata.job_id)
            with open(ann_file_name, 'w') as f:
                for item in ensemble_result:
                    id_num += 1
                    text = item.text
                    index = item.range
                    label = item.label

                    new_line = f"T{id_num}\t{label.lower()} {index[0]} {index[1]}\t{text}\n"
                    f.write(new_line)
