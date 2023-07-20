from math import ceil
import os
from threading import Lock
from typing import List, Dict, Set

import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

from text2phenotype.common.featureset_annotations import MachineAnnotation, Vectorization
from text2phenotype.apiclients.feature_service import FeatureServiceClient
from text2phenotype.apm.metrics import text2phenotype_capture_span
from text2phenotype.common import common
from text2phenotype.common.log import operations_logger
from text2phenotype.constants.features import FeatureType
from text2phenotype.common.data_source import DataSourceContext
from text2phenotype.common.vector_cache import VectorCacheJson

from biomed.models.prediction_engine import PredictionEngine
from biomed.data_sources.data_source import BiomedDataSource
from biomed import RESULTS_PATH
from biomed.common.predict_results import PredictResults
from biomed.constants.constants import ModelType, EXCLUDED_LABELS
from biomed.models.callbacks import ClassMicroF1Score, TimerCallback
from biomed.models.data.generator import LSTMTestingDataGenerator, LSTMTrainingDataGenerator
from biomed.models.model_cache import ModelCache, ModelMetadataCache
from biomed.models.model_metadata import ModelMetadata
from biomed.models.model_report import ModelReportPlots
from biomed.models.model_worker import ModelWorker
from biomed.models.model_wrapper import ModelWrapper
from biomed.models.data_counts import DataCounter
from biomed.train_test.job_metadata import JobMetadata
from biomed.common.voting import vote_with_weight


class ModelBase(PredictionEngine):
    MODEL_METADATA_SUFFIX = '.metadata.json'

    def __init__(self,
                 model_metadata: ModelMetadata = None,
                 data_source: BiomedDataSource = None,
                 job_metadata: JobMetadata = None,
                 model_type: ModelType = None,
                 model_folder_name: str = None):
        """
        :param model_metadata: Any model's metadata may be provided, known or new
        :param data_source: Provided turing a training or testing build to describe the data to be used
        :param job_metadata: Provided during a training or testing build to desribe parameters for the job
        :param model_type: ModelType
        :param model_folder_name: folder for model data
        """
        self.model_metadata_cache = ModelMetadataCache()
        if model_folder_name:
            self.init_from_folder_name(model_folder_name, model_type)
        elif not model_metadata:
            raise ValueError('You must provide a model folder name  or metadata')
        else:
            self.model_metadata: ModelMetadata = model_metadata
            self.model_metadata.model_type = model_type or model_metadata.model_type
        super().__init__(
            model_type=self.model_metadata.model_type,
            job_metadata=job_metadata,
            data_source=data_source,
            binary_classifier=self.model_metadata.binary_classifier)

        self.use_tf_session = True  # default is to use TF1 Session, change this later
        self.model_cache: ModelCache = ModelCache()
        self._feature_size = None
        self.feature_service_client = FeatureServiceClient()

    @property
    def window_size(self):
        return self.model_metadata.window_size

    @property
    def testing_features(self) -> Set[FeatureType]:
        fs = self.training_features
        for concept_feature in self.concept_feature_mapping:
            fs.add(concept_feature)
        return fs

    def init_from_folder_name(self, model_folder_name=None, model_type: ModelType = None):
        if model_folder_name is None:
            raise ValueError("Trying to init model from folder name but folder name is None")
        if model_type is None:
            raise ValueError("Cannot initialize a model with no modeltype")

        #SORCE OF IOPS FROM SSS
        self.model_metadata = self.model_metadata_cache.model_metadata(
            model_type=model_type,
            model_folder=model_folder_name)
        self.model_metadata.model_type = model_type


    @property
    def training_features(self) -> Set[FeatureType]:
        feat_set = set(self.model_metadata.features)
        return feat_set

    @property
    def feature_col_size(self):
        if not self._feature_size:
            fs_client = FeatureServiceClient()
            vectors = fs_client.vectorize(tokens=MachineAnnotation(json_dict_input={'token': ['a'], 'range': [[0, 1]]}),
                                          features=self.model_metadata.features,
                                          tid=self.job_metadata.job_id)
            feature_size = 0
            for feature in self.model_metadata.features:
                if feature not in EXCLUDED_LABELS:
                    if vectors.defaults[feature.name]:
                        feature_size += len(vectors.defaults[feature.name])
                    else:
                        operations_logger.warning(f'Feature {feature.name} not found in vector defaults')
            self._feature_size = feature_size
        return self._feature_size

    @feature_col_size.setter
    def feature_col_size(self, value):
        """Shortcut to manual set on feature_col_size, escapes using FeatureService"""
        self._feature_size = value

    @property
    def num_label_classes(self):
        return len(self.label_enum) if not self.model_metadata.binary_classifier else 2

    @staticmethod
    def plot_model(model):
        """create a graphviz image of the model layers"""
        tf.keras.utils.plot_model(model, to_file="model_plot.png", show_shapes=True, show_layer_names=True)

    def get_non_excluded_feats(self) -> List[FeatureType]:
        return sorted(
            [feature for feature in sorted(self.model_metadata.features) if feature not in EXCLUDED_LABELS])

    @text2phenotype_capture_span()
    def slice_3d_mat(self, mat_3d, feature_col_mapping: Dict[FeatureType, range], tid: str = None):
        features = self.get_non_excluded_feats()
        cols = []
        for f in features:
            cols.extend(list(feature_col_mapping[f]))
        operations_logger.debug(f'Feature Size = {len(cols)}', tid=tid)
        return mat_3d[:, 0:self.model_metadata.window_size, tuple(cols)]

    @text2phenotype_capture_span()
    def predict(self,
                tokens: MachineAnnotation,
                vectors: Vectorization = None,
                mat_3d: np.ndarray = None,
                feature_col_mapping: dict = None,
                text: str = None,
                **kwargs) -> PredictResults:
        if mat_3d is None:
            primary_res = self.predict_from_token_vectors(tokens=tokens, vectors=vectors)
            num_tokens = len(tokens)
        else:
            primary_res = self.prediction_from_matrix_col_mapping(mat_3d=mat_3d,
                                                                  feature_col_mapping=feature_col_mapping)
            num_tokens = len(mat_3d)

        if kwargs.get("test_counts"):
            dcounter = kwargs.get("test_counts")
            dcounter.is_predict_count = True
            n_word_tokens = len(tokens.tokens)
            n_valid_tokens = len(tokens.valid_tokens()[1])
            doc_window_count = len(tokens) - self.window_size  # this will need to be adjusted for window stride
            doc_word_label_counts = None
            dcounter.add_document(n_word_tokens, n_valid_tokens, doc_word_label_counts, doc_window_count)

        voted_results = self.vote_helper(primary_res, num_tokens=num_tokens)

        return self.predict_helper(tokens=tokens, y_voted_weight_np=voted_results)

    def predict_from_token_vectors(self, tokens: MachineAnnotation, vectors: Vectorization):
        """
        Method used during individual testing
        :param tokens: machine annotation object, must have all annotationsfeatures used for model annotated
        :param vectors: vectorization object, must have ALL features used for model vectorized
        :return: 3-d matrix [num_tokens, window_size, label_enum_size]
        """
        mat_3d = self.get_mat_3d_test_generator(vectors=vectors,
                                                num_tokens=len(tokens['token']))
        out_shape = (
            max(mat_3d.num_tokens - self.model_metadata.window_size + 1, 1),
            self.model_metadata.window_size,
            self.num_label_classes)
        results = np.zeros(out_shape)
        idx = 0
        for i in range(len(mat_3d)):
            prediction = self.prediction_from_matrix_col_mapping(mat_3d=mat_3d[i],
                                                                 feature_col_mapping=mat_3d.feature_col_mapping)
            results[idx: idx + prediction.shape[0], :, :] = prediction
            idx += prediction.shape[0]
        return results

    @text2phenotype_capture_span()
    def prediction_from_matrix_col_mapping(self, mat_3d, feature_col_mapping: dict, tid: str = None):
        """
        NOTE: This is the function used by ensembler currently for the LSTM (the model worker calls this directly)
        :param mat_3d: feature matrix [num_tokens, window_size, feature_dim]
        :param feature_col_mapping: dictionary of feature type to which columns it occupies in the model matrix
        :return: matrix of dimension [num tokens, window size, num_label_classes]
        """
        model = self.get_cached_model()

        y_pred_prob = model.predict(
            self.slice_3d_mat(mat_3d=mat_3d, feature_col_mapping=feature_col_mapping, tid=tid),
            self.job_metadata.batch_size)
        # output is mat num tokens x window size x num labels
        return y_pred_prob

    def get_cached_model(self) -> ModelWrapper:
        return self.model_cache.model_keras(
            self.model_metadata.model_file_name,
            self.model_type,
            self.model_metadata.model_file_path,
            use_tf_session=self.use_tf_session,
        )

    def get_model_worker(self, lock: Lock, results: dict, use_tf_session: bool = True, tid: str = None):
        """
        :param lock: lock object used for multithreading
        :param results: input should be empty dictionary, after running will be model name: 3d predicted matrix
        :param use_tf_session: override class default for using tf.Session/Graph
        :return: Model  Worker object
        """
        self.use_tf_session = use_tf_session
        return ModelWorker(model_file_name=self.model_metadata.model_file_name,
                           prediction_function=self.prediction_from_matrix_col_mapping,
                           result_lock=lock,
                           results=results,
                           tid=tid)

    def _get_steps_per_epoch(self, token_count):
        return ceil(token_count / self.job_metadata.batch_size)

    def initialize_model(self):
        """
        Given the number of output classes and input feature dimensions, return Keras BiLSTM

        NOTE: this should be done in a derived class, giving flexibility of what type of model we may use

        :return: keras.Model
        """
        operations_logger.info('Initializing BiLSTM Model')
        num_classes = self.num_label_classes
        feature_dim = self.feature_col_size
        model_layers = []
        model_layers += [
            layers.InputLayer((self.model_metadata.window_size, feature_dim), dtype="float32")
        ]

        # the TF1 models use VarianceScaling as the weight initializer, while TF2 uses GlorotUniform
        # weight_initializer = tf.keras.initializers.VarianceScaling(
        #     scale=1.0, mode="fan_avg", distribution="uniform", seed=None
        # )
        weight_initializer = tf.keras.initializers.GlorotUniform()

        if self.job_metadata.add_dense:
            model_layers += [layers.TimeDistributed(
                layers.Dense(self.job_metadata.reduced_dim, activation='elu', kernel_initializer=weight_initializer))]
            feature_dim = self.job_metadata.reduced_dim

        model_layers += [
            layers.Bidirectional(
                layers.LSTM(
                    self.job_metadata.hidden_dim,
                    return_sequences=True,
                    kernel_initializer=weight_initializer,
                    activation="tanh",
                    dropout=self.job_metadata.dropout,
                    recurrent_dropout=self.job_metadata.dropout,
                    recurrent_activation="sigmoid",  # hard_sigmoid is a linear approximation
                    time_major=False,  # want this to be false, [batch, timesteps, feature]
                    dtype="float32"
                ),
                input_shape=(self.model_metadata.window_size, feature_dim), merge_mode='concat'
            ),
            layers.TimeDistributed(
                layers.Dense(num_classes, activation='softmax', kernel_initializer=weight_initializer)),
        ]

        model = Sequential(model_layers)

        optimizer = optimizers.Adam(learning_rate=self.job_metadata.learning_rate)

        loss_type = self._get_model_loss_type(num_classes)
        self.job_metadata.model_loss = loss_type

        # TODO: create SparseAccuracy metrics for model_base
        # self.__f1_score_metric_handler = SparseMicroF1ScoreNoNa(self.num_label_classes)
        metrics = [
            "accuracy",
            # SparseAccuracyNoPad(exclude_na=False),
            # SparseAccuracyNoPad(exclude_na=True, name="accuracy-nona"),
            # self.__f1_score_metric_handler,
        ]
        model.compile(loss=loss_type,
                      optimizer=optimizer,
                      metrics=metrics,
                      sample_weight_mode='temporal')
        operations_logger.debug('model compiled, beginning fit generator')
        return model

    def _get_model_loss_type(self, num_classes):
        # if there are just two classes use binary entropy
        loss_type = 'categorical_crossentropy' if num_classes > 2 else 'binary_crossentropy'
        operations_logger.info(f'Using loss type: {loss_type}', tid=self.job_metadata.job_id)
        return loss_type

    def _adjust_token_count(self, total_count, ann_files, fs_files):
        """
        Remove any duplicate tokens from the total token count over all documents

        :param total_count:
        :param ann_files:
        :param fs_files:
        :return:
        """
        dup_count = 0
        for ann_file, fs_file in zip(ann_files, fs_files):
            machine_ann = MachineAnnotation(json_dict_input=common.read_json(fs_file))
            duplicate_tokens = self.data_source.get_duplicate_token_idx(
                ann_file=ann_file, machine_annotations=machine_ann)
            dup_count += len(duplicate_tokens)

        return total_count - (len(ann_files) * (self.model_metadata.window_size - 1)) - dup_count

    def _get_doc_support_counts(self, dcounter: DataCounter, ann_files, fs_files):
        """
        Load a given DataCounter instance with the information from ann_files and fs_files
        This method is not to be used with BERT, which updates the counter in windowed_encodings

        This method should be a part of data_counter, since we are loading it!
        But it is using a few PredictionEngine methods to get token counts and labels.

        NOTE: these total counts may not match the "adjusted token count", which removes duplicate tokens
        for reasons that are still unclear.

        :param dcounter: DataCounter, a context specific counter for data
            Updated in place, so after the function call completes it will have the doc info
            TODO: this method really should be a DataCounter method
        :param ann_files: List[str]
            list of the human annotation files
        :param fs_files:
            list of the feature annotation files
        :return: None
            The DataCounter object is updated in place, so dont return it explicitly
        """
        for file_idx, ann_filename in enumerate(ann_files):
            tokens = self._read_annotation_file(fs_files[file_idx])
            doc_labels = self.token_true_label_list(ann_filename, tokens)
            n_word_tokens = len(tokens.tokens)
            n_valid_tokens = len(tokens.valid_tokens()[1])
            # NOTE: this is only true for BiLSTM models; BERT counts windows differently
            # this will need to be adjusted for window stride that isnt equal to 1
            doc_window_count = len(tokens) - self.model_metadata.window_size
            doc_word_labels = doc_labels or None

            # update the counter for each document
            dcounter.add_document(n_word_tokens, n_valid_tokens, doc_word_labels, doc_window_count)
        return

    def __get_generator(self, ann_files: List[str], fs_files: List[str], vector_cache: VectorCacheJson,
                        context: DataSourceContext):
        generator_class = LSTMTrainingDataGenerator if context == DataSourceContext.train else LSTMTestingDataGenerator

        return generator_class(ann_files, fs_files, vector_cache, self.feature_col_size, self.data_source,
                               self.label_enum,
                               self.job_metadata, self.model_metadata)

    def _train_model_on_data(self, model, epochs: int = None):
        if epochs is None:
            epochs = self.job_metadata.epochs

        os.makedirs(os.path.dirname(self.model_metadata.model_file_path), exist_ok=True)

        # TODO: add job specific cache, and removal of cache on exit
        train_vect_cache = VectorCacheJson(DataSourceContext.train)
        test_vect_cache = VectorCacheJson(DataSourceContext.testing)

        ann_files, fs_files = self.data_source.get_matched_annotated_files(self.label_enum)
        train_generator = self.__get_generator(ann_files, fs_files, train_vect_cache, DataSourceContext.train)
        total_token_count = train_generator.vectorize()
        adjusted_token_count = self._adjust_token_count(total_token_count, ann_files, fs_files)
        steps_per_epoch = self._get_steps_per_epoch(adjusted_token_count)
        operations_logger.info(
            f'Feature Dimension is: {self.feature_col_size}, Number of Classes is: {self.num_label_classes}, '
            f'Number of Steps Per Epoch: {steps_per_epoch}, total # of tokens is '
            f'{total_token_count}, adj. # of tokens is {adjusted_token_count}', tid=self.job_metadata.job_id)

        train_counts = DataCounter(
            self.label2id,
            n_features=self.feature_col_size,
            window_size=self.model_metadata.window_size,
            window_stride=self.model_metadata.window_stride,
        )
        # walk through all of the data in train/val datasets once, to get the support counts
        self._get_doc_support_counts(train_counts, ann_files, fs_files)
        self._add_context_support_metrics(train_counts, context=DataSourceContext.train)

        val_ts, validation_anns, validation_fs = None, None, None
        val_generator = None
        if not self.job_metadata.exclude_validation:
            validation_anns, validation_fs = self.data_source.get_matched_annotated_files(
                self.label_enum,
                context=DataSourceContext.validation)
            val_generator = self.__get_generator(validation_anns, validation_fs, test_vect_cache,
                                                 DataSourceContext.validation)
            val_ts = val_generator.vectorize()

            # collect the validation data support metrics
            validate_counts = DataCounter(
                self.label2id, n_features=self.feature_col_size,
                window_size=self.model_metadata.window_size,
                window_stride=self.model_metadata.window_stride,
            )
            self._get_doc_support_counts(validate_counts, validation_anns, validation_fs)
            self._add_context_support_metrics(validate_counts, context=DataSourceContext.validation)

        timer_callback = TimerCallback()

        operations_logger.info(">>> Starting model.fit()")
        if self.job_metadata.exclude_validation or not val_ts:
            operations_logger.debug('training model without validation dirs', tid=self.job_metadata.job_id)
            self._history = model.fit(
                train_generator.next_batch(),
                epochs=epochs,
                steps_per_epoch=steps_per_epoch,
                verbose=2,
                callbacks=[timer_callback]
            )
            model.save(self.model_metadata.model_file_path, save_format="h5")
        else:
            val_steps = self._get_steps_per_epoch(val_ts)
            metrics_callback = ClassMicroF1Score(
                val_steps,
                self.__get_generator(
                    validation_anns,
                    validation_fs,
                    test_vect_cache,
                    DataSourceContext.validation).next_batch(),
                self.model_metadata.model_file_path)

            operations_logger.info(f'validation steps: {val_steps}', tid=self.job_metadata.job_id)

            self._history = model.fit(
                train_generator.next_batch(),
                epochs=epochs,
                steps_per_epoch=steps_per_epoch,
                verbose=2,
                validation_data=val_generator.next_batch(),
                validation_steps=val_steps,
                callbacks=[metrics_callback, timer_callback])

        timer_output = timer_callback.get_durations_dict()
        self._history.history.update(timer_output)
        self.write_train_history()
        self.write_data_support_metrics()
        operations_logger.debug("Epoch durations: {}".format(timer_output[TimerCallback.EPOCH_DUR_KEY]))
        operations_logger.debug("Train duration: {}".format(timer_output[TimerCallback.TRAIN_DUR_KEY]))
        operations_logger.info('********* Training Done', tid=self.job_metadata.job_id)
        return model

    def train(self):
        """
        Train a bidirectional LSTM model and I2B2 training data set or other data set, and also save the model
        :return: The file path to the newly trained model
        """
        model = self.initialize_model()
        model = self._train_model_on_data(model)
        return self.save(model)

    def update_model(self):
        operations_logger.info(
            f"Beginning Update_Model, This method only runs over a single epoch, "
            f"Model Type: {self.model_type}, "
            f"Model File Name: {self.model_metadata.model_file_name}", tid=self.job_metadata.job_id)

        model = self.get_cached_model()
        # if updating a model that previously didnt have a sample weight mode, ensure that it has one
        if self.job_metadata.class_weight:
            model.model.sample_weight_mode = 'temporal'

        model = self._train_model_on_data(model, epochs=1)

        # making sure we write the new model with updated versioning
        self.model_metadata.previous_model_file_name = self.model_metadata.model_file_name
        self.model_metadata.update_model_file_name(self.job_metadata.job_id)
        self.model_metadata.update_model_file_path()
        operations_logger.info(f'Model Update complete, saving updated model to {self.model_metadata.model_file_path}')
        model.save(self.model_metadata.model_file_path)
        return self.save(model)

    def save(self, model: [Sequential, ModelWrapper], **kwargs):
        model_file_name = self.model_metadata.model_file_name
        json_path = os.path.join(RESULTS_PATH, f'{model_file_name}.json')
        operations_logger.debug(
            f"Saving model: {model_file_name}, Writing model file"
            f" to RESULTS_PATH: {RESULTS_PATH}", tid=self.job_metadata.job_id)
        # NOTE: we don't save the model itself here, that is done during or at the end of training
        common.write_json(model.to_json(), json_path)
        self.model_metadata.save()
        tf.keras.backend.clear_session()
        return self.model_metadata.model_file_path

    def write_train_history(self):
        """
        Write the train history dict to an output json file
        """
        if not self._history:
            operations_logger.debug("No train history found", tid=self.job_metadata.job_id)
            return
        report_file_name = f'train_history_{self.model_metadata.model_type.name}.json'
        file_path = os.path.join(RESULTS_PATH, self.job_metadata.job_id, report_file_name)
        common.write_json(self._history.history, file_path)
        operations_logger.debug(f"Wrote train history to {file_path}", tid=self.job_metadata.job_id)

        # TODO(mjp): invert this relationship, so write_train_history exist in ModelReportPlots
        report = ModelReportPlots(self.model_type.name, self.job_metadata.job_id)
        report.plot_train_metrics(self._history.history)
        operations_logger.debug(f"Wrote train history figures to {file_path}", tid=self.job_metadata.job_id)

    @staticmethod
    def predict_helper(y_voted_weight_np: np.ndarray, tokens: MachineAnnotation) -> PredictResults:
        return PredictResults(predicted_probs=y_voted_weight_np,
                              tokens=tokens.tokens,
                              ranges=tokens.range)

    def vote_helper(self, y_pred_prob: np.ndarray, num_tokens: int, tid: str = None):
        """
        wrapper for vote_with_weight
        :param y_pred_prob: array of predicted probabilities of dimension [num_sequences, window_size, num_classes]
        :param num_tokens:
        :param tid:
        :return:
        """
        operations_logger.debug('Started Voting with Weight now...', tid=tid)
        # if just one sequence, can just output that vector
        if y_pred_prob.shape[0] == 1:
            y_voted_weight_np = y_pred_prob[0]
        else:
            # shape of [num_sequences, window_size, num_classes]
            y_voted_weight_np = vote_with_weight(y_pred_prob,
                                                 num_tokens=num_tokens,
                                                 window_size=self.model_metadata.window_size)
        return y_voted_weight_np
