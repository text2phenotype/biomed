"""
Bert model base
Infrastructure inheriting from ModelBase, with additional methods for loading the BERT embeddings
and tokenizing the input MachineAnnotations to a BERT compatible format

Clinical BERT weights are downloaded from: https://github.com/EmilyAlsentzer/clinicalBERT


"""
import copy
import os
from math import ceil
import shutil
from threading import Lock
from typing import List, Union, Tuple, Dict
import random
from collections import defaultdict
from contextlib import redirect_stdout

import numpy as np
import tensorflow as tf
from transformers import (
    BertConfig,
    BertTokenizerFast,
    TFBertModel,
    TFBertForTokenClassification,
    TFBertMainLayer,
)

from text2phenotype.common.log import operations_logger
from text2phenotype.common.featureset_annotations import MachineAnnotation, Vectorization
from text2phenotype.common.data_source import DataSourceContext
from text2phenotype.common.vector_cache import VectorCacheJson
from text2phenotype.constants.environment import Environment
from text2phenotype.constants.features import FeatureType
from text2phenotype.apm.metrics import text2phenotype_capture_span

from biomed import RESULTS_PATH
from biomed.biomed_env import BiomedEnv
from biomed.common.predict_results import PredictResults
from biomed.common.voting import vote_with_weight
from biomed.constants.model_enums import BertEmbeddings
from biomed.models.bert_utils import BERT_PADDING_LABEL
from biomed.models.model_cache import get_full_path
from biomed.models.data_counts import DataCounter
from biomed.models.model_base import ModelBase
from biomed.models.losses import BertTokenClassificationLoss
from biomed.models.callbacks import TimerCallback
from biomed.models.model_worker import ModelWorker
from biomed.models.model_wrapper import ModelWrapper
from biomed.models.metrics import SparseMicroF1ScoreNoNa, SparseAccuracyNoPad
from biomed.models.data.generator import BERTGenerator, chunk_generator


# resource pointer to absolute path for bert embedding base models
EMBEDDINGS_PATH = os.path.join(BiomedEnv.BIOM_MODELS_PATH.value, 'resources/files/bert_embedding')


def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in ``random``, ``numpy``, and/or ``tf`` (if
    installed).

    Args:
        seed (:obj:`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


class BertBase(ModelBase):
    """
    Base class for token classification models using BERT embeddings

    """
    # Fixed length for subtoken window length, used as input to bert embedding layer
    # This needs to be at least twice as long as the window_size, to avoid truncation errors
    # Max value is 512, dictated by Bert himself.
    BERT_MAX_WINDOW_LENGTH = 512

    # The label to use for [CLS], [SEP], [PAD], and sometimes the extended subtokens
    # eg, ["myo", "##card", "##ial"] could be labeled [2, -100, -100]
    # Used to ignore subtokens in loss calculation and metrics
    PADDING_LABEL = BERT_PADDING_LABEL

    # the fixed filename saved by TFBertForTokenClassification.save_pretrained()
    MODEL_FILENAME: str = "tf_model.h5"

    def __init__(self, *args, **kwargs):
        """Takes all the same arguments as ModelBase, add additional args if necessary"""
        super().__init__(*args, **kwargs)

        if not self.model_metadata.embedding_model_name:
            # overwrite the default embedding name (None) from ModelBase using the value in the config
            # ModelBase (super) overwrites model_metadata.embedding_model_name, so we need to overwrite it here
            embedding_name = getattr(kwargs["model_metadata"], "embedding_model_name")
        else:
            # loading a model that has already been trained, should have an embedding model_name
            embedding_name = self.model_metadata.embedding_model_name
        if not embedding_name or embedding_name.lower() not in BertEmbeddings._member_names_:
            msg = (
                "ModelMetadata.embedding_model_name MUST be set when using BertBase; "
                f"found {self.model_metadata.embedding_model_name}, expected one of: {BertEmbeddings._member_names_}"
            )
            raise ValueError(msg)
        self.model_metadata.embedding_model_name = embedding_name  # do the overwrite
        self.embedding_model_name = BertEmbeddings[self.model_metadata.embedding_model_name.lower()].value

        # set the model_name to be fixed, overwriting the versioned name from default ModelMetadata
        self.model_metadata.model_file_name = os.path.join(
            os.path.dirname(self.model_metadata.model_file_name), self.MODEL_FILENAME
        )

        self._tokenizer = self.load_tokenizer(self.embedding_model_name)
        self.bert_layer_ix = None  # which model layer contains the Huggingface bert embedding layer

        # location to store cached input files to the model train/test
        self.cache_root = (
            os.path.join("/tmp", self.job_metadata.job_id) if self.job_metadata.job_id else "/tmp"
        )
        self.feature_service_client = None
        self.use_tf_session = True  # bert default is to use eager execution

    @property
    def model_input_fields(self):
        return ["input_ids", "attention_mask"]  # add "valid_token_mask"

    @staticmethod
    def load_bert_embedding_model(pretrained_name, cache_root: str = EMBEDDINGS_PATH):
        """
        Uses WordPiece model to create fixed-size vocab of individual characters, subwords, and words
        Fixes vocab size to <30k

        :param pretrained_name: str, model folder name to load from `cache_root`
        :param cache_root: str
        :return: Tuple[BertTokenizerFast, TFBertModel]
        """
        cache_path = os.path.join(cache_root, pretrained_name)
        config = BertConfig.from_pretrained(cache_path)  # this config is available in model.config
        embedding_model = TFBertModel.from_pretrained(cache_path, config=config)

        return embedding_model

    def load_bert_token_classifier(self, pretrained_name, embeddings_path: str = EMBEDDINGS_PATH):
        """
        Get the basic 🤗 Transformers Bert for Token Classification model, with specified embedding source weights

        :param pretrained_name: str, folder name in cache_root to load embedding weights from
        :param embeddings_path: The folder to look for the embeddings model,
            defaults to `biomed/resources/files/bert_embedding`
        :return: transformers.TFBertForTokenClassification
        """
        cached_bert_path = os.path.join(embeddings_path, pretrained_name)
        # NOTE: uses default BertConfig
        model = TFBertForTokenClassification.from_pretrained(cached_bert_path, num_labels=self.num_label_classes)
        operations_logger.info(f"Loaded BERT encodings from: {pretrained_name}")
        # set job config params for the model
        model.config.hidden_dropout_prob = self.job_metadata.dropout
        return model

    @staticmethod
    def load_tokenizer(pretrained_name: str, embeddings_path: str = EMBEDDINGS_PATH):
        """
        Get the tokenizer associated with the target embeddings model

        :param pretrained_name: str, folder name in cache_root to load tokenizer from
        :param embeddings_path: The folder to look for the embeddings model,
            defaults to `biomed/resources/files/bert_embedding`
        :return: transformers.BertTokenizerFast
        """
        cached_tokenizer_path = os.path.join(embeddings_path, pretrained_name)
        return BertTokenizerFast.from_pretrained(cached_tokenizer_path)

    def load_model(self):
        """
        This is not used anywhere currently
        Load a model from the model code in initialize_model() and the saved weights

        NOTE: this may not work in a production context!!! only tested locally
        NOTE: This will break if the model structure in initialize_model() does not match the weights

        :return: tf.keras.Model
        """
        # ow. this is super hacky.
        model_file_name = self.model_metadata.model_file_name
        model_folder = os.path.dirname(get_full_path(self.model_type.name, model_file_name))
        try:
            # try loading from dvc'd folder
            # check how we are loading model_file_path if only testing?
            # model.load_weights(model_file_name)
            model = TFBertForTokenClassification.from_pretrained(model_folder)
        except OSError as e:
            operations_logger.info(f"Couldn't find the file in the resources model_folder: {model_folder}")
            model_folder = os.path.join(RESULTS_PATH, self.job_metadata.job_id)
            operations_logger.info(f"Trying to load model from: {model_folder}")
            # model_file_name = common.get_file_list(model_path, '.h5')[-1]
            # model.load_weights(model_file_name)
            model = TFBertForTokenClassification.from_pretrained(model_folder)
        operations_logger.info(f"Loaded TFBertForTokenClassification from: {model_folder}")

        return model

    def initialize_model(self):
        """
        Return Huggingface TFBertForTokenClassification

        :return: tf.keras.Model
        """
        operations_logger.info("Initializing TFBertForTokenClassification model")

        # create the Named Entity Recognition model, based on the target embedding transformer
        # model = self._create_ner_classifier_model(self._transformer_model)

        model = self.load_bert_token_classifier(self.embedding_model_name)

        self.bert_layer_ix = self.__locate_bert_layer(model)
        model.layers[self.bert_layer_ix].trainable = self.job_metadata.train_embedding_layer

        # TODO: Look into using BertAdam, which includes a warm up period for the optimizer
        # may only be for pytorch? from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.job_metadata.learning_rate)

        # categorical_crossentropy
        loss_type = self._get_model_loss_type(self.num_label_classes)
        # update the model_loss parameter in job_metadata for downstream recording
        self.job_metadata.model_loss = loss_type

        self.__loss_fn = BertTokenClassificationLoss.compute_loss
        # self.__loss_fn = model.compute_loss

        # save as private instance var so name can be used in the ModelCheckpoint method
        self.__f1_score_metric_handler = SparseMicroF1ScoreNoNa(self.num_label_classes)
        metrics = [
            SparseAccuracyNoPad(exclude_na=False),
            SparseAccuracyNoPad(exclude_na=True, name="accuracy-nona"),
            self.__f1_score_metric_handler,
        ]
        model.compile(
            loss=self.__loss_fn,
            optimizer=optimizer,
            metrics=metrics,
            sample_weight_mode="temporal",
        )
        operations_logger.debug("model compiled")
        return model

    @staticmethod
    def __locate_bert_layer(model):
        """
        Get the index of the model layer that contains the bert embeddings
        :param model: keras.Model
        :return: int, index of the layer in model.layers that contains the embeddings
        :raises: ValueError if more than one bert embedding layer found, or if no bert embedding layer found
        """
        # locate bert layer
        bert_layer_ix = [
            i for i, layer in enumerate(model.layers) if isinstance(layer, (TFBertMainLayer, TFBertModel))
        ]
        if len(bert_layer_ix) > 1:
            msg = f"Found more than one TFBertModel in model layers: {bert_layer_ix}"
            operations_logger.error(msg)
            raise ValueError(msg)
        elif not bert_layer_ix:
            msg = (
                "Zero TFBertModel layers found in model layers; are you sure you're using the correct model?"
            )
            operations_logger.error(msg)
            raise ValueError(msg)
        else:
            bert_layer_ix = bert_layer_ix[0]
        return bert_layer_ix

    # def _create_ner_classifier_model(self, bert_embedding_model):
    #     """
    #     Create a keras model (functional) with time distributed dense layer following the embedding
    #
    #     TODO: Future work can add additional dense layers to improve model performance
    #
    #     TODO: Future work can add another input of older features (eg sections) following the embedding layer
    #
    #     :param bert_embedding_model: TFBertModel
    #     :return: tf.keras.Model
    #     """
    #     # NOTE: try using TFBertForTokenClassification, since we are using a derived bert embedding?
    #     # It would allow us to use model.save, rather than model.save_weights.
    #
    #     # these window lengths will need to be matched to the WordPiece window token length, from windowed_encodings
    #     input_ids_in = tf.keras.layers.Input(
    #         shape=(self.BERT_MAX_WINDOW_LENGTH,), name="input_token", dtype="int32"
    #     )
    #     input_masks_in = tf.keras.layers.Input(
    #         shape=(self.BERT_MAX_WINDOW_LENGTH,), name="masked_token", dtype="int32"
    #     )
    #     # the first value from the model tuple is the last layer of the embedding model
    #     # TODO: try using positional input for masks, and kwarg input for input_ids
    #     embedding_layer_output = bert_embedding_model(input_ids_in, attention_mask=input_masks_in)[0]
    #     # TODO: add option to use vectorized features concatenated with the embedding layer output
    #     # TODO: try to not use the BiLSTM for classification
    #     layer_out = tf.keras.layers.Bidirectional(
    #         tf.keras.layers.LSTM(
    #             self.job_metadata.hidden_dim,
    #             return_sequences=True,
    #             kernel_initializer=tf.keras.initializers.GlorotUniform(),
    #             dropout=self.job_metadata.dropout,
    #             recurrent_dropout=self.job_metadata.dropout,
    #             recurrent_activation="sigmoid",  # sigmoid|hard_sigmoid, hard_sigmoid is a faster linear approximation
    #             dtype="float32",
    #         )
    #     )(embedding_layer_output)
    #
    #     # TODO: add layer that drops the masked tokens
    #
    #     # # add an extra hidden layer for the classification task, if desired
    #     # layer_out = tf.keras.layers.Dense(self.job_metadata.hidden_dim, activation='relu')(embedding_layer_output)
    #     # # TODO: add batch normalization before dense layer?
    #     # layer_out = tf.keras.layers.Dropout(0.2)(layer_out)
    #     layer_out = tf.keras.layers.TimeDistributed(
    #         tf.keras.layers.Dense(
    #             self.num_label_classes,  # + 1,  # add the padding label
    #             activation="sigmoid",
    #             kernel_initializer=tf.keras.initializers.GlorotUniform(),
    #         )
    #     )(layer_out)
    #     # layer_out = tf.keras.layers.Dense(
    #     #     self.num_label_classes,  # add the padding label
    #     #     activation="sigmoid",
    #     #     kernel_initializer=tf.keras.initializers.GlorotUniform(),
    #     # )(embedding_layer_output)
    #     # we use functional model instead of Sequence because has multiple inputs
    #     model = tf.keras.Model(inputs=[input_ids_in, input_masks_in], outputs=layer_out)
    #     return model

    def save(self, model: [tf.keras.models.Model, ModelWrapper], **kwargs):
        model_results_folder = os.path.join(
            RESULTS_PATH, os.path.dirname(self.model_metadata.model_file_name)
        )
        os.makedirs(model_results_folder, exist_ok=True)

        # TODO: is this saving "model_metadata.json" as well as "tf_model.metadata.json"?
        self.model_metadata.save()  # expects files in model_file_path

        summary_text_file = os.path.join(model_results_folder, f"model_summary_{self.model_type.name}.txt")
        with open(summary_text_file, "w") as f:
            with redirect_stdout(f):
                model.summary()

        # Because model is subclassed, cannot use the following methods or attributes:
        # model.to_yaml(), model.to_json(), model.get_config(), model.save()
        # model.inputs and model.outputs
        # https://stackoverflow.com/questions/51806852/cant-save-custom-subclassed-model
        #
        operations_logger.info(
            f"Saving model: {self.model_metadata.model_file_name}: "
            f" writing model file to RESULTS_PATH: {model_results_folder}",
            tid=self.job_metadata.job_id,
        )

        # save the huggingface tf_model.h5
        model.save_pretrained(model_results_folder)
        # remove the checkpoint file, since it isnt being used and is 3x the saved model size
        # TODO(mjp): figure out how to convert checkpoint into weights for an .h5 file
        checkpoint_dir = os.path.join(model_results_folder, "tf_model")
        if os.path.isdir(checkpoint_dir):
            shutil.rmtree(checkpoint_dir)
        tf.keras.backend.clear_session()
        return model_results_folder

    def get_cached_model(self) -> ModelWrapper:
        """
        Load a singleton cached version of model, wrapped in the ModelWrapper interface
        :return: ModelWrapper
        """
        # model_file_name = os.path.dirname(self.model_metadata.model_file_name)
        model_file_name = self.model_metadata.model_file_name
        return self.model_cache.model_keras(
            model_file_name,
            self.model_metadata.model_type,
            self.model_metadata.model_file_path,
            use_tf_session=self.use_tf_session,
        )

    def get_model_worker(self, lock: Lock, results: dict, use_tf_session: bool = True, tid: str = None):
        """
        Returns a threadable ModelWorker, pointing to a target method the worker will run, `predict`
        :param lock: lock object used for multithreading
        :param results: input should be empty dictionary, after running will be model name: 3d predicted matrix
        :param use_tf_session: override class default for using tf.Session/Graph
        :return: Model  Worker object
        """
        self.use_tf_session = use_tf_session
        return ModelWorker(model_file_name=self.model_metadata.model_file_name,
                           prediction_function=self.predict_from_tokens,
                           result_lock=lock,
                           results=results,
                           tid=tid)

    def predict(self, tokens: MachineAnnotation, **kwargs):
        """
        Wrapper around predict from tokens
        Required abstract method from PredictionEngine

        :param tokens: MachineAnnotation input
        :param kwargs: passthrough
        :return: PredictResults object
        """
        y_pred_prob = self.predict_from_tokens(tokens.tokens, **kwargs)
        return PredictResults(predicted_probs=y_pred_prob)

    def predict_from_tokens(self, tokens: List[str], **kwargs):
        """
        Predict label probabilities from list of tokens
        :param tokens: List[str]
            list of tokenized words in document
        :param kwargs: to allow for kwargs from model base to be passed in,
            eg test_counts:DataCounter
        :return: np.ndarray
            prediction array, shape (n_tokens, n_labels)
        """
        test_counts: DataCounter = kwargs.get('test_counts')
        if test_counts:
            test_counts.is_predict_count = True
        y_prob_out_list = []
        chunk_size = Environment.BIOMED_MAX_DOC_WORD_COUNT.value
        # chunk_generator returns generator chunks from the full tokens list, without lookahead
        # tokens_chunk is a generator itself, so needs to be evaluated when it is used (eg wrap with list())
        for tokens_chunk in chunk_generator(tokens, chunk_size):

            doc_dict = {"tokens": list(tokens_chunk)}  # don't use word_token_labels or valid_tokens
            encoded_doc_dict = self.windowed_encodings(doc_dict, test_counts)
            # isolate target fields & predict
            x_input_dict = {name: encoded_doc_dict[name] for name in self.model_input_fields}
            # get prediction
            y_pred_prob = self.return_prediction(x_input_dict, encoded_doc_dict["valid_token_mask"])
            y_prob_out_list.append(y_pred_prob)
        y_probs = np.vstack(y_prob_out_list)
        return y_probs

    def num_batches_by_samples(self, n_samples):
        """
        Calculate the number of batches for the given number of samples
            (docs, windows, tokens, whatever a sample is to user)
        Because batches are set by file, this should be calculated per document and then summed for
        total batch count
        """
        n_batches = int(np.ceil(n_samples / self.job_metadata.batch_size))
        return n_batches

    @text2phenotype_capture_span()
    def return_prediction(self, x_input: dict, token_start_mask: List[List[bool]], tid: str = None):
        """
        Takes encoded subtokens dict and use the model to return the weighted prediction matrix
            input_dict = dict('input_ids': List[List], 'attention_mask': List[List], 'token_start_mask': List[List])
        :return:
        """
        model = self.get_cached_model()
        # model = self.load_model()

        y_probs = []

        n_windows = len(x_input["input_ids"])
        window_indexes = list(range(n_windows))
        batch_size = self.job_metadata.batch_size
        n_batches = self.num_batches_by_samples(n_windows)
        for batch_ix in range(n_batches):
            batch_window_ix = window_indexes[batch_ix * batch_size: batch_ix * batch_size + batch_size]
            x_dict = {name: np.array(x_input[name])[batch_window_ix, ...] for name in self.model_input_fields}

            # using the call method for the model bc the predict method doesnt like receiving a dict as input
            logits = model(x_dict, training=False)[0]
            tf.keras.backend.clear_session()

            y_pred_prob_subtoken = tf.nn.softmax(logits).numpy()
            y_probs.append(y_pred_prob_subtoken)

        # if we are using strides, this is where we'd vote
        y_probs = np.vstack(y_probs)
        # this mask flattens the inputs into a single sequence, (n_input_tokens, n_labels)
        y_pred_valid_probs = y_probs[token_start_mask, :]
        return y_pred_valid_probs

    def train(self):
        """
        Train an LSTM token classifier model, and save the model
        :return: The file path to the newly trained model
        """
        operations_logger.info("***** Starting train()")
        if self.job_metadata.random_seed:
            set_seed(self.job_metadata.random_seed)
        model = self.initialize_model()
        model = self._train_model_on_data(model)

        # TODO: take the model saved in the checkpoint and use that as our primary saved model, rather than final epoch
        model_folder_path = self.save(model)
        return model_folder_path

    def _get_doc_data(self, fs_file_path, ann_file_path=None) -> Tuple[MachineAnnotation, List[int]]:
        """
        Read a feature service and human annotation file and return the input vectors
        Hopefully, the word token labels and machine_annotation tokens are aligned

        :param fs_file_path: str, absolute path to the target feature service file
        :param ann_file_path: str, absolute path to the target human annotation file
        :return: Tuple[MachineAnnotation, List[int]]
            MachineAnnotation from the FeatureService file
            list of token label ids as ints for each token in MachineAnnotation
        """
        machine_annotation = self._read_annotation_file(fs_file_path)
        word_token_labels = None
        if ann_file_path is not None:
            word_token_labels = self.token_true_label_list(ann_file_path, machine_annotation)

        return machine_annotation, word_token_labels

    def _text_to_features_cached(
        self,
        ann_files: List[str],
        fs_files: List[str],
        max_failure_pct: float,
        vector_cache: VectorCacheJson,
        data_counter: DataCounter = None,
        context: Union[DataSourceContext, str] = DataSourceContext.train,
    ):
        """
        Vectorize a collection of text files, store the results in vector_cache, keyed by fs_file

        TODO: replace this with TFTokenClassificationDataset

        :param fs_files: list of raw text file paths, used for the cache key
        :param ann_files: list of ann file paths
        :param max_failure_pct: The maximum percent of file failures before stopping processing.
        :param data_counter: Object that keeps track of how much data we are using for this dataset
            A newly initialized counter is passed in, and counts are updated internally
        :param vector_cache: VectorCache opject that will handle all your caching needs
        :return: None
        """
        # fs_client = FeatureServiceClient()
        max_failures = ceil(max_failure_pct * len(fs_files))
        num_failures = 0

        for fs_file_path, ann_file_path in zip(fs_files, ann_files):
            doc_dict = None
            try:
                machine_annotation, word_token_labels = self._get_doc_data(fs_file_path, ann_file_path)
                valid = machine_annotation.valid_tokens()

                # add additional feature vectors here
                # vectors: Vectorization = self.get_training_vectors(machine_annotation, fs_client=fs_client)

                doc_dict = {
                    "tokens": machine_annotation.tokens,
                    "token_ranges": machine_annotation.range,
                    "word_token_labels": word_token_labels,
                    "valid_tokens": valid[0],
                    # "feature_vectors": vectors.to_dict(),
                }

            except Exception:
                operations_logger.info(f"Failed to preprocess file: {fs_file_path}")
                num_failures += 1
                if num_failures >= max_failures:
                    raise ValueError(
                        f"{num_failures} files failed to be vectorized "
                        f"({max_failures} max allowed), destroying job"
                    )

            input_features_dict = self.windowed_encodings(doc_dict, data_counter)

            # cache the inputs
            vector_cache[fs_file_path] = input_features_dict

        operations_logger.info(
            f"[{str(context)}] Completed vectorization and caching of {len(fs_files)} files"
        )

    @staticmethod
    def get_doc_encodings(tokenizer, tokens, max_length):
        """
        Given a list of tokens, return the WordPiece expanded subtoken ids and associated attention_mask

        :param tokenizer: transformers.tokenization_bert_fast.BertTokenizerFast
        :param tokens: List[str]
        :param max_length: int, target window length for subtoken windows. MUST be <=512
        :return: transformers.BatchEncoding
            BatchEncoding object, attributes are accessible as a named tuple or a dict
            Current settings return the following attributes:
                ['input_ids', 'token_type_ids', 'attention_mask', 'offset_mapping', 'overflow_to_sample_mapping']

            Each attribute will be a list of lists (there are no ndarrays)
        """
        # TODO: add stride here
        encodings = tokenizer(
            tokens,
            is_split_into_words=True,
            padding='max_length',  # will pad sequence to length of max_length parameter
            truncation=True,
            max_length=max_length,
            return_offsets_mapping=True,
            return_overflowing_tokens=True,
        )
        return encodings

    @staticmethod
    def find_valid_subtoken_mask(subtoken_offset_mapping) -> List[List[bool]]:
        """
        Return a list of lists containing booleans, where True for the first subtoken in each full word token
        Sets special subtokens as False, eg [PAD], [CLS], [SEP]

        Example:
        subtoken words may look like ['[CLS]',  'in',  'ex', '##tre', '##mit', '##ies', '.', "[SEP]"]
        The subtoken offset mapping may look like [[(0,0), (0, 2), (0, 2), (2, 5), (5,8), (8,11), (0, 1), (0,0)]]
        And the resulting mask would be [[False, True, True, False, False, False, True, False]]

        :param subtoken_offset_mapping: List[List[Tuple[int, int]]]
            A List over the document windows, with each window as a list of tuples for each whole word offset
            Note that this requires a list of lists; a single list will not work as expected
        :return: List[List[bool]]
        """
        token_start_mask = []
        for i, doc_window_offset in enumerate(subtoken_offset_mapping):
            arr_offset = np.array(doc_window_offset)

            # special tokens have 0 for start and 0 for end offsets
            special_token_mask = (arr_offset[:, 0] == 0) & (arr_offset[:, 1] == 0)
            # any subtokens beyond the first one for an extended token won't start with 0
            extra_subtoken_mask = arr_offset[:, 0] != 0
            # mask where True for all labeled word tokens start
            doc_token_start_mask = ~(extra_subtoken_mask | special_token_mask)
            # we want this mask if we are doing "same" labels, rather than "pad"
            # and to trim down the resulting logits from the bert model
            token_start_mask.append(doc_token_start_mask.tolist())
        return token_start_mask

    def encode_doc_subtoken_labels(self, label_ids: List[int], valid_subtoken_mask: List[List[bool]]):
        """
        For a list of document token labels and the tokenized windowed encoded_doc,
        return the windowed labels and the aligned mask of word token start elements

        :param label_ids: List[int]
            A 1D List of all token label ids for a given doc.
        :param valid_subtoken_mask: List[List[bool]]
            A list of windows (list) and booleans for whether or not
            the token is the first token of an expanded set of subtokens, generated via `find_valid_subtoken_mask`

        :return: List of Lists
            Contains the windowed label_ids associated with each subtoken
                Subtokens following the first subtoken in a whole word are given the label -100 (from self.PADDING_LABEL)
                Special tokens are set to self.PADDING_LABEL (-100)
                NOTE: this could lead to unequal number of whole word tokens in each window, which np.array does not like
        """
        encoded_labels = []
        label_ix = 0
        for i, doc_token_start_mask in enumerate(valid_subtoken_mask):
            # create an empty array of -100; this is the expected value by the classifier and loss function
            window_enc_labels = np.ones(len(doc_token_start_mask), dtype=int) * self.PADDING_LABEL
            n_window_labels = sum(doc_token_start_mask)

            # if label_ids slice is larger than Trues in token_start_mask, should automatically trim
            # if label_ids slice is smaller than Trues in token_start_mask, WE MAY HAVE A PROBLEM
            # TODO: test what happens when we somehow end up with more labels than valid matches with the start mask
            window_enc_labels[doc_token_start_mask] = label_ids[label_ix: label_ix + n_window_labels]

            encoded_labels.append(window_enc_labels.tolist())
            label_ix += n_window_labels
        return encoded_labels

    @staticmethod
    def sanitize_tokens(tokens: list):
        # problems exist within the hugging face tokenizer when the length of the token != len utf encoded tokens
        sanitized_tokens = copy.deepcopy(tokens)
        for idx in range(len(sanitized_tokens)):
            if len(sanitized_tokens[idx]) != len(sanitized_tokens[idx].encode('utf-8')):
                sanitized_tokens[idx] = '-' * len(sanitized_tokens[idx]) # replacement char is arbitrary punctuation
        return sanitized_tokens

    def windowed_encodings(self, doc_dict: dict, dcounter: DataCounter = None, window_size=None) -> Dict:
        """
        Read in list of cache keys (eg fs_file paths) and the corresponding vector cache, and
        return a tensorflow dataset

        NOTE: this returns the WHOLE dataset, and will not work for datasets that dont fit in memory

        :param doc_dict: Dict
            Requires fields: 'tokens', 'valid_tokens', optional fields "word_token_labels"
            TODO: take InputDoc object as input, rather than a dict
        :param dcounter: DataCounter instance, updated in place with new doc counts
        :param window_size: (optional) if provided, will use this window size over the model metadata
        :return: dict
             "input_ids", "attention_mask", "encoded_labels", "valid_token_mask"

        """
        dataset_dict = defaultdict(list)
        if window_size is None:
            window_size = self.model_metadata.window_size

        # TODO: doc_dict should be an InputExample object
        doc_tokens = self.sanitize_tokens(doc_dict["tokens"])
        doc_labels = doc_dict.get("word_token_labels")  # list of ints, or None if we dont have labels
        # doc_valid_tokens is a list of valid tokens, taken from machine_annotation.valid_tokens()[0]
        doc_valid_tokens = doc_dict.get("valid_tokens")
        # feature_vectors = cache_dict["feature_vectors"]

        doc_encodings = self.get_doc_encodings(
            self._tokenizer, doc_tokens, max_length=window_size
        )

        valid_token_mask = self.find_valid_subtoken_mask(doc_encodings["offset_mapping"])

        if doc_labels:
            encoded_labels = self.encode_doc_subtoken_labels(doc_labels, valid_token_mask)
        else:
            encoded_labels = None

        # update the feature vectors so they fit in a windowed 3D matrix
        # TODO: add feature vector reshaping here

        # Collect document counts
        if dcounter:
            num_word_tokens = len(doc_tokens)
            n_valid_tokens = len(doc_dict["valid_tokens"]) if doc_valid_tokens else None
            doc_window_count = len(doc_encodings["input_ids"])
            word_labels = doc_labels or None
            dcounter.add_document(num_word_tokens, n_valid_tokens, word_labels, doc_window_count)

        # extend to the dataset dict; removes distinction of doc, but keeps windows distinct for a doc
        # TODO: use the InputFeatures object here instead of a dict
        dataset_dict["input_ids"] = doc_encodings.input_ids
        dataset_dict["attention_mask"] = doc_encodings.attention_mask
        dataset_dict["encoded_labels"] = encoded_labels
        dataset_dict["valid_token_mask"] = valid_token_mask

        return dataset_dict

    @staticmethod
    def sample_weights_from_labels(labels, label2ix, class_weights_dict):
        """
        Create list of weights for each token index in labels

        :param labels: List[str]
        :param label2ix: Dict[str, int]
            dictionary converts string label name to label index
        :param class_weights_dict: Dict[int, float]
            dictionary for each label index to the associated weighting for that label
        :return: List[float
        """
        # create the sample weights
        sample_weights = [
            [class_weights_dict[label2ix[lbl]] for lbl in window_label] for window_label in labels
        ]
        return sample_weights

    @staticmethod
    def _cache_field_to_extended_list(field_name: str, vector_cache: VectorCacheJson, key_list: List[str]):
        """
        Concatenate the field values (often lists) for each doc over all docs

        :param field_name: field to concatenate on
        :param vector_cache: cache object
        :param key_list: doc keys to iterate through
        :return: List[Any]
        """
        new_list = []
        for cache_key in key_list:
            new_list.extend(vector_cache[cache_key][field_name])
        return new_list

    def _train_model_on_data(self, model, epochs: int = None):
        """

        TODO: load files as InputExample and tokenize as InputFeatures
        TODO: use utils_ner for data loading
        TODO: split the data loading out from the train method, use a DatasetGenerator

        :param model: keras.Model
        :param epochs: number of epochs to iterate over
        :return:
        """
        epochs = epochs or self.job_metadata.epochs
        os.makedirs(os.path.dirname(self.model_metadata.model_file_path), exist_ok=True)

        # create the local vector caches
        train_vector_cache = VectorCacheJson(DataSourceContext.train.value, cache_root=self.cache_root)
        val_vector_cache = VectorCacheJson(DataSourceContext.validation.value, cache_root=self.cache_root)

        # match the feature files and ann files
        train_ann_files, train_fs_files = self.data_source.get_matched_annotated_files(
            self.label_enum, DataSourceContext.train
        )
        train_counts = DataCounter(
            self.label2id,
            n_features=self.feature_col_size,
            window_size=self.model_metadata.window_size
        )

        # cache the tokenized docs per fs_file key
        self._text_to_features_cached(
            train_ann_files,
            train_fs_files,
            self.job_metadata.max_train_failures_pct,
            train_vector_cache,
            data_counter=train_counts,
            context=DataSourceContext.train,
        )

        # create a generator-based dataset
        train_generator = BERTGenerator(
            train_fs_files,
            train_vector_cache,
            self.model_input_fields,
            self.model_metadata.window_size,
            class_weight=self.job_metadata.class_weight,
            shuffle_files=True)
        # training workflow expects batch-wise input, shuffle is generally good practice
        train_data = train_generator.generator.shuffle(Environment.BIOMED_MAX_DOC_WORD_COUNT.value).batch(self.job_metadata.batch_size)
        val_data = None

        self._add_context_support_metrics(train_counts, DataSourceContext.train)

        if not self.job_metadata.exclude_validation:
            val_ann_files, val_fs_files = self.data_source.get_matched_annotated_files(
                self.label_enum, context=DataSourceContext.validation
            )
            validation_counts = DataCounter(
                self.label2id, n_features=self.feature_col_size, window_size=self.model_metadata.window_size
            )
            self._text_to_features_cached(
                val_ann_files,
                val_fs_files,
                self.job_metadata.max_train_failures_pct,
                val_vector_cache,
                data_counter=validation_counts,
                context=DataSourceContext.validation,
            )

            # same as above with train
            val_generator = BERTGenerator(
                val_fs_files,
                val_vector_cache,
                self.model_input_fields,
                self.model_metadata.window_size,
                class_weight=self.job_metadata.class_weight,  # not needed?
            )
            val_data = val_generator.generator.batch(self.job_metadata.batch_size)

            self._add_context_support_metrics(validation_counts, DataSourceContext.validation)

        # set callbacks
        timer_callback = TimerCallback()
        checkpoint_path = os.path.splitext(self.model_metadata.model_file_path)[0]
        # TODO: at end of training, save callback output weights as true model output
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(checkpoint_path, "checkpoints"),
            monitor="val_f1_score_no_na",
            mode="max",
            save_weights_only=False,
            save_best_only=True,
        )
        # TODO: can I add tensorboard to a remote worker? Where would be good to save the tf logs?
        # need to start a tensorboard worker on the remote, which is usually a separate command in cli
        # tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
        callbacks = [
            timer_callback,
            model_checkpoint_callback,
            # tensorboard_callback,
        ]

        self._history = model.fit(
            train_data,
            epochs=epochs,
            batch_size=self.job_metadata.batch_size,
            validation_data=val_data,
            callbacks=callbacks,
            verbose=2
        )

        timer_output = timer_callback.get_durations_dict()
        self._history.history.update(timer_output)
        self.write_train_history()
        self.write_data_support_metrics()
        operations_logger.debug("Epoch durations: {}".format(timer_output[TimerCallback.EPOCH_DUR_KEY]))
        operations_logger.debug("Train duration: {}".format(timer_output[TimerCallback.TRAIN_DUR_KEY]))
        operations_logger.info("********* Training Done", tid=self.job_metadata.job_id)
        return model

    @property
    def feature_col_size(self):
        return 2

    def get_vectors(
            self, annotation: MachineAnnotation = None,
            fs_client=None, features: List[FeatureType] = None) -> Vectorization:
        """
        :return: None bc Bert Model does not use vectors, this also avoids calls to feature service that
        thelstm model makes during testing
        """
        return None

    def vote_helper(self, y_pred_prob: np.ndarray, num_tokens: int, tid: str = None):
        # TODO create generalized vote helper that works for any stride size
        operations_logger.debug("Started Voting with Weight now...", tid=tid)
        # of dimension [number of tokens, num_classes]
        if y_pred_prob.shape[0] == 1:
            y_voted_weight_np = y_pred_prob[0]
        elif self.model_metadata.window_size == self.model_metadata.window_stride:
            # no overlapping windows, so we can just reshape
            y_voted_weight_np = y_pred_prob.reshape(-1, y_pred_prob.shape[2])
        else:
            # should be of dimension [number of tokens, num_classes]
            y_voted_weight_np = vote_with_weight(
                y_pred_prob, num_tokens=num_tokens, window_size=self.model_metadata.window_size
            )
        return y_voted_weight_np
