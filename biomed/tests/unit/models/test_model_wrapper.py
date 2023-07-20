import os
import unittest

import tensorflow as tf
import numpy as np

from biomed.constants.model_constants import ModelClass
from biomed.models.model_cache import ModelMetadataCache
from text2phenotype.common import common
from biomed.constants.constants import ModelType, get_version_model_folders
from biomed.models.model_wrapper import ModelWrapper, ModelWrapperSession
from biomed.resources import LOCAL_FILES
from biomed.constants.constants import BIOMED_VERSION_TO_MODEL_VERSION


def find_model_class(model_class: ModelClass, model_type: ModelType, metadata_cache: ModelMetadataCache):
    model_folders = []
    for model_folder in get_version_model_folders(model_type=model_type):
        meta = metadata_cache.model_metadata(model_type=model_type, model_folder=model_folder)
        if meta.model_class == model_class:
            model_folders.append(model_folder)
    return model_folders


class TestModelWrapper(unittest.TestCase):
    # get the most recent version
    BIOMED_VERSION = list(BIOMED_VERSION_TO_MODEL_VERSION.keys())[-1]
    META_CACHE = ModelMetadataCache()
    # model types can be any modl type using a specific model class in latest prod version
    LSTM_MODEL_TYPE = ModelType.diagnosis
    LSTM_MODELS = find_model_class(ModelClass.lstm_base, ModelType.diagnosis, META_CACHE)

    BERT_MODEL_TYPE = ModelType.diagnosis
    BERT_MODELS = find_model_class(ModelClass.bert, ModelType.diagnosis, META_CACHE)

    @staticmethod
    def _find_model_file(model_type_name, model_folder_name):
        folder_path = os.path.join(LOCAL_FILES, model_type_name, model_folder_name)
        model_file_name = common.get_file_list(folder_path, ".h5")[0]
        return model_file_name

    def _get_model_absolute_path(self, model_type, model_folder):
        # for bert, this should point to the tf_model.h5 filename
        full_model_path = self._find_model_file(model_type.name, model_folder)
        return full_model_path

    def assert_lstm_predict(self, wrapped_model):
        input_shape = (1,) + wrapped_model.model.layers[0].input_shape[1:]
        output_shape = (1,) + wrapped_model.model.layers[-1].output_shape[1:]
        x = np.zeros(input_shape)
        out = wrapped_model.predict(x)
        self.assertEqual(output_shape, out.shape)
        self.assertIsInstance(out, np.ndarray)

    def assert_lstm_call(self, wrapped_model):
        input_shape = (1,) + wrapped_model.model.layers[0].input_shape[1:]
        output_shape = (1,) + wrapped_model.model.layers[-1].output_shape[1:]
        x = np.zeros(input_shape)
        # check the call output, and make sure the tensor has been evaluated
        out = wrapped_model(x)
        self.assertEqual(output_shape, out.shape)
        self.assertIsInstance(out, np.ndarray)

    def assert_bert_predict(self, wrapped_model):
        input_shape = (2, 64)
        output_shape = (2, 64, 3)
        x = np.zeros(input_shape, dtype=np.int32)

        out = wrapped_model.predict({"input_ids": x, "attention_mask": x})
        self.assertEqual(output_shape, out[0].shape)
        self.assertIsInstance(out[0], np.ndarray)

    def assert_bert_call(self, wrapped_model):
        input_shape = (2, 64)
        output_shape = (2, 64, 3)
        x = np.zeros(input_shape, dtype=np.int32)
        # check the call output, and make sure the tensor has been evaluated
        out = wrapped_model({"input_ids": x, "attention_mask": x})

        self.assertEqual(output_shape, out[0].shape)
        self.assertIsInstance(out[0], np.ndarray)

    def test_init_known_model_folder_bert(self):
        full_model_path = self._get_model_absolute_path(self.BERT_MODEL_TYPE, self.BERT_MODELS[0])
        wrapped_model = ModelWrapper(model_file_path=full_model_path)
        self.assertIsInstance(wrapped_model, ModelWrapper)
        self.assertIsInstance(wrapped_model.model, tf.keras.Model)

    def test_init_known_model_folder_bilstm(self):
        full_model_path = self._get_model_absolute_path(model_type=self.LSTM_MODEL_TYPE,
                                                        model_folder=self.LSTM_MODELS[0])
        wrapped_model = ModelWrapper(model_file_path=full_model_path)
        self.assertIsInstance(wrapped_model, ModelWrapper)
        self.assertIsInstance(wrapped_model.model, tf.keras.Model)

    def test_init_no_h5_name_in_path(self):
        # a model folder without the .h5 filename should give an OSError
        model_type = ModelType.drug
        model_folders = get_version_model_folders(model_type, biomed_version=self.BIOMED_VERSION)
        folder_path = os.path.join(LOCAL_FILES, model_type.name, model_folders[0])
        with self.assertRaises(OSError):
            _ = ModelWrapper(model_file_path=folder_path)

    def test_init_invalid_h5_name(self):
        # amazingly, this will work since the folder is required to have tf_model.h5 in it,
        model_type = ModelType.drug
        model_folders = get_version_model_folders(model_type, biomed_version=self.BIOMED_VERSION)
        folder_path = os.path.join(LOCAL_FILES, model_type.name, model_folders[0], "saved_model.pb")
        with self.assertRaises(OSError):
            _ = ModelWrapper(model_file_path=folder_path)

    def test_init_no_h5_name_on_bert(self):
        # amazingly, this will work since the folder is required to have tf_model.h5 in it,
        model_type = ModelType.drug
        model_folders = get_version_model_folders(model_type, biomed_version=self.BIOMED_VERSION)
        folder_path = os.path.join(LOCAL_FILES, model_type.name, model_folders[0])
        with self.assertRaises(OSError):
            _ = ModelWrapper(model_file_path=folder_path)

    def test_graph_session_exist(self):
        full_model_path = self._get_model_absolute_path(model_type=self.LSTM_MODEL_TYPE,
                                                        model_folder=self.LSTM_MODELS[0])
        wrapped_model = ModelWrapperSession(model_file_path=full_model_path)

        self.assertIsInstance(wrapped_model.session, tf.compat.v1.Session)
        self.assertIsInstance(wrapped_model.session.graph, tf.compat.v1.Graph)

    def test_lstm_model_wrapper(self):
        full_model_path = self._get_model_absolute_path(model_type=self.LSTM_MODEL_TYPE,
                                                        model_folder=self.LSTM_MODELS[0])
        wrapped_model = ModelWrapper(model_file_path=full_model_path)
        self.assert_lstm_predict(wrapped_model)
        self.assert_lstm_call(wrapped_model)

    def test_bert_model_wrapper(self):
        full_model_path = self._get_model_absolute_path(model_type=self.BERT_MODEL_TYPE,
                                                        model_folder=self.BERT_MODELS[0])
        wrapped_model = ModelWrapper(model_file_path=full_model_path)

        self.assert_bert_predict(wrapped_model)
        self.assert_bert_call(wrapped_model)

    def test_lstm_model_wrapper_session(self):
        full_model_path = self._get_model_absolute_path(model_type=self.LSTM_MODEL_TYPE,
                                                        model_folder=self.LSTM_MODELS[0])
        wrapped_model = ModelWrapperSession(model_file_path=full_model_path)
        self.assert_lstm_predict(wrapped_model)
        self.assert_lstm_call(wrapped_model)


if __name__ == '__main__':
    unittest.main()
