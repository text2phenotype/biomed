"""
ModelWrapper
Originally intended to hold the states associated with tf.Graph and tf.Session
in a Tensorflow 1.0 context.
With eager execution in Tensorflow 2.0, the use of Graphs and Sessions is no longer necessary,
and in some cases (BERT transformers) results in a memory leak over subsequent predictions

TODO(mjp): evaluate whether or not we need to continue using ModelWrapper for ModelCache usage
"""
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

from transformers import TFBertForTokenClassification

from biomed.biomed_env import BiomedEnv
from biomed.common.mat_3d_generator import Mat3dGenerator

from text2phenotype.common.log import operations_logger

# force eager execution
tf.compat.v1.enable_eager_execution()


def tensor_to_numpy(t: tf.Tensor) -> np.ndarray:
    """
    Convert a tensor (lazy or eager) to a numpy array
    :param t: tf.Tensor, output from model() or model(predict)
    :return:
    """
    try:
        t_ndarr = t.eval()
    except NotImplementedError as e:
        if "eval is not supported when eager execution is enabled" in e.args[0]:
            t_ndarr = t.numpy()
        else:
            raise
    except AttributeError:
        operations_logger.error("Couldn't eval() on model prediction outputs")
        t_ndarr = t
    return t_ndarr


class ModelWrapper:
    def __init__(self, model_file_path: str):
        """
        Initialize wrapped model from model path

        :param model_file_path: absolute file path to the target h5 file
        """
        self.file_name = model_file_path
        try:
            # loads .h5 or .pb file directly
            operations_logger.info(f"LOADING: {model_file_path}")
            self.model = load_model(model_file_path)
            # can sometimes get error
            # OSError: SavedModel file does not exist at: /Users/michaelpesavento/src/biomed/biomed/resources/files/drug_bert/drug_cbert_20210107_w64/
            # when we pass in a model_file_path without a filename
        except ValueError as e:
            # Huggingface transformer models will fail to load using keras.load_model
            if "No model found in config file" not in e.args[0]:
                raise e
            try:
                # this wont work if we use load_model on a folder for something other than bert
                # TODO: add load with config file?
                model_folder = os.path.dirname(model_file_path)
                self.model = TFBertForTokenClassification.from_pretrained(model_folder)
            except Exception as e:
                operations_logger.error(f"Unable to load model: {e}")
                raise e

    def __call__(self, *args, **kwargs):
        """
        Wrapper around the forward pass of the model
        tf.function decorator does JiT compiling of the model() call
           but it alson doesnt allow any non-tf operations to occur,
           such as conversion back to numpy
        :param args:
        :param kwargs:
        :return:
        """
        # bert returns a tuple, rather than a tensor (generally length 1)
        out_tensor = self.model(*args, **kwargs)

        if isinstance(out_tensor, (tuple, list)):
            out = [tensor_to_numpy(t) for t in out_tensor]
        else:
            out = tensor_to_numpy(out_tensor)
        # bilstm returns an evaluated tensor
        return out

    def predict(self, *args, **kwargs):
        return self.model.predict(*args, **kwargs)

    def predict_on_batch(self, *args, **kwargs):
        return self.model.predict_on_batch(*args, **kwargs)

    def predict_generator(self, generator: Mat3dGenerator) -> np.ndarray:
        return self.model.predict_generator(
            generator,
            workers=BiomedEnv.MODEL_PREDICTION_WORKERS.value,
            use_multiprocessing=BiomedEnv.MODEL_USE_MULTIPROCESSING.value,
            max_queue_size=BiomedEnv.MODEL_SEQUENCE_MAX_QUEUE.value)

    def predict_classes(self, x, batch_size: int):
        return self.model.predict_classes(x, batch_size=batch_size)

    def train_on_batch(self, x, y, sample_weight=None, class_weight=None):
        return self.model.train_on_batch(x, y, sample_weight=sample_weight, class_weight=class_weight)

    def fit(self, generator, **kwargs):
        return self.model.fit(generator, **kwargs)

    def to_json(self):
        return self.model.to_json()

    def save(self, model_file_path):
        self.model.save(model_file_path, save_format="h5")


class ModelWrapperSession(ModelWrapper):
    """
    An extention of ModelWrapper that uses tf.Graph and tf.Sessions
    for thread safety in older models (eg LSTM models)
    """

    def __init__(self, model_file_path: str):
        self.graph = tf.compat.v1.Graph()
        with self.graph.as_default():
            self.session = tf.compat.v1.Session()
            # TODO: reenable finalize, prevent memory leaks
            # self.session.graph.finalize()  # lock the graph to avoid memory leaks
            with self.session.as_default():
                super().__init__(model_file_path)

    def __call__(self, *args, **kwargs):
        # do some fancy juggling for models that return lazy tensors from the model call
        with self.graph.as_default(), self.session.as_default():
            tensors = self.model(*args, **kwargs)
            if isinstance(tensors, (tuple, list)):
                out = [tensor_to_numpy(t) for t in tensors]
            else:
                out = tensor_to_numpy(tensors)
        return out

    def predict(self, *args, **kwargs):
        with self.graph.as_default(), self.session.as_default():
            out = super().predict(*args, **kwargs)
        # clear the session to avoid memory leak on predict
        # NOTE: this may add more time to predictions, as we are deallocating memory more frequently
        tf.keras.backend.clear_session()
        return out

    def predict_on_batch(self, *args, **kwargs):
        with self.graph.as_default(), self.session.as_default():
            return super().predict_on_batch(*args, **kwargs)

    def predict_generator(self, generator: Mat3dGenerator) -> np.ndarray:
        with self.graph.as_default(), self.session.as_default():
            return super().predict_generator(generator)

    def predict_classes(self, x, batch_size: int):
        with self.graph.as_default(), self.session.as_default():
            return super().predict_classes(x, batch_size)

    def train_on_batch(self, x, y, sample_weight=None, class_weight=None):
        with self.graph.as_default(), self.session.as_default():
            return super().train_on_batch(x, y, sample_weight=sample_weight, class_weight=class_weight)

    def fit(self, generator, **kwargs):
        with self.graph.as_default(), self.session.as_default():
            return super().fit(generator, **kwargs)

    def to_json(self):
        with self.graph.as_default(), self.session.as_default():
            return super(ModelWrapperSession, self).to_json()

    def save(self, model_file_path):
        with self.graph.as_default(), self.session.as_default():
            super().save(model_file_path)
