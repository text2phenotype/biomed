from abc import ABC, abstractmethod
import time
import datetime
import numpy as np

from tensorflow.keras.callbacks import Callback

from text2phenotype.common.log import operations_logger


class MetricCallback(Callback, ABC):
    """
    Implements a variation of Early Stopping, only keeping the best performing model on validation
    """

    def __init__(self, n_batches, val_data, model_file):
        super().__init__()
        self.__n_batches = n_batches
        self.validation_data = val_data
        self.__model_file = model_file
        self.__best_metric = float("-inf")

    def on_epoch_end(self, epoch, logs={}):
        self._begin_epoch_end()

        operations_logger.info(
            f"Computing model performance metric at end of epoch {epoch + 1}..."
        )

        for i in range(self.__n_batches):
            batch = next(self.validation_data)
            x, y = batch[:2]

            predictions = self.model.predict(x)

            self.__update_counts(y, predictions)

        metric = self.compute_metric()
        operations_logger.info(f"Epoch {epoch + 1} validation set class performance: {metric}")

        if metric > self.__best_metric:
            operations_logger.info(f"Current model performance {metric} better than previous.")
            self.model.save(self.__model_file, save_format="h5")
            self.__best_metric = metric

    @abstractmethod
    def compute_metric(self):
        pass

    @abstractmethod
    def _begin_epoch_end(self):
        pass

    def __update_counts(self, y_true, y_pred):
        for exp, pred in zip(y_true, y_pred):
            self._update_counts(np.argmax(exp, axis=1), np.argmax(pred, axis=1))


class ClassRecall(MetricCallback):
    def __init__(self, n_batches, val_data, model_file):
        super().__init__(n_batches, val_data, model_file)
        self.__n_correct = 0
        self.__n_processed = 0

    def compute_metric(self) -> float:
        return self.__n_correct / self.__n_processed

    def _begin_epoch_end(self):
        self.__n_correct = 0
        self.__n_processed = 0

    def _update_counts(self, exp, pred):
        for e, p in zip(exp, pred):
            if not e:
                continue

            self.__n_correct += e == p
            self.__n_processed += 1


class ClassMicroF1Score(MetricCallback):
    def __init__(self, n_batches, val_data, model_file):
        super().__init__(n_batches, val_data, model_file)
        self.__fp_count = 0
        self.__tp_count = 0
        self.__fn_count = 0

    def compute_metric(self):
        # tp / (tp + .5(fp + fn))
        f_count = self.__fp_count + self.__fn_count
        if not self.__tp_count + f_count:
            return 0

        return self.__tp_count / (self.__tp_count + (0.5 * f_count))

    def _begin_epoch_end(self):
        self.__fp_count = 0
        self.__tp_count = 0
        self.__fn_count = 0

    def _update_counts(self, exp, pred):
        for e, p in zip(exp, pred):
            # excluding TP na tokens.  these are the majority of the tokens in a document, so can greatly bias
            # the resulting f1 score.
            if not e and not p:
                continue

            if e == p:
                self.__tp_count += 1
            else:
                if e:
                    self.__fn_count += 1

                if p:
                    self.__fp_count += 1


class TimerCallback(Callback):
    """
    Record the wall-clock duration it took for each epoch, and over the complete train step
    """

    EPOCH_DUR_KEY = "epoch_durations_sec"
    TRAIN_DUR_KEY = "train_duration_sec"
    AVG_EPOCH_DUR_KEY = "avg_epoch_duration_sec"

    def __init__(self):
        super().__init__()
        self._epoch_durations_sec = []
        self.epoch_start_time = None
        self._train_durations_sec = []
        self.train_start_time = None

    @staticmethod
    def stringify_sec_to_human(seconds: float):
        """
        Convert time interval in seconds to a human readable string, eg '1 day, 17:12:57'

        :param seconds: float
        :return:
        """
        return "{:8>}".format(str(datetime.timedelta(seconds=seconds)))

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        self._epoch_durations_sec.append(time.time() - self.epoch_start_time)
        operations_logger.info(
            "Epoch {} took {}".format(
                epoch + 1, self.stringify_sec_to_human(self._epoch_durations_sec[-1])
            )
        )

    def on_train_begin(self, logs=None):
        self.train_start_time = time.time()

    def on_train_end(self, logs=None):
        self._train_durations_sec = time.time() - self.train_start_time
        operations_logger.info(
            "Model training took {}".format(self.stringify_sec_to_human(self._train_durations_sec))
        )

    def get_durations_dict(self):
        return {
            self.EPOCH_DUR_KEY: self._epoch_durations_sec,
            self.TRAIN_DUR_KEY: self._train_durations_sec,
            self.AVG_EPOCH_DUR_KEY: np.mean(self._epoch_durations_sec),
        }
