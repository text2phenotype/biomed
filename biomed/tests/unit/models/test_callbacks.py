import unittest
from unittest import mock

from biomed.models import callbacks


class TestClassAccuracy(unittest.TestCase):
    def test_get_recall(self):
        callback = callbacks.ClassRecall(None, None, None)

        exp = [0, 1, 1, 0, 1, 1]
        obs = [0, 1, 1, 1, 1, 0]
        callback._update_counts(exp, obs)

        self.assertEqual(3/4, callback.compute_metric())


class TestClassMicroF1Score(unittest.TestCase):
    def test_get_f1(self):
        callback = callbacks.ClassMicroF1Score(None, None, None)

        exp = [0, 0, 0, 1, 1, 1, 2, 2, 2]
        obs = [0, 0, 1, 1, 1, 0, 0, 1, 2]

        callback._update_counts(exp, obs)

        # tp / (tp + .5(fp + fn))
        # tp: 3
        # fp: 2
        # fn: 3
        # 3 / 5.5
        self.assertEqual(3 / 5.5, callback.compute_metric())


class TestTimerCallback(unittest.TestCase):
    start_time = 1500  # epoch time
    epoch_duration = 100  # in sec
    train_duration = 1000  # in sec

    @mock.patch('time.time', mock.MagicMock(return_value=(start_time + epoch_duration)))
    def test_epoch_duration(self):
        timer = callbacks.TimerCallback()
        timer.epoch_start_time = self.start_time
        timer.on_epoch_end(0)
        self.assertEqual(timer._epoch_durations_sec[0], self.epoch_duration)

    @mock.patch('time.time', mock.MagicMock(return_value=(start_time + train_duration)))
    def test_train_duration(self):
        timer = callbacks.TimerCallback()
        timer.train_start_time = self.start_time
        timer.on_train_end()
        self.assertEqual(timer._train_durations_sec, self.train_duration)

    @mock.patch('time.time', mock.MagicMock(return_value=(start_time + train_duration)))
    def test_get_durations_dict(self):
        # using train_duration because time patch only gives one output
        expected_times = {
            callbacks.TimerCallback.EPOCH_DUR_KEY: [self.train_duration, self.train_duration],
            callbacks.TimerCallback.TRAIN_DUR_KEY: self.train_duration,
            callbacks.TimerCallback.AVG_EPOCH_DUR_KEY: self.train_duration,
        }
        timer = callbacks.TimerCallback()
        timer.train_start_time = self.start_time
        timer.epoch_start_time = self.start_time
        timer.on_epoch_end(0)
        timer.on_epoch_end(1)
        timer.on_train_end()
        out_times = timer.get_durations_dict()
        self.assertDictEqual(out_times, expected_times)

    def test_stringify_sec_to_human(self):
        time_interval = 12345.6
        expected = "3:25:45.600000"
        time_str_out = callbacks.TimerCallback.stringify_sec_to_human(time_interval)
        self.assertEqual(time_str_out, expected)

        time_interval = 1234567.89
        expected = "14 days, 6:56:07.890000"
        time_str_out = callbacks.TimerCallback.stringify_sec_to_human(time_interval)
        self.assertEqual(time_str_out, expected)


if __name__ == '__main__':
    unittest.main()
