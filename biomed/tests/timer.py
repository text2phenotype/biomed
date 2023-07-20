import datetime
from collections import OrderedDict


class Timer:

    def __init__(self, name=None, content=None, status=None):
        """
        :param name: identifies the timer, such as a filename
        :param content: payload of the message, for example, the 'text' or the 'response'
        :param status: bool True/False/None, such as status was 'success'
        """
        self.name = name
        self._content = content
        self._status = status
        self._start = None
        self._stop = None

        self.start()

    def start(self):
        self._start = Timer.now()
        self._stop = None

        return self._start

    def stop(self, status=None, content=None):
        self._status = status
        self._content = content

        self._stop = Timer.now()

        return self._stop

    def success(self, content=None):
        return self.stop(True, content)

    def failed(self, content=None):
        return self.stop(False, content)

    def elapsed(self):
        if self._stop is None:
            self.stop()

        return (self._stop - self._start).total_seconds()

    @staticmethod
    def now():
        """
        :return: datetime object (datetime.datetime.now())
        """
        return datetime.datetime.now()

    def to_dict(self):
        return {'name': self.name,
                'elapsed': self.elapsed(),
                'start': str(self._start),
                'stop': str(self._stop),
                'content': self._content,
                'status': self._status}


class TimerBatch:

    def __init__(self):
        self.items = OrderedDict()
        self.timer = Timer('TimerBatch')

    def filename(self):
        return f'{self.timer.name}.json'

    def get(self, index) -> Timer:
        return self.items[index]

    def add(self, timer: Timer):
        self.items[timer.name] = timer
        self.timer.stop()

    def to_dict(self):
        out = OrderedDict()
        out['TimerBatch'] = self.timer.to_dict()

        for key, item in self.items.items():
            out[key] = item.to_dict()

        return out
