from typing import List, AnyStr, Optional, Union, Tuple, Dict

import numpy as np


class DataCounter:
    """
    Keep track of the support data counts over the dataset documents
    """

    def __init__(self, label2id, n_features, window_size, window_stride=None):
        """
        Initialize the counter
        :param label2id: dict mapping a label string to the label id, eg
            {"na": 0, "allergy": 1, "med", 2}
        :param n_features: int
            size of feature vectors, ie number of feature columns after being vectorized
        :param window_size: int
            number of tokens in each window
        :param window_stride:
            number of tokens the window shifts for each prediction batch
            1 = shifts every token (BiLSTM default
            window_size = no overlap between predictions
        """
        self.label2id = label2id
        self.id2label = {v: k for k, v in self.label2id.items()}
        self.n_classes = len(self.label2id)
        self.n_features = n_features
        self.window_size = window_size
        self.window_stride = window_stride or window_size  # use window_size as stride for non-overlapping windows
        self.n_documents: int = 0
        self.total_token_count: int = 0
        self.doc_token_counts: List[int] = []
        self.total_valid_token_count: int = 0
        self.doc_valid_token_counts: List[int] = []
        self.total_num_windows: int = 0
        self.doc_num_windows: List[int] = []
        self.doc_token_label_counts: List[Dict[str, int]] = []
        self.total_token_label_counts = {name: 0 for name in self.label2id}
        self.is_predict_count = False  # set as True if we are doing counting in a model.predict() method

    def __repr__(self):
        """dataclass-like repr"""
        return (
            f"{self.__class__.__name__}(" +
            ", ".join([
                f"{name}={val}"
                for name, val in self.__dict__.items()
                if not isinstance(val, list)
            ]) +
            ")"
        )

    @property
    def label_names(self):
        return list(self.label2id.keys())

    def add_document(
            self,
            doc_num_tokens: int,
            doc_num_valid_tokens: int,
            doc_token_labels: List[int],
            doc_window_count: int):
        """
        Update the counter with given document data
        :param doc_num_tokens: int
            Total number of word tokens in a given document
        :param doc_num_valid_tokens: int
            Count of valid tokens in a document; comes from len(machine_annotation.valid_tokens()[0])
        :param doc_token_labels: List[int]
            list of document token label integers
            eg: [0, 0, 0, 1, 2, 0]
        :param doc_window_count: int
            number of windows in given document
        :return: None
        """
        self.n_documents += 1
        self.total_token_count += doc_num_tokens
        self.doc_token_counts.append(doc_num_tokens)
        if doc_num_valid_tokens:
            self.total_valid_token_count += doc_num_valid_tokens
            self.doc_valid_token_counts.append(doc_num_valid_tokens)
        if doc_token_labels:
            self.update_doc_label_counts(doc_token_labels)
        self.total_num_windows += doc_window_count
        self.doc_num_windows.append(doc_window_count)

    def to_json(self):
        """Return attributes as a dictionary"""
        return self.__dict__

    def _inc_doc_labels(self, doc_token_label_count: Dict[AnyStr, int]):
        """
        Append and increment the doc label dict and total counts
        :param doc_token_label_count: Dict[Str, int]
            key must be in `self.label2id`
            eg: { "na": 12243, "diagnosis": 282, "signsymptom": 118}
        :return:
        """
        self.doc_token_label_counts.append(doc_token_label_count)
        for label, ct in doc_token_label_count.items():
            # cast as int rather than np.int64 so json doesnt convert to string
            self.total_token_label_counts[label] += int(ct)

    def update_doc_label_counts(self, doc_word_labels: List[int]):
        """
        Update the document label counts separately from the doc token counts
        For example, we are updating the tokens in a predict() call, but
        adding the labels for each document in the test().
        """
        doc_token_label_count = self.label_ids_to_counts(doc_word_labels)
        self._inc_doc_labels(doc_token_label_count)

    def label_ids_to_counts(self, doc_word_labels):
        """
        Utility function to count how many of each label exist
        :param doc_word_labels: List[int]
        :return: Dict[str, int]
            dict with the label string and the number of occurrences of the label_id in doc_word_labels
            eg: { "na": 12243, "diagnosis": 282, "signsymptom": 118}
        """
        word_label_counts = {name: 0 for name in self.label_names}
        label_count_tuple = np.unique(doc_word_labels, return_counts=True)
        for label_ix, count in zip(label_count_tuple[0], label_count_tuple[1]):
            word_label_counts[self.id2label[int(label_ix)]] += count
        return word_label_counts
