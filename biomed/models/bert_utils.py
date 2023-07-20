from dataclasses import dataclass, asdict, fields
from typing import List, Optional

import numpy as np

# defines the label to use for special tokens ([CLS], [SEP], [PAD])
# as well as used for subtokens following the initial subtoken on a whole word
# Defined here to avoid circular import between bert loss and bert BertBase
BERT_PADDING_LABEL = -100


def dataclass_from_dict(klass, d):
    """Utility method to return dataclass instance from a dict"""
    try:
        fieldtypes = {f.name: f.type for f in fields(klass)}
        return klass(**{f: dataclass_from_dict(fieldtypes[f], d[f]) for f in d})
    except Exception:
        return d  # Not a dataclass field


@dataclass
class InputDoc:
    """
    Dataclass to hold required raw content from a data document, used as input to BertBase.windowed_encodings()
    NOTE(mjp): not currently used, breaks too many abstractions relying on dicts
    """
    tokens: List[str]  # list of strings containing the document tokens
    valid_tokens: Optional[List[str]] = None  # used for DataCounter purposes, not required
    word_token_labels: Optional[List[int]] = None  # predictions don't have labels

    def __len__(self):
        return len(self.tokens)

    def to_json(self):
        return asdict(self)

    @classmethod
    def from_json(cls, doc_dict):
        return cls(**doc_dict)


@dataclass
class InputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    Each field is a list of lists; the outer list is the list of windows, inner is the values for that window

    This object would be the output from BertBase.windowed_encodings() for each document fed in as an InputDoc
    NOTE(mjp): not currently used

    Required Args:
        input_ids: List[List[int]], list of the vocabulary id for a given subtoken;
        attention_mask: List[List[int]], binary mask for whether a given element in input_ids has a word/token
            that should be parsed by the model. Zeros are used for empty padding values
        valid_token_mask: List[List[bool]], True if the subtoken is the start of a valid whole word token
    Optional arg:
        encoded_labels: A list of lists of the label index values for each subtoken.
            This list will use a padding label (generally -100) for all special tokens and extended subtokens
            See BertBase.encode_doc_subtoken_labels() for more information

    """
    input_ids: List[List[int]]
    attention_mask: List[List[int]]
    valid_token_mask: List[List[bool]]
    encoded_labels: Optional[List[List[int]]] = None  # optional cause predictions dont have labels

    def __len__(self):
        """
        how many windows do we have for the given document?
        :return:
        """
        return len(self.input_ids)

    @property
    def window_size(self):
        """Get the window size used in the encodings"""
        return len(self.input_ids[0])

    def to_json(self):
        return self.__dict__.copy()

    @classmethod
    def from_json(cls, doc_dict):
        return cls(**doc_dict)
