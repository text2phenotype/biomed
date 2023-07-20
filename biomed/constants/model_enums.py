"""
Store enumerated classes used in defining a model or ensemble
Should have no external dependencies from the rest of the repository
"""
from enum import IntEnum, Enum


class ModelType(IntEnum):
    """
    Each model type is unique by a set of class labels that a model is trying to predict
    Only models of a single type can be ensembled.
    I.e., a model may have different features but needs to generate the same prediction classes
    """
    deid = 1
    vital_signs = 2
    device_procedure = 3
    demographic = 4
    lab = 5
    family_history = 6
    meta = 7
    covid_lab = 8
    disability = 9
    imaging_finding = 10
    oncology = 11
    smoking = 12
    drug = 13
    diagnosis = 14
    doc_type = 15
    sequioa_bladder = 16
    bladder_risk = 17
    date_of_service = 18
    sdoh = 19
    genetics = 20
    procedure = 21


class ModelClass(IntEnum):
    """
    Enum that corresponds to mapping in get_model (see model/get_model.py)
    This is used for getting the model class, either during training/testing a model or from within the ensembler
    Therefore the ensembler does not have a mapping in this way.
    """
    lstm_base = 1  # used for all LSTM base models (currently ModelBase)
    doc_type = 2  # derived from ModelBase
    bert = 3  # used in all BertBase models
    meta = 4  # should be removed! used by MetaLSTM, which is unused
    spacy = 5  # it's a spacy model (unclear type expectations)


class BertEmbeddings(Enum):
    """
    map of embedding model "names" (folders in resources/files/bert_embedding)
    """
    bert = "bert-base-uncased"
    # there is a bio_bert and a clinical_bert, so we distinguish those embeddings
    bio_clinical_bert = "bio_clinical_bert_all_notes_150000"


class VotingMethodEnum(Enum):
    """
    This is an enum used to map strings (eg from metadata) to voting function names
    """
    model_avg = "model_avg"
    model_weighted_avg = "model_weighted_avg"
    weighted_entropy = "weighted_entropy"
    threshold = "threshold"
    threshold_categories = "threshold_categories"
    rf_classifier = "rf_classifier"
