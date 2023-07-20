import os
import sys
import datetime

from biomed.models.model_metadata import ModelMetadata
from biomed.train_test.job_metadata import JobMetadata
from biomed.data_sources.data_source import BiomedDataSource
from biomed.constants.constants import ModelType
from text2phenotype.constants.environment import Environment
from biomed.models.get_model import get_model_class_from_model_type

# should be less than one third (ore one quarter) of the BERT_WINDOW_LENGTH
# otherwise get truncation errors
WINDOW_SIZE = 64
WINDOW_STRIDE = WINDOW_SIZE

MODEL_TYPE = ModelType.diagnosis_bert.value
MODEL_TYPE_NAME = ModelType(MODEL_TYPE).name

TIME_STR = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
JOB_ID = f"{MODEL_TYPE_NAME}_{TIME_STR}_w{WINDOW_SIZE}_subset"

TESTING_PARAMS = {
    "parent_dir": "/opt/S3/biomed-data",
    "model_type": MODEL_TYPE,
    "window_size": WINDOW_SIZE,
    "window_stride": WINDOW_STRIDE,
    "features": [],
    "learning_rate": 1e-5,
    "batch_size": 16,
    "train": True,
    "test": True,
    "embedding_model_name": "bert",
    "train_embedding_layer": True,
    "epochs": 1,
    "job_id": JOB_ID,
    "async_mode": False,

    # diagnosis_bert
    "feature_set_version": "dev/20201113c",
    "original_raw_text_dirs": [
        "I2B2/2014 De-identification and Heart Disease Risk Factors Challenge/gold_raw_text/demographic/surrogates_v1"
    ],
    "ann_dirs": [
        "nick.colangelo/2020-11-22/diseasesign"
    ],
    "feature_set_subfolders": ["train"],
    "testing_fs_subfolders": ["test"],
    "validation_fs_subfolders": ["test"],

    # for drug_bert
    # "original_raw_text_dirs": [
    #     "mimic/20190207_andy/txt/Consult",
    #     "mimic/20190207_andy/txt/DischargeSummary/12",
    #     "mtsamples/clean"
    # ],
    # "ann_dirs": [
    #     # "briana.galloway/BIOMED-1215-summary",
    #     # "deleys.brandman/annotation_BIOMED-655",
    #     "deleys.brandman/BIOMED-1991-subset",
    # ],
    # "feature_set_version": "dev/20201012",
    # "feature_set_subfolders": ["a", "b", "c", "d"],
    # "testing_fs_subfolders": ["e"],
    # "validation_fs_subfolders": ["e"],
    # "model_folder_name": "drug_bert_20201220-093316_w64_subset",

    "source_bucket": "biomed-data",
    "dest_bucket": "biomed-data",
}


def main(params):
    """main entry point"""
    # TODO: just read in a target .json file to parse parameters.

    model_metadata = ModelMetadata(**params)
    data_source = BiomedDataSource(**params)
    job_metadata = JobMetadata.from_dict(params)

    # Get the model class
    model_type = ModelType(params["model_type"])
    model_class = get_model_class_from_model_type(model_type)
    model_folder_name = params.get("model_folder_name", None)

    model_base = model_class(
        model_metadata=model_metadata,
        data_source=data_source,
        job_metadata=job_metadata,
        model_folder_name=model_folder_name,
        model_type=model_type,
    )

    if job_metadata.train:
        model_base.train()
    if job_metadata.test:
        model_base.test()


if __name__ == "__main__":
    os.environ["MDL_COMN_USE_STORAGE_SVC"] = "False"
    # os.environ["MDL_COMN_STORAGE_CONTAINER_NAME"] = "biomed-data"
    Environment.load()
    sys.exit(main(TESTING_PARAMS))
