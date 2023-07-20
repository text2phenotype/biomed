import json
import os
import pkg_resources

from text2phenotype.common import aws
from text2phenotype.common import common
from text2phenotype.common.log import operations_logger

BUCKET_NAME = "biomed-data"

# list of target files to sync; tuple is filename, src subfolder in FS_DEV_VERSION, and destination subfolder
FILENAME_SPLITS = [
    ("acute-kidney-failure.txt", "e", "train"),
    ("chronic-sinusitis.txt", "a", "test"),
    ("myoclonic-epilepsy.txt", "e", "train"),
]

# subset of "features" in machine annotation we want to store
KEEP_FEATURES = ["token", "range", "allergy_regex", "drug_rxnorm"]

DATASET = "mtsamples/clean"
ANN_PREFIX = "deleys.brandman/annotation_BIOMED-655/"

FS_DEV_VERSION = "dev/20201012"


def run_sync():
    target_root = pkg_resources.resource_filename("biomed.tests", "fixtures/data")

    # get a client
    # need to be authed to aws-dev
    client = aws.get_s3_client()

    # sync raw text
    raw_text_folder = os.path.join(target_root, "")
    raw_text_keys = [os.path.join(DATASET, name[0]) for name in FILENAME_SPLITS]
    raw_text_filepaths = [os.path.join(raw_text_folder, DATASET, name[0]) for name in FILENAME_SPLITS]
    for key, fn in zip(raw_text_keys, raw_text_filepaths):
        aws.download_file(client, BUCKET_NAME, key, fn)
        operations_logger.info(f"Wrote {fn}")

    # sync target annotations
    # we strip off the ANN_PREFIX when saving locally
    ann_folder = os.path.join(target_root, "annotations")
    ann_keys = [
        os.path.join(ANN_PREFIX, DATASET, name[0].replace(".txt", ".ann"))
        for name in FILENAME_SPLITS]
    ann_local_filepaths = [
        os.path.join(ann_folder, DATASET, name[0].replace(".txt", ".ann")) for name in FILENAME_SPLITS
    ]
    for key, fn in zip(ann_keys, ann_local_filepaths):
        aws.download_file(client, BUCKET_NAME, key, fn)
        operations_logger.info(f"Wrote {fn}")

    # sync target feature files
    fs_folder = os.path.join(target_root, "features")
    fs_keys = [
        os.path.join(FS_DEV_VERSION, src_subfolder, DATASET, name.replace(".txt", ".json"))
        for name, src_subfolder, _ in FILENAME_SPLITS]
    fs_local_filepaths = [
        os.path.join(fs_folder, dest_subfolder, DATASET, name.replace(".txt", ".json"))
        for name, _, dest_subfolder in FILENAME_SPLITS]

    # this is where we would create subfolders, if so desired
    for key, fn in zip(fs_keys, fs_local_filepaths):
        # download the json string, convert to dict, and only keep the keys we want
        file_str = aws.get_object_str(client, BUCKET_NAME, key)
        json_blob = json.loads(file_str)
        fs_out = {k: v for k, v in json_blob.items() if k in KEEP_FEATURES}
        common.write_json(fs_out, fn)
        operations_logger.info(f"Wrote {fn}")


if __name__ == "__main__":
    run_sync()
