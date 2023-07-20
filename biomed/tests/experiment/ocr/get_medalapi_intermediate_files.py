"""
Given a job ID from text2phenotypeAPI, download all intermediate files for the given job

Requires AWS_PROFILE to be set to the target account, and be authenticated to AWS

EG:
get_text2phenotypeapi_intermediate_files --job_id=a1af53a843594912938e59ee8e3b58f9 \
    --bucket=prod-nlp-biomed-us-east-1 \
    --output_dir=/Users/michaelpesavento/data/ocr_compare/output/biomedAPI/intermediate_files

If using pycharm, enter this in the run configuration "parameters":
--job_id=a1af53a843594912938e59ee8e3b58f9 --bucket=prod-nlp-biomed-us-east-1 --output_dir=/Users/michaelpesavento/data/ocr_compare/output/biomedAPI/intermediate_files

"""
import argparse
import os
import json

from text2phenotype.common import aws


def parse_arguments():
    parser = argparse.ArgumentParser(description="Get text2phenotypeAPI job intermediate files")
    parser.add_argument("--job_id", type=str, required=True, help="uuid of target job")
    parser.add_argument("--output_dir", type=str, required=False, help="Absolute output path to write files")
    parser.add_argument("--bucket", type=str, default="prod-nlp-biomed-us-east-1")
    return parser.parse_args()


def main(args):
    output_dir = args.output_dir
    bucket = args.bucket
    job_id = args.job_id.replace("-", "")
    prefix_key = os.path.join("processed/jobs", job_id)

    s3_job_keys = aws.list_keys(bucket, prefix_key)
    if not s3_job_keys:
        raise ValueError(
            f"No completed jobs found with the given bucket and prefix: s3://{bucket}/{prefix_key}")

    manifest_key = [key for key in s3_job_keys if key.endswith(f"{job_id}.manifest.json")][0]

    s3_client = aws.get_s3_client()
    job_manifest = aws.get_object(s3_client=s3_client, bucket=args.bucket, key=manifest_key)
    job_manifest = json.loads(job_manifest)
    doc_info = job_manifest["document_info"]
    doc_id_list = list(doc_info.keys())

    # get the intermediate files for each doc_id
    for i, doc_id in enumerate(doc_id_list):
        doc_prefix = f"processed/documents/{doc_id}"
        s3_keys = list(aws.get_matching_s3_keys(bucket, prefix=doc_prefix))
        local_path = os.path.join(output_dir, doc_id)
        os.makedirs(local_path, exist_ok=True)
        aws.sync_down_files(s3_client, bucket, s3_keys, local_folder_path=local_path)
        print(f"({i+1}/{len(doc_id_list)}) Downloaded {doc_id} locally")

    print(f"Finished writing {len(doc_id_list)} documents to {output_dir}")


if __name__ == "__main__":
    main(parse_arguments())

