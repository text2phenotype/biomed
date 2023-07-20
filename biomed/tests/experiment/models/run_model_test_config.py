"""
Read a set of parameter config files for model test evaluations, and send them to a Biomed service
via the BioMedClient

To use this script, it looks at the `MDL_COMN_BIOMED_API_BASE` environment variable to see
where to look for the biomed service endpoint.

If running locally (python -m biomed), the service will be running at "localhost:8080"
If running remotely (yacht), the service will be running at "localhost:10000", or whichever port
the service is running on after using `yacht create port-forward {service_name}`

Of note, each task sent via the BioMedClient will time out after 300 seconds. This will result
in the task timing out and the for loop continuing on with the next task, while the service
is still running the initial (potentially long) task.

From observation, it appears that this will queue up the requests in the biomed service, so it will
eventually get to all of the sent requests.

Better would be to have a service monitoring task completion, and that the biomed service has
multiple workers that could process the queue. rabbitmq may be useful for something like this.

"""

import sys
import os
import json
import time
from typing import List, Dict, AnyStr, Tuple

from text2phenotype.common.log import operations_logger
from text2phenotype.common import common
from text2phenotype.constants.environment import Environment
from text2phenotype.apiclients.biomed import BioMedClient

from biomed import RESULTS_PATH
from biomed.tests.experiment.models.create_iterable_model_test_datasets import AwsProfile

MAX_TOKEN_COUNT = 100000

JOB_ID_BASE = "diseasesign_compare_20200924"


def run_traintest_job(data_source: dict, job_metadata: dict, model_metadata: dict):
    """
    Run a job on the locally running Biomed worker

    useful trick: pass the unified param dict in for each of the method arguments

    :param data_source:
    :param job_metadata:
    :param model_metadata:
    :return:
    """
    client = BioMedClient(max_doc_word_count=MAX_TOKEN_COUNT)
    if not client.live():
        raise ConnectionError(f"Biomed client isnt live, check {os.environ['MDL_COMN_BIOMED_API_BASE']}")
    operations_logger.info(f"client is live: {client.live()}")
    operations_logger.info(f"client is ready: {client.ready()}")
    # this should be a blocking call
    operations_logger.info(f"Running {job_metadata['job_id']}")
    start_time = time.time()
    result = None
    try:
        result = client.run_train_test_job(
            model_metadata=model_metadata,
            job_metadata=job_metadata,
            data_source=data_source)
        operations_logger.info(f"Job completed: {result}")
    except Exception as e:
        operations_logger.error(e)
    finally:
        operations_logger.info(f"train_test_job took (or timed out after) {time.time() - start_time} s")
    return result  # should just be the job_id


def main():

    data_source_profile = "phi"
    is_phi = data_source_profile == AwsProfile.phi

    config_out_path = os.path.join(
        RESULTS_PATH,
        JOB_ID_BASE,
        "configs",
        data_source_profile,  # phi/ vs dev/
    )
    config_filenames = [fn for fn in os.listdir(config_out_path) if fn.endswith(".json")]
    operations_logger.info(f"Found {len(config_filenames)} for job iteration")

    # set our profile according to the data source we are running our model on
    os.environ["AWS_PROFILE"] = f"aws-{data_source_profile}"
    operations_logger.info(f"aws profile: {data_source_profile}")

    # HACK: super hacky! but we dont want to run everything
    # very model specific in the name formatting == BAD
    run_pairs = [
        ('ensemble', 'frea'),
        # ('ensemble', 'kohli'),
        # ('ensemble', 'siolas'),
    ]
    run_list = []
    for model, data in run_pairs:
        if not model.isdigit() and model != "ensemble":
            model = f"disease_sign_{model}_070720_all"
        data = f"disease_sign_{data}_070720_all"
        name = f"diseasesign_compare_20200924_model-{model}_data-{data}.json"
        run_list.append(name)
    # end HACK

    for config_filename in config_filenames:
        if config_filename not in run_list:
            operations_logger.info(f"Skipping {config_filename}")
            continue

        operations_logger.info(f"loading config file: {config_filename}")
        if is_phi:
            config_json = common.read_json(os.path.join(config_out_path, config_filename))['data']
        else:
            # this used to read in the escaped string, but no longer
            config_json = common.read_json(os.path.join(config_out_path, config_filename))

        # run the job on a local client
        run_traintest_job(config_json, config_json, config_json)
        operations_logger.info(f"Finished job: {config_filename}")

    return 0


if __name__ == "__main__":
    """
    To run this script, feature-service and biomed must both be running locally
    """
    # set this on the worker?
    # os.environ["MDL_COMN_STORAGE_CONTAINER_NAME"] = "biomed-data"
    os.environ["MDL_COMN_STORAGE_CONTAINER_NAME"] = "prod-nlp-train1-us-east-1"

    # set these on the client, aka here
    os.environ['MDL_COMN_USE_STORAGE_SVC'] = "False"  # want to point to S3 storage, not local
    # os.environ['MDL_COMN_BIOMED_API_BASE'] = "http://localhost:8080"
    os.environ['MDL_COMN_BIOMED_API_BASE'] = "http://localhost:10000"
    Environment.load()

    sys.exit(main())
