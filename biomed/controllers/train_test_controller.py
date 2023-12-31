import asyncio
import connexion
from random import randint

import json

from text2phenotype.common.log import operations_logger
from text2phenotype.services.queue.drivers.rmq_updated import RMQBasicPublisher

from biomed.biomed_env import BiomedEnv
from biomed.data_contracts.biomed_request import BiomedRequest
from train_test_build import run_job


def train_test(biomed_request: BiomedRequest = None):
    if connexion.request.is_json:
        req = BiomedRequest.from_dict(connexion.request.get_json())
    else:
        req = BiomedRequest.from_dict(biomed_request)
    req, tid = get_job_id(req)
    operations_logger.info(f'Request with tid {tid} received')
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        operations_logger.info("CREATING LOOP")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    future_val = asyncio.ensure_future(att_helper(req.data))
    loop.run_until_complete(future_val)
    return tid


async def att_helper(metadata):
    run_job(metadata)


def train_test_task(biomed_request: BiomedRequest = None):
    if connexion.request.is_json:
        req = BiomedRequest.from_dict(connexion.request.get_json())
    else:
        req = BiomedRequest.from_dict(biomed_request)
    req, tid = get_job_id(req)
    operations_logger.info(f'Request with tid {tid} received')

    client = RMQBasicPublisher(BiomedEnv.TRAIN_TEST_TASKS_QUEUE.value,
                               client_tag='train_test_task controller')
    client.publish_message(json.dumps(req.data))

    return tid


def get_job_id(req):
    if 'job_id' in req.data:
        tid = req.data['job_id']
    elif req.tid:
        tid = req.tid
        req.data['job_id'] = tid
    else:
        tid = str(randint(0, 1000))
        req.data['job_id'] = tid
    return req, tid
