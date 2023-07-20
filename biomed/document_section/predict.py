import os
from typing import List

import fasttext

from biomed.biomed_env import BiomedEnv
from biomed.common.aspect_response import AspectResponse
from biomed.common.biomed_ouput import BiomedOutput
from biomed.constants.constants import ModelType

from text2phenotype.apm.metrics import text2phenotype_capture_span
from text2phenotype.common import common
from text2phenotype.doc_type.predict import get_doc_types as mdl_get_doc_types
from text2phenotype.tasks.task_enums import TaskOperation


DOCTYPE_RESOURCE_DIR = os.path.join(BiomedEnv.BIOM_MODELS_PATH.value, 'resources', 'files', 'doc_type')
CLASSIFIER = fasttext.load_model(os.path.join(DOCTYPE_RESOURCE_DIR, 'fasttext_label_8020_20210201.bin'))
STOP_WORDS = eval(common.read_text(os.path.join(DOCTYPE_RESOURCE_DIR, 'stopwords.txt')))


@text2phenotype_capture_span()
def get_doc_types(text: str, **kwargs) -> List[dict]:
    predictions = mdl_get_doc_types(text, CLASSIFIER, STOP_WORDS)
    if not predictions:
        return []

    results = []
    for prediction in predictions:
        results.append(BiomedOutput(label=prediction["label"],
                                    text=prediction["text"],
                                    range=list(prediction["range"]),
                                    lstm_prob=prediction["prob"]))

    return AspectResponse(TaskOperation.doctype.value,
                          response_list=results).to_versioned_json(ModelType.doc_type)
