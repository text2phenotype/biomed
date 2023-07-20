from collections import defaultdict
from typing import Dict
import string

from biomed.biomed_env import BiomedEnv
from biomed.constants.constants import ModelType, get_ensemble_version
from biomed.meta.ensembler import Ensembler

from text2phenotype.apiclients.feature_service import FeatureServiceClient
from text2phenotype.apm.metrics import text2phenotype_capture_span
from text2phenotype.common.featureset_annotations import MachineAnnotation, Vectorization
from text2phenotype.constants.features.label_types import BladderRiskLabel, SequoiaBladderLabel, LabelEnum


@text2phenotype_capture_span()
def get_bladder_risk_tokens(text: str,
                            tid: str = None,
                            tokens: MachineAnnotation = None,
                            vectors: Vectorization = None,
                            biomed_version: str = BiomedEnv.DEFAULT_BIOMED_VERSION.value) -> Dict:
    return __get_predictions(text, ModelType.bladder_risk, BladderRiskLabel, tid=tid, tokens=tokens, vectors=vectors,
                             biomed_version=biomed_version)


@text2phenotype_capture_span()
def get_sequoia_risk_tokens(text: str,
                            tid: str = None,
                            tokens: MachineAnnotation = None,
                            vectors: Vectorization = None,
                            biomed_version: str = BiomedEnv.DEFAULT_BIOMED_VERSION.value) -> Dict:
    return __get_predictions(text, ModelType.sequioa_bladder, SequoiaBladderLabel, tid=tid, tokens=tokens,
                             vectors=vectors, biomed_version=biomed_version)


def __get_predictions(text: str, model_type: ModelType, label_type: LabelEnum, tid: str = None,
                      tokens: MachineAnnotation = None, vectors: Vectorization = None,
                      biomed_version: str = BiomedEnv.DEFAULT_BIOMED_VERSION.value) -> Dict:
    if not text:
        return {}

    ensemble_version = get_ensemble_version(model_type, biomed_version)
    ensembler = Ensembler(
        model_type=model_type,
        voting_method=ensemble_version.voting_method,
        model_file_list=ensemble_version.model_names)

    feature_service_client = FeatureServiceClient()
    ensembler_feats = set(ensembler.feature_list)
    if not tokens:
        tokens = feature_service_client.annotate(text, features=ensembler_feats, tid=tid)
        if not tokens:
            return {}

    if not vectors:
        vectors = feature_service_client.vectorize(tokens, features=ensembler.feature_list, tid=tid)

    res = ensembler.predict(tokens, vectors=vectors, use_generator=BiomedEnv.BIOMED_USE_PREDICT_GENERATOR.value)

    results = defaultdict(list)
    token_list = tokens.tokens
    category_predictions = res.predicted_category
    predicted_probabilities = res.predicted_probs

    for i in range(len(token_list)):
        predicted_category = int(category_predictions[i])
        token = token_list[i]

        if predicted_category != 0 and token not in string.punctuation:
            label = label_type.get_from_int(predicted_category).value.persistent_label
            output = {"text": token,
                      "range": tokens.range[i],
                      "score": predicted_probabilities[i, predicted_category]}

            results[label].append(output)

    return results
