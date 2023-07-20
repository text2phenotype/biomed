import string
from typing import List

from biomed.biomed_env import BiomedEnv
from biomed.common.aspect_response import AspectResponse
from biomed.common.biomed_ouput import BiomedOutput
from biomed.constants.constants import ModelType, DEFAULT_MIN_SCORE, get_ensemble_version
from biomed.meta.ensembler import Ensembler
from biomed.procedure.procedure import get_procedures

from text2phenotype.apiclients.feature_service import FeatureServiceClient
from text2phenotype.apm.metrics import text2phenotype_capture_span
from text2phenotype.common.featureset_annotations import MachineAnnotation, Vectorization
from text2phenotype.constants.features.label_types import DeviceProcedureLabel, ProcedureLabel


@text2phenotype_capture_span()
def get_device_procedure_tokens(text: str, tid: str = None, min_score: float = DEFAULT_MIN_SCORE,
                                annotations: MachineAnnotation = None, vectors: Vectorization = None,
                                models: list = None,
                                biomed_version=BiomedEnv.DEFAULT_BIOMED_VERSION.value) -> List[dict]:
    """
    Get the device_procedure tokens
    """
    if not text:
        return []
    ensemble_version = get_ensemble_version(ModelType.device_procedure, biomed_version)
    ensembler = Ensembler(
        model_type=ModelType.device_procedure,
        model_file_list=models or ensemble_version.model_names,
        voting_method=ensemble_version.voting_method,
    )

    feature_service_client = FeatureServiceClient()
    ensembler_feats = set(ensembler.feature_list)
    if not annotations:
        annotations = feature_service_client.annotate(text, features=list(ensembler_feats), tid=tid)
        if not annotations:
            return []

    if not vectors:
        vectors = feature_service_client.vectorize(annotations, features=ensembler.feature_list, tid=tid)

    res = device_procedure_predict_represent(annotations,
                                             use_generator=BiomedEnv.BIOMED_USE_PREDICT_GENERATOR.value,
                                             vectors=vectors,
                                             ensembler=ensembler,
                                             tid=tid)

    res.post_process(text=text, min_score=min_score)

    return res.to_json()[res.category_name]


@text2phenotype_capture_span()
def device_procedure_predict_represent(tokens: MachineAnnotation,
                                       vectors: Vectorization,
                                       text: str,
                                       biomed_version: str = BiomedEnv.DEFAULT_BIOMED_VERSION.value,
                                       ensembler: Ensembler = None,
                                       use_generator: bool = False,
                                       **kwargs) -> dict:
    if not tokens.output_dict:
        return dict()
    if not ensembler:
        ensemble_version = get_ensemble_version(ModelType.device_procedure, biomed_version)
        ensembler = Ensembler(
            model_type=ModelType.device_procedure,
            model_file_list=ensemble_version.model_names,
            voting_method=ensemble_version.voting_method,
        )
    res = ensembler.predict(tokens, vectors=vectors, use_generator=use_generator, **kwargs)
    results = []
    token_list = tokens.tokens
    category_predictions = res.predicted_category
    predicted_probabilities = res.predicted_probs
    for i in range(len(token_list)):
        if category_predictions[i] != 0 and token_list[i] not in string.punctuation:
            results.append(
                BiomedOutput(
                    label=DeviceProcedureLabel.get_from_int(category_predictions[i]).value.persistent_label,
                    text=token_list[i],
                    range=tokens.range[i],
                    lstm_prob=predicted_probabilities[i, int(category_predictions[i])]))
    proc_label = ProcedureLabel.get_category_label().persistent_label
    procedures = AspectResponse.from_json(
        biomed_output_list=get_procedures(
            text=text, tokens=tokens, vectors=vectors, biomed_version=biomed_version)[proc_label],
        category_name=proc_label,
        biomed_output_class=BiomedOutput
    )

    results.extend(procedures.response_list)
    device_aspect = AspectResponse(
        category_name=DeviceProcedureLabel.get_category_label().persistent_label,
        response_list=results)
    device_aspect.post_process(text=text)
    return device_aspect.to_versioned_json(model_type=ModelType.device_procedure, biomed_version=biomed_version)
