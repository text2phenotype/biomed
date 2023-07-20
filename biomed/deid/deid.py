import string
from typing import (
    List,
)

from text2phenotype.common.featureset_annotations import MachineAnnotation, Vectorization

from text2phenotype.apm.metrics import text2phenotype_capture_span
from text2phenotype.constants.features import PHILabel

from biomed.biomed_env import BiomedEnv
from biomed.common.aspect_response import AspectResponse
from biomed.constants.constants import (
    get_ensemble_version,
    ModelType,
)
from biomed.constants.model_constants import DeidConstants
from biomed.meta.ensembler import Ensembler


@text2phenotype_capture_span()
def get_phi_tokens(
        tokens: MachineAnnotation,
        vectors: Vectorization,
        use_generator: bool = BiomedEnv.BIOMED_USE_PREDICT_GENERATOR.value,
        models: List[str] = None,
        threshold_categories: List[str] = DeidConstants.rare_varying_classes,
        threshold: float = 0.5,
        return_all: bool = False,
        biomed_version: str = BiomedEnv.DEFAULT_BIOMED_VERSION.value,
        **kwargs) -> dict:
    """
    get a list of predicted phi tokens, their positions, probability of phi type of the input text
    :param tokens: clinical text like 'Patient name is Andrew McMurry'
    :param models: the deid models to use for predictions
    :param threshold_categories: which categories of PHI automatically get called PHI if any model predicts them wi
     prob > threshold
    :param use_threshold: whether or not to use thresholding ensembling(any vote as phi -> phi)
    :param threshold: probablity of phi must be greater than this
    :param return_all: whether or not to return all the tokens or just the phi tokens
    :param tid: transaction id
    :return: a list of phi token dictionary, or empty list if no PHI
    [{'phi': 'PATIENT', 'range': range(16, 22), 'score': 0.979, 'text': 'Andrew'},
     {'phi': 'PATIENT', 'range': range(23, 30), 'score': 0.983, 'text': 'McMurry'}]
    """
    ensemble_version = get_ensemble_version(ModelType.deid, biomed_version)
    ensembler = Ensembler(
        model_type=ModelType.deid,
        model_file_list=models or ensemble_version.model_names,
        voting_method=ensemble_version.voting_method,
        threshold=threshold,
        threshold_categories=threshold_categories,
    )

    if not tokens.output_dict:
        return dict()

    res = ensembler.predict(tokens, vectors=vectors, use_generator=use_generator, **kwargs)
    phi = res.token_dict_list(label_enum=PHILabel, filter_nas=not return_all)
    phi = [phi_ent for phi_ent in phi if phi_ent.text not in string.punctuation]

    # sort phi
    phi_label = PHILabel.get_category_label().persistent_label
    phi_response = AspectResponse(phi_label, phi)
    return phi_response.to_versioned_json(model_type=ModelType.deid, biomed_version=biomed_version)
