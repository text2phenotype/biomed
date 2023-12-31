from typing import List

from biomed.biomed_env import BiomedEnv
from biomed.common.aspect_response import AspectResponse
from biomed.common.biomed_ouput import BiomedOutput
from biomed.constants.constants import get_ensemble_version
from biomed.constants.model_constants import ModelType
from biomed.meta.ensembler import Ensembler
from text2phenotype.common.featureset_annotations import MachineAnnotation, Vectorization
from text2phenotype.common.log import operations_logger
from text2phenotype.constants.features import SocialRiskFactorLabel


def get_sdoh_response(
        tokens: MachineAnnotation,
        vectors: Vectorization,
        tid: str = None,
        biomed_version: str = BiomedEnv.DEFAULT_BIOMED_VERSION.value,
        **kwargs
) -> dict:
    """this function takes a clinical text and returns a list of predicted sdoh"""
    # output must be either None or list of sdoh tokens
    if not tokens.output_dict:
        operations_logger.info('Featureset annotation returned None')
        return {}
    ensemble_version = get_ensemble_version(ModelType.sdoh, biomed_version)
    ensembler = Ensembler(
        model_type=ModelType.sdoh,
        model_file_list=ensemble_version.model_names,
        voting_method=ensemble_version.voting_method,
    )

    if vectors is None:
        operations_logger.error("No Vectors Provided")

    res = ensembler.predict(tokens, vectors=vectors,
                            tid=tid)

    biomed_output_list: List[BiomedOutput] = res.token_dict_list(label_enum=SocialRiskFactorLabel)

    aspect_resp = AspectResponse(
        response_list=biomed_output_list,
        category_name=SocialRiskFactorLabel.get_category_label().persistent_label
    )
    return aspect_resp.to_versioned_json(biomed_version=biomed_version, model_type=ModelType.sdoh)
