from biomed.biomed_env import BiomedEnv
from biomed.common.aspect_response import AspectResponse
from biomed.constants.constants import get_ensemble_version
from biomed.constants.model_constants import ModelType, ProcedureConstants
from biomed.meta.ensembler import Ensembler
from text2phenotype.common.featureset_annotations import MachineAnnotation, Vectorization


def get_procedures(
        tokens: MachineAnnotation,
        vectors: Vectorization,
        text: str,
        biomed_version: str = BiomedEnv.DEFAULT_BIOMED_VERSION.value,
        **kwargs
):
    if not tokens.output_dict:
        return dict()
    ensemble_version = get_ensemble_version(ModelType.procedure, biomed_version)
    ensembler = Ensembler(
        model_type=ModelType.procedure,
        model_file_list=ensemble_version.model_names,
        voting_method=ensemble_version.voting_method,
    )

    res = ensembler.predict(tokens, vectors=vectors)
    out_list = res.token_dict_list(label_enum=ProcedureConstants.label_class)
    return AspectResponse(
        ProcedureConstants.label_class.get_category_label().persistent_label,
        out_list).to_versioned_json(model_type=ModelType.procedure, biomed_version=biomed_version)
