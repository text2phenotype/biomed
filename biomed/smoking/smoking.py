import string

from biomed.biomed_env import BiomedEnv
from biomed.common.aspect_response import AspectResponse
from biomed.common.biomed_ouput import SummaryOutput
from biomed.constants.constants import ModelType, get_ensemble_version
from biomed.constants.model_constants import SmokingConstants
from biomed.meta.ensembler import Ensembler

from text2phenotype.apm.metrics import text2phenotype_capture_span
from text2phenotype.common.featureset_annotations import MachineAnnotation, Vectorization
from text2phenotype.constants.features.label_types import SmokingLabel


@text2phenotype_capture_span()
def get_smoking_status(tokens: MachineAnnotation = None, vectors: Vectorization = None,
                       biomed_version: str = BiomedEnv.DEFAULT_BIOMED_VERSION.value,
                       **kwargs) -> dict:
    if not tokens.output_dict:
        return dict()
    ensemble_version = get_ensemble_version(ModelType.smoking, biomed_version)
    ensembler = Ensembler(
        model_type=ModelType.smoking,
        model_file_list=ensemble_version.model_names,
        voting_method=ensemble_version.voting_method,
    )

    ensembler_feats = set(ensembler.feature_list)
    ensembler_feats.update(SmokingConstants.required_representation_features)

    res = smoking_predict_represent(tokens,
                                    use_generator=BiomedEnv.BIOMED_USE_PREDICT_GENERATOR.value,
                                    vectors=vectors,
                                    ensembler=ensembler,
                                    tid=kwargs.get('tid'))

    return res.to_versioned_json(model_type=ModelType.smoking, biomed_version=biomed_version)


@text2phenotype_capture_span()
def smoking_predict_represent(tokens: MachineAnnotation, use_generator: bool, ensembler: Ensembler,
                              **kwargs) -> AspectResponse:

    res = ensembler.predict(tokens, use_generator=use_generator, **kwargs)
    category_predictions = res.predicted_category
    predicted_probabilities = res.predicted_probs

    results = []
    for i in range(len(tokens.tokens)):
        curr_token = tokens.tokens[i]
        pred_category = int(category_predictions[i])
        if pred_category == SmokingLabel.unknown.value.column_index or curr_token in string.punctuation:
            continue

        results.append(SummaryOutput(label=SmokingLabel.get_from_int(pred_category).value.persistent_label,
                                     text=curr_token,
                                     range=tokens.range[i],
                                     lstm_prob=predicted_probabilities[i, pred_category]))

    return AspectResponse(category_name=SmokingLabel.get_category_label().persistent_label, response_list=results)
