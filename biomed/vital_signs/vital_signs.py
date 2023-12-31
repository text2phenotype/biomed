from biomed.biomed_env import BiomedEnv
from biomed.common.annotation_matching import get_closest_date
from biomed.common.aspect_response import AspectResponse
from biomed.common.biomed_ouput import BiomedOutput, BiomedOutputWithDate
from biomed.constants.constants import get_ensemble_version, ModelType
from biomed.meta.ensembler import Ensembler
from text2phenotype.common.featureset_annotations import MachineAnnotation, Vectorization
from text2phenotype.constants.features import FeatureType
from text2phenotype.constants.features.label_types import VitalSignsLabel


def get_vital_signs(
        tokens: MachineAnnotation,
        vectors: Vectorization,
        text: str,
        biomed_version: str = BiomedEnv.DEFAULT_BIOMED_VERSION.value,
        **kwargs) -> dict:
    if not tokens.output_dict:
        return dict()

    ensemble_version = get_ensemble_version(ModelType.vital_signs, biomed_version)
    ensembler = Ensembler(
        model_type=ModelType.vital_signs,
        model_file_list=ensemble_version.model_names,
        voting_method=ensemble_version.voting_method,
    )

    res = ensembler.predict(tokens, vectors=vectors, **kwargs)
    category_predictions = res.predicted_category
    predicted_probabilities =res.predicted_probs
    results = []
    for i in range(len(tokens)):
        if category_predictions[i] != VitalSignsLabel.na.value.column_index:
            lstm_prob = predicted_probabilities[i, int(category_predictions[i])]
            token_text = tokens.tokens[i]
            range_txt = tokens.range[i]

            closest_date_obj = get_closest_date(token_idx=i, dates=tokens[FeatureType.date_comprehension])


            results.append(
                BiomedOutputWithDate(
                    text=token_text,
                    range=range_txt,
                    lstm_prob=lstm_prob,
                    label=VitalSignsLabel.get_from_int(category_predictions[i]).value.persistent_label,
                    date=closest_date_obj))

    out = AspectResponse(category_name=VitalSignsLabel.get_category_label().persistent_label,
                          response_list=results)
    out.post_process(text=text)

    return out.to_versioned_json(model_type=ModelType.vital_signs, biomed_version=biomed_version)
