from biomed.biomed_env import BiomedEnv
from biomed.common.aspect_response import AspectResponse
from biomed.common.biomed_ouput import SummaryOutput
from biomed.common.helpers import get_pref_umls_concept_polarity, get_longest_pref_text
from biomed.constants.constants import get_ensemble_version
from biomed.constants.model_constants import ModelType, MODEL_TYPE_2_CONSTANTS
from biomed.meta.ensembler import Ensembler
from text2phenotype.apm.metrics import text2phenotype_capture_span
from text2phenotype.common.featureset_annotations import MachineAnnotation, Vectorization
from text2phenotype.constants.features import FeatureType
from text2phenotype.constants.umls import SemTypeCtakesAsserted


@text2phenotype_capture_span()
def family_history(tokens: MachineAnnotation,
                   vectors: Vectorization,
                   text: str,
                   biomed_version: str = BiomedEnv.DEFAULT_BIOMED_VERSION.value,
                   **kwargs) -> dict:
    """
    return the diagnosis terms recommended bilstm diagnosis model and append the attribute if clinical find the
    attribute of this only append attribute of the terms if name in ['DiseaseDisorderMention', 'SignSymptomMention']
    # TODO: need to add SignSymptom to chunker_clinical also
    """
    # JIRA BIOMED-544
    # JIRA BIOMED-653 
    if not tokens.output_dict:
        return dict()
    model_type = ModelType.family_history
    model_constants = MODEL_TYPE_2_CONSTANTS[model_type]

    ensemble_version = get_ensemble_version(model_type, biomed_version)
    ensembler = Ensembler(
        model_type=model_type,
        model_file_list=ensemble_version.model_names,
        voting_method=ensemble_version.voting_method,
    )

    res = ensembler.predict(tokens, vectors=vectors, **kwargs)
    predicted_category = res.predicted_category

    biomed_output_list = []
    for i in range(len(tokens)):
        if predicted_category[i] != 0:
            token_text = tokens.tokens[i]
            range_txt = tokens.range[i]

            # get original umls concepts
            clinical_res = tokens[FeatureType.clinical, i]

            umls_concept, polarity, _ = get_pref_umls_concept_polarity(
                clinical_res, [SemTypeCtakesAsserted.DiseaseDisorder.name, SemTypeCtakesAsserted.SignSymptom.name],
                get_longest_pref_text)

            biomed_output_list.append(
                SummaryOutput(
                    text=token_text,
                    range=range_txt,
                    lstm_prob=res.predicted_probs[i, int(predicted_category[i])],
                    umlsConcept=umls_concept,
                    attributes={'polarity': polarity},
                    label=model_constants.label_class.get_from_int(predicted_category[i]).value.persistent_label))

    return AspectResponse(
        model_constants.label_class.get_category_label().persistent_label,
        biomed_output_list).to_versioned_json(model_type=model_type, biomed_version=biomed_version)
