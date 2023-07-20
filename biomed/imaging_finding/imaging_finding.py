from biomed.biomed_env import BiomedEnv
from biomed.common.biomed_summary import FullSummaryResponse
from biomed.constants.constants import get_ensemble_version, ModelType
from text2phenotype.common.feature_data_parsing import is_digit_punctuation
from text2phenotype.common.featureset_annotations import MachineAnnotation, Vectorization
from text2phenotype.apm.metrics import text2phenotype_capture_span
from text2phenotype.constants.english_stopwords import ENGLISH_STOPWORDS
from text2phenotype.constants.features.label_types import FindingLabel

from biomed.meta.ensembler import Ensembler
from biomed.common.biomed_ouput import BiomedOutput
from biomed.common.aspect_response import AspectResponse
from biomed.common.combined_model_label import ImagingFindingLabel


@text2phenotype_capture_span()
def imaging_and_findings(tokens: MachineAnnotation,
                         vectors: Vectorization,
                         text: str,
                         biomed_version: str = BiomedEnv.DEFAULT_BIOMED_VERSION.value,
                         **kwargs) -> dict:
    """
    this function returns drug terms from ctakes drug_ner and add terms from bilstm model
    that is not appeared in drug_ner but predicted as drug
    """
    if not tokens.output_dict:
        return dict()
    ensemble_version = get_ensemble_version(ModelType.imaging_finding, biomed_version)
    ensembler = Ensembler(
        model_type=ModelType.imaging_finding,
        model_file_list=ensemble_version.model_names,
        voting_method=ensemble_version.voting_method,
    )

    res = ensembler.predict(tokens, vectors=vectors)
    category_predictions = res.predicted_category
    predicted_probabilities = res.predicted_probs
    findings_res = []
    diagnostic_imaging_studies = []

    for i in range(len(tokens['token'])):
        if category_predictions[i] != ImagingFindingLabel.na.value.column_index:
            pred_cat_int = int(category_predictions[i])
            lstm_prob = predicted_probabilities[i, pred_cat_int]
            token_text = tokens.tokens[i]
            range_txt = tokens.range[i]

            if token_text.lower() in ENGLISH_STOPWORDS or '..' in token_text or is_digit_punctuation(text):
                continue
            elif category_predictions[i] == ImagingFindingLabel.finding.value.column_index:
                findings_res.append(
                    BiomedOutput(
                        text=token_text,
                        range=range_txt,
                        lstm_prob=lstm_prob,
                        label=ImagingFindingLabel.finding.value.persistent_label))
            else:
                diagnostic_imaging_studies.append(
                    BiomedOutput(
                        text=token_text,
                        range=range_txt,
                        lstm_prob=lstm_prob,
                        label=ImagingFindingLabel.get_from_int(
                            pred_cat_int).value.persistent_label))
    full_response = FullSummaryResponse()

    full_response.add(
        AspectResponse(
            category_name=FindingLabel.get_category_label().persistent_label,
            response_list=findings_res))
    full_response.add(
        AspectResponse(
            category_name=ImagingFindingLabel.xray.value.category_label,
            response_list=diagnostic_imaging_studies))

    full_response.postprocess(text=text)

    return full_response.to_json(model_type=ModelType.imaging_finding, biomed_version=biomed_version)
