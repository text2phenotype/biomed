import datetime
import string
from typing import Tuple

from biomed.biomed_env import BiomedEnv
from biomed.common.biomed_summary import FullSummaryResponse
from biomed.common.combined_model_label import DrugLabel
from biomed.common.helpers import get_pref_umls_concept_polarity, get_first
from biomed.common.annotation_matching import get_closest_nearby_annotation
from text2phenotype.common.featureset_annotations import MachineAnnotation, Vectorization

from text2phenotype.apm.metrics import text2phenotype_capture_span
from text2phenotype.constants.features import FeatureType
from text2phenotype.constants.features.label_types import MedLabel

from biomed.meta.ensembler import Ensembler
from biomed.common.biomed_ouput import MedOutput, SummaryOutput
from biomed.common.aspect_response import AspectResponse
from biomed.constants.constants import MED_BLACK_LIST, ModelType, get_ensemble_version
from text2phenotype.constants.umls import SemTypeCtakesAsserted


@text2phenotype_capture_span()
def meds_and_allergies(tokens: MachineAnnotation,
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
    ensemble_version = get_ensemble_version(ModelType.drug, biomed_version)
    ensembler = Ensembler(
        model_type=ModelType.drug,
        model_file_list=ensemble_version.model_names,
        voting_method=ensemble_version.voting_method,
    )

    res = ensembler.predict(tokens, vectors=vectors)
    category_predictions = res.predicted_category
    predicted_probabilities = res.predicted_probs
    med_result = []
    allergy_result = []
    dates = tokens[FeatureType.date_comprehension]
    date_indexes = sorted([int(idx) for idx in dates.token_indexes])

    for i in range(len(tokens['token'])):
        if category_predictions[i] != MedLabel.na.value.column_index:
            lstm_prob = predicted_probabilities[i, int(category_predictions[i])]
            token_text = tokens.tokens[i]
            range_txt = tokens.range[i]

            clinical_res = tokens[FeatureType.drug_rxnorm, i]
            umls_concept, _, attributes = get_pref_umls_concept_polarity(
                clinical_res, [SemTypeCtakesAsserted.Medication.name], get_first)

            # get umls concepts for predicted meds
            if umls_concept:
                if token_text.lower() in MED_BLACK_LIST or '..' in token_text or token_text in string.octdigits \
                        or (lstm_prob <= 0 or not umls_concept or not umls_concept.get('cui')):
                    continue
                # include all meds predicted as not NA that have a umls rxnorm code associated with them
                if category_predictions[i] == DrugLabel.allergy.value.column_index:
                    allergy_result.append(
                        SummaryOutput(
                            text=token_text,
                            range=range_txt,
                            lstm_prob=predicted_probabilities[
                                i, DrugLabel.allergy.value.column_index],
                            umlsConcept=umls_concept,
                            attributes=attributes,
                            label=DrugLabel.allergy.value.persistent_label))
                elif category_predictions[i] == DrugLabel.med.value.column_index:
                    closest_date_idx = get_closest_nearby_annotation(i, date_indexes)
                    if closest_date_idx:
                        closest_date = dates[closest_date_idx][0]
                        closest_date_obj = datetime.date(
                            year=closest_date['year'],
                            month=closest_date['month'],
                            day=closest_date['day'])
                    else:
                        closest_date_obj = None
                    med_result.append(
                        MedOutput(
                            text=token_text,
                            range=range_txt,
                            lstm_prob=lstm_prob,
                            umlsConcept=umls_concept,
                            attributes=attributes,
                            label=MedLabel.med.value.persistent_label,
                            date=closest_date_obj))
    full_response = FullSummaryResponse()
    full_response.add(
        AspectResponse(
            category_name=DrugLabel.med.value.category_label,
            response_list=med_result))
    full_response.add(
        AspectResponse(
            category_name=DrugLabel.allergy.value.category_label,
            response_list=allergy_result))

    full_response.postprocess(text=text)
    return full_response.to_json(biomed_version=biomed_version, model_type=ModelType.drug)
