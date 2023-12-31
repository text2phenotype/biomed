import datetime

from text2phenotype.common.featureset_annotations import MachineAnnotation, Vectorization
from text2phenotype.apm.metrics import text2phenotype_capture_span
from text2phenotype.common.log import operations_logger
from text2phenotype.constants.features import (
    FeatureType,
    LabLabel,
    CovidLabLabel)
from text2phenotype.constants.umls import SemTypeCtakesAsserted

from biomed.biomed_env import BiomedEnv
from biomed.common.helpers import get_prefered_covid_concept, get_pref_umls_concept_polarity, get_first
from biomed.common.annotation_matching import get_closest_nearby_annotation, get_closest_date
from biomed.constants.constants import get_ensemble_version, ModelType
from biomed.common.biomed_ouput import LabOutput, CovidLabOutput
from biomed.common.aspect_response import LabResponse
from biomed.models.model_cache import ModelCache
from biomed.meta.ensembler import Ensembler


@text2phenotype_capture_span()
def summary_lab_value(tokens: MachineAnnotation,
                      vectors: Vectorization,
                      text: str,
                      biomed_version: str = BiomedEnv.DEFAULT_BIOMED_VERSION.value,
                      **kwargs) -> dict:
    # get predictions
    if not tokens.output_dict:
        return dict()
    ensemble_version = get_ensemble_version(ModelType.lab, biomed_version)
    ensembler = Ensembler(
        model_type=ModelType.lab,
        model_file_list=ensemble_version.model_names,
        voting_method=ensemble_version.voting_method,
    )

    res = ensembler.predict(tokens, vectors=vectors, **kwargs)
    category_predictions = res.predicted_category
    predicted_probabilities = res.predicted_probs
    dates = tokens[FeatureType.date_comprehension]
    date_indexes = sorted([int(idx) for idx in dates.token_indexes])

    results = []
    # get tui and cui rules
    cache = ModelCache()
    cui_rule = cache.cui_rule()
    tui_rule = cache.tui_rule()
    # loop through and extract predicted labs that also match hepc and don't fail cui/tui rule tests
    for idx in range(len(tokens)):
        # if predicted to be a lab name
        if category_predictions[idx] != LabLabel.na.value.column_index:
            lab_annotations = tokens[FeatureType.lab_hepc, idx]

            umls_concept, _, attributes = get_pref_umls_concept_polarity(
                lab_annotations, [SemTypeCtakesAsserted.Lab.name], get_first)
            lstm_prob = predicted_probabilities[idx, int(category_predictions[idx])]
            if umls_concept or lstm_prob > 0.5:
                # this should always be true w/ model average
                closest_date_idx = get_closest_nearby_annotation(idx, date_indexes)
                if closest_date_idx:
                    closest_date = dates[closest_date_idx][0]
                    closest_date_obj = datetime.date(
                        year=closest_date['year'],
                        month=closest_date['month'],
                        day=closest_date['day'])
                else:
                    closest_date_obj = None
                # pick first lab concept and use that
                concept = LabOutput(umlsConcept=umls_concept,
                                    text=tokens.tokens[idx],
                                    lstm_prob=lstm_prob,
                                    range=tokens.range[idx],
                                    label=LabLabel.get_from_int(category_predictions[idx]).name,
                                    attributes=attributes,
                                    date=closest_date_obj)

                # get negative filters
                if umls_concept:
                    tui_permitted = 'lab' in tui_rule.get(concept.tui, {}).get('aspect_list', [])
                    cui_not_mapped = not cui_rule.get(concept.cui, {}).get('aspect_list')
                    cui_mapped_known_lab = 'lab' in cui_rule.get(concept.cui, {}).get('aspect_list', [])
                    if not (tui_permitted or cui_not_mapped or cui_mapped_known_lab):
                        continue

                results.append(concept)

    operations_logger.debug('Lab Extraction Completed', tid=kwargs.get('tid'))

    out = LabResponse(LabLabel.get_category_label().persistent_label, results)
    out.post_process(text=text)
    # filter out post processed labs without any attributes or UMLS concept
    for i in range(len(out.response_list)-1, -1, -1):
        concept = out.response_list[i]
        has_no_attribute = (concept.labValue.text is None and concept.labUnit.text is None and concept.labInterp.text is None)
        has_no_umls = concept.cui is None
        if has_no_umls and has_no_attribute:
            bad_resp = out.response_list.pop(i)

    return out.to_versioned_json(model_type=ModelType.lab, biomed_version=biomed_version)


@text2phenotype_capture_span()
def get_covid_labs(tokens: MachineAnnotation,
                   vectors: Vectorization,
                   text: str,
                   biomed_version: str = BiomedEnv.DEFAULT_BIOMED_VERSION.value,
                   **kwargs) -> dict:
    if not tokens.output_dict:
        return dict()
    ensemble_version = get_ensemble_version(ModelType.covid_lab, biomed_version)
    ensembler = Ensembler(
        model_type=ModelType.covid_lab,
        model_file_list=ensemble_version.model_names,
        voting_method=ensemble_version.voting_method,
    )

    # get predictions
    res = ensembler.predict(tokens, vectors=vectors)
    category_predictions = res.predicted_category
    predicted_probabilities = res.predicted_probs

    results = []

    # loop through and extract predicted labs that also match hepc and don't fail cui/tui rule tests
    for i in range(len(tokens)):
        if category_predictions[i] != LabLabel.na.value.column_index:
            lstm_prob = predicted_probabilities[i, int(category_predictions[i])]
            token_text = tokens.tokens[i]
            range_txt = tokens.range[i]

            umls_concept, polarity = get_prefered_covid_concept(tokens, i)
            closest_date_obj = get_closest_date(token_idx=i, dates=tokens[FeatureType.date_comprehension])

            manufacturer = get_nearest_covid_manufacturer(tokens=tokens, token_idx=i)

            results.append(
                CovidLabOutput(
                    text=token_text,
                    range=range_txt,
                    lstm_prob=lstm_prob,
                    umlsConcept=umls_concept,
                    label=CovidLabLabel.get_from_int(category_predictions[i]).value.persistent_label,
                    labManufacturer=manufacturer,
                    date=closest_date_obj))
    output = LabResponse(CovidLabLabel.get_category_label().persistent_label, results)

    output.post_process(text=text)

    return output.to_versioned_json(model_type=ModelType.covid_lab, biomed_version=biomed_version)


def get_nearest_covid_manufacturer(
        tokens: MachineAnnotation,
        token_idx: int,
        max_token_dist: int = 400,
        max_token_man_name=6) -> list:
    """
    :param tokens: feature set annotation for the document
    :param token_idx: index for a covid_lab_name that we want to try and find a manufacturer for
    :param max_token_dist: maximum number of tokens between the manufacturer name and covid lab name that we are
    willing to call connected
    :param max_token_man_name: maximum number of token that can be in a manufacturer's name
    :return: a list of word, range_start, range_end for lab manufacturer name tokens
    """
    # find the closest token annotated by feature service as lab_manufacturer names
    manufacturers_idx = sorted([int(i) for i in tokens[FeatureType.covid_lab_manufacturer].token_indexes])
    nearest_manufact_idx = get_closest_nearby_annotation(
        token_index=token_idx,
        annotation_token_indexes=manufacturers_idx,
        max_token_annot_dist=max_token_dist)
    if nearest_manufact_idx is not None:
        man_txt = tokens.tokens[nearest_manufact_idx]
        man_range = tokens.range[nearest_manufact_idx]

        if nearest_manufact_idx > token_idx:
            # check tokens that come after the closest right lab manufacturer to see if they are also
            # part of lab manufacturer name
            for i in range(1, max_token_man_name):
                if tokens[FeatureType.covid_lab_manufacturer, nearest_manufact_idx + i]:
                    man_txt = f'{man_txt} {tokens.tokens[nearest_manufact_idx + i]}'
                    man_range[1] = tokens.range[nearest_manufact_idx + i][1]
                else:
                    break
            out = [man_txt, man_range[0], man_range[1]]
        else:
            # check tokens that come before the closest left lab manufacturer to see if they are also part of
            # lab manufacturer name
            for i in range(1, max_token_man_name):
                if tokens[FeatureType.covid_lab_manufacturer, nearest_manufact_idx - i]:
                    man_txt = f'{tokens.tokens[nearest_manufact_idx + i]} {man_txt}'
                    man_range[0] = tokens.range[nearest_manufact_idx - i][0]

                else:
                    break
            out = [man_txt, man_range[0], man_range[1]]

    else:
        out = []

    return out
