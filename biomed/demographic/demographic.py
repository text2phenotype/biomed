import bisect
from datetime import datetime
from typing import List, Tuple

from biomed.common.aspect_response import AspectResponse
from biomed.common.biomed_ouput import BiomedOutput
from biomed.constants.model_enums import VotingMethodEnum
from text2phenotype.common.featureset_annotations import MachineAnnotation, Vectorization

from text2phenotype.apm.metrics import text2phenotype_capture_span
from text2phenotype.common.log import operations_logger
from text2phenotype.constants.features import (
    DemographicEncounterLabel,
    FeatureType,
)

from biomed.biomed_env import BiomedEnv
from biomed.meta.ensembler import Ensembler
from biomed.constants.constants import ModelType, get_ensemble_version


@text2phenotype_capture_span()
def get_demographic_tokens(tokens: MachineAnnotation,
                           vectors: Vectorization,
                           date_matches: List[Tuple[datetime, Tuple[int, int]]] = None,
                           tid: str = None,
                           biomed_version: str = BiomedEnv.DEFAULT_BIOMED_VERSION.value,
                           **kwargs
                           ) -> dict:
    """this function takes a clinical text and returns a list of predicted demographics, or None"""
    # output must be either None or list of demographics tokens
    if not tokens.output_dict:
        operations_logger.info('Featureset annotation returned None')
        return {}
    ensemble_version = get_ensemble_version(ModelType.demographic, biomed_version)
    ensembler = Ensembler(
        model_type=ModelType.demographic,
        model_file_list=ensemble_version.model_names,
        voting_method=ensemble_version.voting_method,
    )

    if vectors is None:
        operations_logger.error("No Vectors Provided")

    res = ensembler.predict(tokens, use_generator=BiomedEnv.BIOMED_USE_PREDICT_GENERATOR.value, vectors=vectors,
                            tid=tid)

    demographic_tokens: List[BiomedOutput] = res.token_dict_list(label_enum=DemographicEncounterLabel)

    # add earliest date as dob
    dob = sorted(date_matches)[0] if date_matches else None
    if dob:
        dob_text = str(dob[0].month) + '/' + str(dob[0].day) + '/' + str(dob[0].year)
        demographic_tokens.append(BiomedOutput(text=dob_text, label=DemographicEncounterLabel.dob.name,
                                               lstm_prob=0, range=list(dob[1])))

    token_ranges = tokens.range
    # add city and state w/ prob = 0 extracted from zipcode
    for predicted in demographic_tokens:
        if DemographicEncounterLabel.pat_zip.value.persistent_label == predicted.label:
            index = bisect.bisect_left(token_ranges,
                                       predicted.range)

            demographic_tokens.extend(add_zipcode_demographics(zipcode_annotation=tokens[FeatureType.zipcode, index],
                                                               prefix='pat'))
        if DemographicEncounterLabel.dr_zip.value.persistent_label == predicted.label:
            index = bisect.bisect_left(token_ranges,
                                       predicted.range)
            demographic_tokens.extend(
                add_zipcode_demographics(tokens[FeatureType.zipcode, index], 'dr'))

    demographic_response = AspectResponse(
        DemographicEncounterLabel.get_category_label().persistent_label,
        demographic_tokens)
    return demographic_response.to_json()


def add_zipcode_demographics(zipcode_annotation, prefix: str) -> List[BiomedOutput]:
    zip_predictions = []
    if zipcode_annotation:
        for address_type in zipcode_annotation[0]:
            text = zipcode_annotation[0][address_type]
            if address_type != 'country':
                label = DemographicEncounterLabel[f'{prefix}_{address_type}']
                zip_predictions.append(
                    BiomedOutput(
                        text=text,
                        label=label.value.persistent_label,
                        lstm_prob=0,
                        range=[0, 0]
                    )
                )
    return zip_predictions
