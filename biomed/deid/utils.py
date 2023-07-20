import string
from typing import (
    Iterable,
    List,
)

from text2phenotype.apm.metrics import text2phenotype_capture_span

from text2phenotype.constants.features import (
    DEM_to_PHI,
    DemographicEncounterLabel,
    PHILabel,
)
from text2phenotype.tasks.task_info import ChunksIterable


from biomed.common.aspect_response import AspectResponse
from biomed.common.biomed_ouput import BiomedOutput


@text2phenotype_capture_span()
def iter_deid_from_demographics(chunk_mapping: ChunksIterable,
                                **kwargs) -> Iterable[BiomedOutput]:
    for chunk_span, chunk_data in chunk_mapping:
        dem_preds = chunk_data.get(DemographicEncounterLabel.get_category_label().persistent_label, [])
        yield (chunk_span, demographics_chunk_to_deid(dem_preds).to_json())


@text2phenotype_capture_span()
def deid_from_demographics(chunk_mapping: ChunksIterable,
                           **kwargs) -> Iterable[BiomedOutput]:
    return list(iter_deid_from_demographics(chunk_mapping, **kwargs))


def demographics_chunk_to_deid(demographics_response: List[dict]) -> AspectResponse:
    phi_entries = []
    for pred_idx in range(len(demographics_response)):
        pred = demographics_response[pred_idx]
        biomed_out = BiomedOutput(**pred)
        phi_label: PHILabel = DEM_to_PHI.get(DemographicEncounterLabel.from_brat(biomed_out.label))
        if phi_label != PHILabel.na and biomed_out.valid_range() and biomed_out.text not in string.punctuation:
            biomed_out.label = phi_label.value.persistent_label
            phi_entries.append(biomed_out)
    return AspectResponse(PHILabel.get_category_label().persistent_label, phi_entries)
