from biomed.common.aspect_response import AspectResponse
from biomed.common.biomed_summary import FullSummaryResponse
from text2phenotype.constants.features.label_types import FamilyHistoryLabel, ProblemLabel, SignSymptomLabel


def remove_family_history_from_disease(
        diagnosis_response: FullSummaryResponse,
        family_history_response: FullSummaryResponse) -> FullSummaryResponse:
    """
    Removes all family history tokens from the diagnosis response
    """
    problem_cat_name = ProblemLabel.get_category_label().persistent_label
    sign_cat_name = SignSymptomLabel.get_category_label().persistent_label
    fam_cat_name = FamilyHistoryLabel.get_category_label().persistent_label
    diagnosis_out_list = diagnosis_response.filter_out_response_by_range(
        list_1=diagnosis_response[problem_cat_name].response_list,
        filter_biomed_output_list=family_history_response[fam_cat_name].response_list
    )
    signsymptom_out_list = diagnosis_response.filter_out_response_by_range(
        list_1=diagnosis_response[sign_cat_name].response_list,
        filter_biomed_output_list=family_history_response[fam_cat_name].response_list
    )

    diagnosis_response[problem_cat_name] = AspectResponse(
        category_name=problem_cat_name,
        response_list=diagnosis_out_list
    )

    diagnosis_response[sign_cat_name] = AspectResponse(
        category_name=sign_cat_name,
        response_list=signsymptom_out_list
    )

    return diagnosis_response
