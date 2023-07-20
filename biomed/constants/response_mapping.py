from text2phenotype.constants.features import CancerLabel, LabLabel, CovidLabLabel, MedLabel, AllergyLabel, ProblemLabel, \
    SignSymptomLabel, DeviceProcedureLabel, SmokingLabel, VitalSignsLabel
from text2phenotype.constants.features.label_types import (
    DiagnosticImagingLabel,
    DemographicEncounterLabel,
    FindingLabel,
    PHILabel,
    GeneticsLabel,
    DisabilityLabel,
    FamilyHistoryLabel,
    BladderRiskLabel,
    SequoiaBladderLabel,
    SocialRiskFactorLabel, ProcedureLabel)

from text2phenotype.tasks.task_enums import TaskOperation

from biomed.common.aspect_response import LabResponse, AspectResponse
from biomed.common.biomed_ouput import CovidLabOutput, LabOutput, MedOutput, SummaryOutput, VitalSignOutput, \
    BiomedOutput, CancerOutput, GeneticsOutput
from biomed.constants.constants import ICD10_CATEGORY, DATE_OF_SERVICE_CATEGORY

KEY_TO_ASPECT_OUTPUT_CLASSES = {
    CovidLabLabel.get_category_label().persistent_label: (LabResponse, CovidLabOutput),
    LabLabel.get_category_label().persistent_label: (LabResponse, LabOutput),
    MedLabel.get_category_label().persistent_label: (AspectResponse, MedOutput),
    AllergyLabel.get_category_label().persistent_label: (AspectResponse, SummaryOutput),
    ProblemLabel.get_category_label().persistent_label: (AspectResponse, SummaryOutput),
    SignSymptomLabel.get_category_label().persistent_label: (AspectResponse, SummaryOutput),
    DeviceProcedureLabel.get_category_label().persistent_label: (AspectResponse, BiomedOutput),
    SmokingLabel.get_category_label().persistent_label: (AspectResponse, BiomedOutput),
    VitalSignsLabel.get_category_label().persistent_label: (AspectResponse, VitalSignOutput),
    CancerLabel.get_category_label().persistent_label: (AspectResponse, CancerOutput),
    DiagnosticImagingLabel.get_category_label().persistent_label: (AspectResponse, BiomedOutput),
    FindingLabel.get_category_label().persistent_label: (AspectResponse, SummaryOutput),
    DemographicEncounterLabel.get_category_label().persistent_label: (AspectResponse, BiomedOutput),
    PHILabel.get_category_label().persistent_label: (AspectResponse, BiomedOutput),
    GeneticsLabel.get_category_label().persistent_label: (AspectResponse, GeneticsOutput),
    DisabilityLabel.get_category_label().persistent_label: (AspectResponse, BiomedOutput),
    TaskOperation.doctype.value: (AspectResponse, BiomedOutput),
    ICD10_CATEGORY: (AspectResponse, SummaryOutput),
    DATE_OF_SERVICE_CATEGORY: (AspectResponse, BiomedOutput),
    BladderRiskLabel.get_category_label().persistent_label: (AspectResponse, BiomedOutput),
    SequoiaBladderLabel.get_category_label().persistent_label: (AspectResponse, BiomedOutput),
    FamilyHistoryLabel.get_category_label().persistent_label: (AspectResponse, SummaryOutput),
    SocialRiskFactorLabel.get_category_label().persistent_label: (AspectResponse, SummaryOutput),
    ProcedureLabel.get_category_label().persistent_label: (AspectResponse, BiomedOutput)
}
