from text2phenotype.common.log import operations_logger
from text2phenotype.constants.features.label_types import (LabelEnum, AllergyLabel, MedLabel, ProblemLabel, SignSymptomLabel,
                                                  DisabilityLabel, DiagnosticImagingLabel)


class CombinedAnnotation:
    column_index: int  # IMMUTABLE integer value identifying index of this label in Biomed arrays
    label: str  # human-readable/annotator-friendly label that will appear in annotation tool
    persistent_label: str  # IMMUTABLE: persistent label for matching in SANDS
    category_label: str

    def __init__(self, val: LabelEnum, col_index):
        self.column_index = col_index
        self.label = val.value.label
        self.persistent_label = val.value.persistent_label
        self.category_label = val.get_category_label().persistent_label
        self.order = val.value.order


class DrugLabel(LabelEnum):
    na = CombinedAnnotation(AllergyLabel.na, 0)
    allergy = CombinedAnnotation(AllergyLabel.allergy, 1)
    med = CombinedAnnotation(MedLabel.med, 2)

    @classmethod
    def from_brat(cls, brat_value: str):
        if brat_value.lower() == 'medication':
            value = cls.med
        elif brat_value.lower() in cls.__members__:
            value = cls[brat_value.lower()]
        else:
            operations_logger.debug(f'{brat_value.lower()} not in LabelEnum')
            value = cls.na
        return value


class DiseaseSignSymptomLabel(LabelEnum):
    na = CombinedAnnotation(ProblemLabel.na, 0)
    diagnosis = CombinedAnnotation(ProblemLabel.diagnosis, 1)
    signsymptom = CombinedAnnotation(SignSymptomLabel.signsymptom, 2)

    @classmethod
    def from_brat(cls, brat_value: str):
        if brat_value.lower() in [ProblemLabel.problem.value.persistent_label, ProblemLabel.problem.name,
                                  cls.diagnosis.name,
                                  cls.diagnosis.value.persistent_label]:
            value = cls.diagnosis
        elif brat_value.lower() in [cls.signsymptom.value.persistent_label, cls.signsymptom.name]:
            value = cls.signsymptom
        else:
            operations_logger.debug(f'{brat_value.lower()} not in LabelEnum')
            value = cls.na
        return value


class ImagingFindingLabel(LabelEnum):
    na = CombinedAnnotation(DisabilityLabel.na, 0)
    finding = CombinedAnnotation(DisabilityLabel.finding, 1)
    mri = CombinedAnnotation(DiagnosticImagingLabel.mri, 2)
    ecg = CombinedAnnotation(DiagnosticImagingLabel.ecg, 3)
    echo = CombinedAnnotation(DiagnosticImagingLabel.echo, 4)
    xray = CombinedAnnotation(DiagnosticImagingLabel.xray, 5)
    ct = CombinedAnnotation(DiagnosticImagingLabel.ct, 6)
    us = CombinedAnnotation(DiagnosticImagingLabel.us, 7)
    other_imaging = CombinedAnnotation(DiagnosticImagingLabel.other, 8)
