import dataclasses
from enum import IntEnum, Enum
from typing import Dict, List, Union, Tuple

from biomed.constants.model_enums import ModelType, ModelClass, VotingMethodEnum
from text2phenotype.common.log import operations_logger
from text2phenotype.constants.features import LabelEnum
from text2phenotype.constants.features.label_types import FamilyHistoryLabel, ProcedureLabel
from text2phenotype.constants.umls import SemTypeCtakesAsserted
from text2phenotype.constants.features import (
    FeatureType, SocialRiskFactorLabel,
    PHILabel, LabLabel, CancerLabel, GeneticsLabel, SmokingLabel,
    DemographicEncounterLabel, VitalSignsLabel, DeviceProcedureLabel, DocumentTypeLabel,
    BladderRiskLabel, SequoiaBladderLabel)

from biomed.common.combined_model_label import DrugLabel, DiseaseSignSymptomLabel, ImagingFindingLabel


@dataclasses.dataclass
class EnsembleVersion:
    """
    Use this instead of model_version_mapping?
    """
    version: str = None
    model_names: List[str] = None
    voting_method: VotingMethodEnum = VotingMethodEnum.model_avg
    voter_model: Union[str, None] = None

    # TODO: add sanity check that version was set; everything must be versioned!
    # TODO: check that TRAINABLE_VOTING_METHODS have voter_model filled, use voting_methods.is_trained_voter()


@dataclasses.dataclass()
class ModelConstants:
    """
    A base class used to create constants for defining specific models based around a model_type
    and the associated labels with that model type.
    Includes functionality to define a voter method that is specific to a particular version
    To add a new model type steps are simply: add to model type enum, create a model constants enum,
    add model_type to model constants mapping at the bottom of this file.
    """
    token_umls_representation_feat: Dict[FeatureType, List[SemTypeCtakesAsserted]] = None
    label_class: LabelEnum = None
    ensemble_version_list: List[EnsembleVersion] = None
    production = True
    required_representation_features: List = None
    rare_varying_classes: List[int] = None

    def __repr__(self):
        """Display the attributes in the object repr"""
        return (
            f"{self.__class__.__name__}(" +
            ", ".join([
                f"{name}={val}"
                for name, val in self.__dict__.items()
            ]) +
            ")"
        )

    @classmethod
    def get_ensemble_version(cls, version: str) -> Union[EnsembleVersion, None]:
        """
        Return the target object for a specific model version

        :param version: str, the target version string that matches the EnsembleVersion.version
            in the desired EnsembleVersion. NOTE this is not the biomed version!
        :returns: EnsembleVersion
            The object that contains all necessary information to define an ensemble version
            or None if there is no ensemble version for the ModelConstant type
        """
        if not cls.ensemble_version_list:
            operations_logger.error("No versioned ensembles exist for this ModelType")
            return None
        all_versions = {ensemble.version: ensemble for ensemble in cls.ensemble_version_list}
        if version not in all_versions.keys():
            raise KeyError(f"'{version}' not found. Possible versions: [{all_versions.keys()}]")
        return all_versions[version]

    @classmethod
    def model_version_mapping(cls) -> Union[Dict[str, List[str]], None]:
        """
        Get a dict map from version string to the model folder names used in that version

        eg:
            >>> LabConstants.model_version_mapping()
            {
                '1.00': ['5175', 'phi_lab_20201115_a1', '5183'],
                '1.01': ['phi_lab_20210726_v1'],
                '1.02': ['phi_lab_nmibc_covid_2021_08_13']
            }
        :return: dict
            Map from ModelType version to the model folder names used in the ensemble
            Returns None if no ensemble_version_list is set for the ModelConstant
        """
        if not cls.ensemble_version_list:
            operations_logger.error("No versioned ensembles exist for this ModelType")
            return None
        out = {ensemble.version: ensemble.model_names for ensemble in cls.ensemble_version_list}
        return out


class DiagnosisConstants(ModelConstants):
    ensemble_version_list = [
        EnsembleVersion(
            version='2.00',
            model_names=[
                '5532',
                '5582',
                'phi_disease_sign_persaud_20201115_b1',
                'phi_disease_sign_frea_20201115_all1'
            ],
            voting_method=VotingMethodEnum.weighted_entropy,
        ),
        EnsembleVersion(
            version='2.01',
            model_names=[
                "5532",
                "diagnosis_cbert_20210119_w64",
                "phi_diagnosis_bert_la_v1"
            ],
            voting_method=VotingMethodEnum.weighted_entropy,
        ),
        EnsembleVersion(
            version='2.01-rf',
            model_names=[
                "5532",
                "diagnosis_cbert_20210119_w64",
                "phi_diagnosis_bert_la_v1"
            ],
            voting_method=VotingMethodEnum.rf_classifier,
            voter_model="voter_rf_diagnosis_20210611"
        ),
    ]
    label_class = DiseaseSignSymptomLabel
    token_umls_representation_feat = {
        FeatureType.clinical: [
            SemTypeCtakesAsserted.DiseaseDisorder.name, SemTypeCtakesAsserted.SignSymptom.name]}
    required_representation_features = {FeatureType.covid_representation, FeatureType.clinical, FeatureType.icd10_diagnosis}


class DeidConstants(ModelConstants):
    ensemble_version_list = [
        EnsembleVersion(
            version='1.01',
            model_names=['10144'],
            voting_method=VotingMethodEnum.threshold_categories,
        )]
    label_class = PHILabel
    rare_varying_classes = [PHILabel.medicalrecord.value.column_index, PHILabel.username.value.column_index,
                            PHILabel.device.value.column_index,
                            PHILabel.bioid.value.column_index, PHILabel.idnum.value.column_index,
                            PHILabel.healthplan.value.column_index, PHILabel.url.value.column_index,
                            PHILabel.patient.value.column_index, PHILabel.phone.value.column_index,
                            PHILabel.street.value.column_index]
    required_representation_features = {FeatureType.date_comprehension}


class DemographicConstants(ModelConstants):
    label_class = DemographicEncounterLabel
    ensemble_version_list = [
        EnsembleVersion(
            version='1.00',
            model_names=['demographic_20201115_2d3', '5122', 'phi_demographic_20201214e4'],
            voting_method=VotingMethodEnum.threshold,
        ),
        EnsembleVersion(
            version='2.00',
            model_names=['5721', 'phi_demographic_2021_03_10_a'],
            voting_method=VotingMethodEnum.threshold,
        ),
    ]


class VitalSignConstants(ModelConstants):
    label_class = VitalSignsLabel
    ensemble_version_list = [
        EnsembleVersion(
            version='2.00',
            model_names=['phi_vital_signs_2021_03_26'],
            voting_method=VotingMethodEnum.weighted_entropy,
        )]
    required_representation_features = {
        FeatureType.date_comprehension
    }


class SDOHConstants(ModelConstants):
    label_class = SocialRiskFactorLabel
    ensemble_version_list = [
        EnsembleVersion(
            version='0.00',
            model_names=['phi_sdoh_20210719_v6'],
            voting_method=VotingMethodEnum.threshold,
        )]
    production = True


class ProcedureConstants(ModelConstants):
    label_class = ProcedureLabel
    ensemble_version_list = [
        EnsembleVersion(
            version='0.00',
            model_names=['phi_procedure_20210816_v2'],
            voting_method=VotingMethodEnum.model_avg,
        )]
    production = True


class LabConstants(ModelConstants):
    label_class = LabLabel
    ensemble_version_list = [
        EnsembleVersion(
            version='1.00',
            model_names=['5175', 'phi_lab_20201115_a1', '5183'],
            voting_method=VotingMethodEnum.weighted_entropy,  # ??? did this change?
        ),
        EnsembleVersion(
            version='1.01',
            model_names=['phi_lab_20210726_v1'],
            voting_method=VotingMethodEnum.model_avg,
        ),
        EnsembleVersion(
            version='1.02',
            model_names=['phi_lab_nmibc_covid_2021_08_13'],
            voting_method=VotingMethodEnum.model_avg,
        ),
        EnsembleVersion(
            version='1.03',
            model_names=['phi_lab_nmibc_covid_uc_2021_09_08'],
            voting_method=VotingMethodEnum.model_avg,
        ),
    ]
    token_umls_representation_feat = {FeatureType.lab_hepc: [SemTypeCtakesAsserted.Lab.name]}
    required_representation_features = {FeatureType.date_comprehension, FeatureType.lab_hepc}


class CovidLabConstants(LabConstants):
    # overwrite the LabConstants versions
    ensemble_version_list = [
        EnsembleVersion(
            version='1.00',
            model_names=['phi_covid_lab_20201014a1'],
            voting_method=VotingMethodEnum.threshold,
        ),
        EnsembleVersion(
            version='1.01',
            model_names=['phi_covid_labs_v2_2020_11_12_2'],
            voting_method=VotingMethodEnum.threshold,
        ),
    ]
    required_representation_features = {
        FeatureType.covid_representation,
        FeatureType.covid_lab_manufacturer,
        FeatureType.date_comprehension}


class DeviceProcedureConstants(ModelConstants):
    label_class = DeviceProcedureLabel
    ensemble_version_list = [
        EnsembleVersion(
            version='1.00',
            model_names=['phi_device_procedure_20201115_a2'],
            voting_method=VotingMethodEnum.threshold,
        ),
    ]
    token_umls_representation_feat = {FeatureType.tf_procedure: [SemTypeCtakesAsserted.Procedure.name]}
    required_representation_features = {FeatureType.tf_procedure}


class ImagingFindingConstants(ModelConstants):
    label_class = ImagingFindingLabel
    ensemble_version_list = [
        EnsembleVersion(
            version='2.00',
            model_names=['phi_imaging_finding_biomed_2106'],
            voting_method=VotingMethodEnum.threshold,
        ),
    ]


class OncologyConstants(ModelConstants):
    label_class = CancerLabel
    ensemble_version_list = [
        EnsembleVersion(
            version='2.00',
            model_names=['5586', 'onc_summary_5ep_lr0.001_a1'],
            voting_method=VotingMethodEnum.threshold,
        ),
        EnsembleVersion(
            version='2.10',
            model_names=['10059', 'onc_summary_5ep_lr0.001_a1'],
            voting_method=VotingMethodEnum.threshold,
        ),
        EnsembleVersion(
            version='3.00',
            model_names=['10118', 'oncology2246_lr00005_ep5_a4'],
            voting_method=VotingMethodEnum.threshold,
        ),
        EnsembleVersion(
            version='3.01',
            model_names=['10118', 'oncology2324_lr00005_ep5_a1'],
            voting_method=VotingMethodEnum.threshold,
        ),
        EnsembleVersion(
            version='nmibc_1.00',
            model_names=['oncology2324_lr00005_ep5_a1'],
            voting_method=VotingMethodEnum.threshold,
        ),
    ]
    token_umls_representation_feat = {
        FeatureType.morphology_code: [SemTypeCtakesAsserted.DiseaseDisorder.name],
        FeatureType.morphology: [SemTypeCtakesAsserted.DiseaseDisorder.name],
        FeatureType.topography: [SemTypeCtakesAsserted.AnatomicalSite.name],
        FeatureType.topography_code: [SemTypeCtakesAsserted.AnatomicalSite.name]}
    required_representation_features = {
        FeatureType.morphology, FeatureType.morphology_code, FeatureType.topography, FeatureType.topography_code}


class GeneticsConstants(ModelConstants):
    production = False
    label_class = GeneticsLabel
    ensemble_version_list = [
        EnsembleVersion(
            version='0.00',
            model_names=['10111', 'genetics2269_lr001_ep5_a1'],
            voting_method=VotingMethodEnum.threshold,
        )]


class SmokingConstants(ModelConstants):
    ensemble_version_list = [
        EnsembleVersion(
            version='0.00',
            model_names=['4320'],
            voting_method=VotingMethodEnum.threshold,  # VotingMethodEnum.model_avg
        )]
    label_class = SmokingLabel
    required_representation_features = {FeatureType.smoking}


class DrugConstants(ModelConstants):
    ensemble_version_list = [
        EnsembleVersion(
            version='2.00',
            model_names=['phi_drug_BIOMED_2106_d', '5592'],
            voting_method=VotingMethodEnum.threshold,
        ),
        EnsembleVersion(
            version='2.01',
            model_names=['phi_drug_BIOMED_2623_20210831_c', '5592'],
            voting_method=VotingMethodEnum.model_avg,
        )
    ]
    label_class = DrugLabel
    token_umls_representation_feat = {FeatureType.drug_rxnorm: [SemTypeCtakesAsserted.Medication.name]}
    required_representation_features = {FeatureType.date_comprehension, FeatureType.drug_rxnorm}
    rare_varying_classes = [DrugLabel.allergy.value.column_index]


class DocTypeConstants(ModelConstants):
    ensemble_version_list = [
        EnsembleVersion(
            version='0.00',
            model_names=['4710'],
            voting_method=VotingMethodEnum.model_avg,
        )]
    label_class = DocumentTypeLabel
    required_representation_features = {FeatureType.document_type}
    model_class_enum = ModelClass.doc_type


class DateOfServiceConstants(ModelConstants):
    ensemble_version_list = [
        EnsembleVersion(
            version='0.00',
            model_names=[],  # NOTE(mjp): why are we not listing the models here?
            voting_method=VotingMethodEnum.model_avg,
        )]
    model_class_enum: ModelClass = ModelClass.spacy


class BladderRiskConstants(ModelConstants):
    label_class = BladderRiskLabel
    ensemble_version_list = [
        EnsembleVersion(
            version='0.00',
            model_names=['2470_bladdder_risk_bert_a2'],
            voting_method=VotingMethodEnum.threshold,  # VotingMethodEnum.model_avg,
        )]
    model_class_enum: ModelClass = ModelClass.bert


# TODO: this should be transient
class SequioaBladderConstants(ModelConstants):
    label_class = SequoiaBladderLabel
    ensemble_version_list = [
        EnsembleVersion(
            version='0.00',
            model_names=['2607_sequoia_lstm_d1'],
            voting_method=VotingMethodEnum.threshold,  # VotingMethodEnum.model_avg,,
        )]
    model_class_enum: ModelClass = ModelClass.lstm_base


class MetaConstants(ModelConstants):
    production = False
    model_class_enum = ModelClass.meta


class DisabilityConstants(ModelConstants):
    production = False


class FamilyHistoryConstants(ModelConstants):
    label_class = FamilyHistoryLabel
    token_umls_representation_feat = {
        FeatureType.clinical: [SemTypeCtakesAsserted.DiseaseDisorder.name, SemTypeCtakesAsserted.SignSymptom.name]}
    ensemble_version_list = [
        EnsembleVersion(
            version='0.00',
            model_names=['phi_family_history_20210628_v1.3'],
            voting_method=VotingMethodEnum.model_avg,
        )]


# Map from ModelType to the "dataclass" with the associated model constants
MODEL_TYPE_2_CONSTANTS: Dict[ModelType, ModelConstants] = {
    ModelType.deid: DeidConstants,
    ModelType.vital_signs: VitalSignConstants,
    ModelType.device_procedure: DeviceProcedureConstants,
    ModelType.demographic: DemographicConstants,
    ModelType.lab: LabConstants,
    ModelType.covid_lab: CovidLabConstants,
    ModelType.disability: DisabilityConstants,
    ModelType.imaging_finding: ImagingFindingConstants,
    ModelType.oncology: OncologyConstants,
    ModelType.smoking: SmokingConstants,
    ModelType.drug: DrugConstants,
    ModelType.meta: MetaConstants,
    ModelType.diagnosis: DiagnosisConstants,
    ModelType.doc_type: DocTypeConstants,
    ModelType.date_of_service: DateOfServiceConstants,
    ModelType.genetics: GeneticsConstants,
    ModelType.bladder_risk: BladderRiskConstants,
    ModelType.sequioa_bladder: SequioaBladderConstants,
    ModelType.family_history: FamilyHistoryConstants,
    ModelType.sdoh: SDOHConstants,
    ModelType.procedure: ProcedureConstants

}
