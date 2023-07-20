from text2phenotype.constants.features.label_types import *


annotator_to_tag_tog_username= {

    'anne.frea': 'afrea',
    'christine.persaud': 'cpersaud',
    'despina.siolas': 'dsiolas',
    'satjiv.kohli': 'skohli',
    'nick.colangelo': 'CEPUser',
    'Briana.Galloway': 'CEPUser',
    'briana.galloway': 'CEPUser',
}


annotation_label_to_tag_tog_project_folder = {
    ProblemLabel: 'diagnosis_signsymptom_validation',
    SignSymptomLabel: 'diagnosis_signsymptom_validation',


    AllergyLabel: 'biomed_clinical_summary',
    MedLabel: 'biomed_clinical_summary',

    LabLabel: 'lab_validation',

    CancerLabel: 'oncology_summary',
    GeneticTestLabel: 'oncology_summary',

    EventDateLabel: 'document_type_event_date', # TODO FIX THIS

    DeviceProcedureLabel: 'Covid_Specific',
    CovidLabLabel: 'Covid_Specific',
    FindingLabel: 'Covid_Specific',

    VitalSignsLabel: 'vital_signs',

    SmokingLabel: 'smoking_status',

    DemographicEncounterLabel: 'demographics',

    DiagnosticImagingLabel: 'diagnostic_imaging_studies'
}



tag_tog_project_folders = [
    'Covid_Specific', 'lab_validation', 'document_type_event_date', 'device_procedure',
    'diagnosis_signsymptom_validation', 'biomed_clinical_summary', 'vital_signs', 'smoking_status', 'demographics',
                            'diagnostic_imaging_studies']



special_reqs = ['lab_validation', 'document_type_event_date', 'Covid_Specific']

