from text2phenotype.constants.features.label_types import DocumentTypeLabel

mimic_to_text2phenotype_mapping = {
    "Discharge summary": DocumentTypeLabel.dischargesummary,
    "Radiology": DocumentTypeLabel.diagnosticimagingstudy,
    "ECG": DocumentTypeLabel.diagnosticimagingstudy,
    "Nursing/other": DocumentTypeLabel.nursing,
    "Echo": DocumentTypeLabel.diagnosticimagingstudy,
    "Physician": DocumentTypeLabel.historyandphysical,
    "Nursing": DocumentTypeLabel.nursing,
    "Rehab Services": DocumentTypeLabel.progressnote,
    "Respiratory": DocumentTypeLabel.progressnote,
    "Consult": DocumentTypeLabel.consultnote,
}

MimicDocumentTypeLabel = ["Discharge summary", "Radiology", "ECG", "Nursing/other", "Echo", "Physician", "Nursing",
                          "General", "Nutrition", "Rehab Services", "Respiratory", "Social Work", "Case Management",
                          "Pharmacy", "Consult"]
