import os

from biomed.biomed_env import BiomedEnv

############################################################################################
#
# HEPC Hepatitis C Virus Infection (HCV)
#
############################################################################################
RICARDO_HPI_PDF = os.path.join(BiomedEnv.DATA_ROOT.value, 'himss', '1-vista-pcp-note', '1-vista-pcp-note.pdf')
RICARDO_HPI_TXT = os.path.join(BiomedEnv.DATA_ROOT.value, 'himss', '1-vista-pcp-note', '1-vista-pcp-note.txt')

RICARDO_LAB_COMMON_TXT = os.path.join(BiomedEnv.DATA_ROOT.value, 'himss', '0-order-lab-common-ccd', '0-order-lab-common-ccd.xml.html.txt')
RICARDO_CT_HCV_TXT = os.path.join(BiomedEnv.DATA_ROOT.value, 'himss', '1-order-ct-hcv-ccd', '1-order-ct-hcv-ccd.xml.html.txt')
RICARDO_LAB_HCV_TXT = os.path.join(BiomedEnv.DATA_ROOT.value, 'himss', '1-order-lab-hcv-ccd', '1-order-lab-hcv-ccd.xml.html.txt')

RICARDO_FINAL_TXT = os.path.join(BiomedEnv.DATA_ROOT.value, 'himss', '6-text2phenotype-smart', '6-text2phenotype-smart.txt')

HCV_CONSULT_NOTE_PDF = os.path.join(BiomedEnv.DATA_ROOT.value, 'emr', 'echo', 'Cerner', 'ConsultNote_HCV.pdf')
HCV_CONSULT_NOTE_TXT = f"{HCV_CONSULT_NOTE_PDF}.txt"

############################################################################################
#
# OpenEMR: Stephan Garcia
#
############################################################################################
OPEN_EMR_DIR = os.path.join(BiomedEnv.DATA_ROOT.value, 'emr', 'OpenEMR')

STEPHAN_GARCIA_PDF = os.path.join(OPEN_EMR_DIR, 'stephan-garcia.pdf')
TINA_MARMOL_PDF = os.path.join(OPEN_EMR_DIR, 'tina-mormol.pdf')
JOHN_STEVENS_PDF = os.path.join(OPEN_EMR_DIR, 'john-stevens.pdf')
DAVID_VAUGHN_PDF = os.path.join(OPEN_EMR_DIR, 'david-vaughan.pdf')
CAROLYN_BLOSE_PDF = os.path.join(OPEN_EMR_DIR, 'carolyn-blose.pdf')

STEPHAN_GARCIA_TXT = f"{STEPHAN_GARCIA_PDF}.txt"
TINA_MARMOL_TXT = f"{TINA_MARMOL_PDF}.txt"
JOHN_STEVENS_TXT = f"{JOHN_STEVENS_PDF}.txt"
DAVID_VAUGHN_TXT = f"{DAVID_VAUGHN_PDF}.txt"
CAROLYN_BLOSE_TXT = f"{CAROLYN_BLOSE_PDF}.txt"

############################################################################################
#
# MTSAMPLES
#
############################################################################################

MTSAMPLES_DIR = os.path.join(BiomedEnv.DATA_ROOT.value, 'mtsamples', 'clean')

############################################################################################
#
# REINSURANCE -- "400 page documents"
#
############################################################################################
REINSURANCE_DIR = os.path.join(BiomedEnv.DATA_ROOT.value, 'Example', 'ReInsurance', 'biomed')
REINSURANCE_SHORTER_TXT = f"{SWISS_RE_DIR}/shorter/shorter_text.txt"
REINSURANCE_LONGER_TXT = f"{SWISS_RE_DIR}/longer/longer_text.txt"

############################################################################################
#
# Cancer
#
############################################################################################
CANCER_DIR = os.path.join(BiomedEnv.DATA_ROOT.value, 'Example', 'Roche')

CANCER_SUMMARY_PDF = os.path.join(CANCER_DIR,
                                        'A2_Anon_Unstructured_Data',
                                        'Material1',
                                        'Steven_Keating_Health_Summary_Report.pdf')
CANCER_SUMMARY_TXT = f"{CANCER_SUMMARY_PDF}.txt"

############################################################################################
#
# CMS BLUEBUTTON
#
############################################################################################
BLUEBUTTON_TEST_DIR = os.path.join(BiomedEnv.DATA_ROOT.value, 'Example', 'bluebutton', 'Test Data')
BLUEBUTTON_RECORDS_DIR = os.path.join(BLUEBUTTON_TEST_DIR, 'Sample Medical Records')
BLUEBUTTON_RADV_FILE = os.path.join(BLUEBUTTON_TEST_DIR,
                                  'Risk Adjustment Data Validation (RADV)',
                                  'RADV Sample CMS-Generated Attestation.txt')

############################################################################################
#
# FHIR
#
############################################################################################
LAB_RESULT_JSON = os.path.join(BiomedEnv.DATA_ROOT.value, 'fhir', 'stephan-garcia', 'DiagnosticReport-bmp.fhir.json')
LAB_RESULT_TXT = f"{LAB_RESULT_JSON}.txt"

############################################################################################
#
# SHRINE
#
############################################################################################

# Aspect.Lab
SHRINE_LOINC_BSV = os.path.join(BiomedEnv.DATA_ROOT.value, 'shrine', 'HMS_SHRINE_ONTOLOGY_FLAT_LOINC.bsv')

# Aspect.medication
SHRINE_RXNORM_BSV = os.path.join(BiomedEnv.DATA_ROOT.value, 'shrine', 'HMS_SHRINE_ONTOLOGY_FLAT_RXNORM.bsv')
SHRINE_NDFRT_BSV = os.path.join(BiomedEnv.DATA_ROOT.value, 'shrine', 'HMS_SHRINE_ONTOLOGY_FLAT_NDFRT.bsv')

# Aspect.diagnosis | Aspect.problem
SHRINE_ICD9_BSV = os.path.join(BiomedEnv.DATA_ROOT.value, 'shrine', 'HMS_SHRINE_ONTOLOGY_FLAT_ICD9.bsv')
SHRINE_ICD10_BSV = os.path.join(BiomedEnv.DATA_ROOT.value, 'shrine', 'HMS_SHRINE_ONTOLOGY_FLAT_ICD10.bsv')

# Aspect.demographics
SHRINE_RACE_BSV = os.path.join(BiomedEnv.DATA_ROOT.value, 'shrine', 'HMS_SHRINE_ONTOLOGY_FLAT_RACE.bsv')

# Aspect.immunization
VACCINE_CSX_BSV = os.path.join(BiomedEnv.DATA_ROOT.value, 'umls', 'cvx', 'cvx.bsv')


############################################################################################
#
# PUBMED
#
############################################################################################
PUBMED_DIR = os.path.join(BiomedEnv.DATA_ROOT.value, 'pubmed')
PUBMED_MEDLINE_DIR = os.path.join(PUBMED_DIR, 'medline')
