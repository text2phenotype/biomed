import unittest

from biomed.common.helpers import feature_list_helper
from biomed.deid.deid import get_phi_tokens
from biomed.demographic.demographic import get_demographic_tokens
from biomed.reassembler.reassemble_functions import reassemble_demographics, reassemble_phi_token_chunk_results
from text2phenotype.apiclients.feature_service import FeatureServiceClient
from text2phenotype.common import common

from biomed.deid.global_redaction_helper_functions import global_redact_and_filter
from biomed.tests.fixtures.example_file_paths import carolyn_blose_txt_filepath
from biomed.deid.global_redaction_helper_functions import redact_text
from text2phenotype.constants.deid import DeidGroupings
from text2phenotype.tasks.task_enums import TaskOperation, TaskEnum


class TestGlobalRedaction(unittest.TestCase):
    TEXT = common.read_text(carolyn_blose_txt_filepath)
    maxDiff=None
    fs_client = FeatureServiceClient()
    ANNOTATIONS, VECTORS = fs_client.annotate_vectorize(
        text=TEXT,
        features=feature_list_helper({TaskOperation.phi_tokens, TaskOperation.demographics}))
    DEMOGRAPHICS_TOKENS = get_demographic_tokens(tokens=ANNOTATIONS, vectors=VECTORS)
    DEMOGRAPHICS_JSON = reassemble_demographics([((0, len(TEXT)), DEMOGRAPHICS_TOKENS)])
    phi_tokens = get_phi_tokens(tokens=ANNOTATIONS, vectors=VECTORS)
    PHI_TOKENS = reassemble_phi_token_chunk_results(
        chunk_mapping=[((0, len(TEXT)), phi_tokens)],
        other_enum_to_chunk_mapping={TaskEnum.demographics: [((0, len(TEXT)), DEMOGRAPHICS_TOKENS)]}
    )

    def test_safe_harbor_redaction(self):
        final_redaction_summary = global_redact_and_filter(
            demographic_json=self.DEMOGRAPHICS_JSON,
            phi_token_json=self.PHI_TOKENS,
            machine_annotation=self.ANNOTATIONS)
        carolyn_blose_safe_harbor_text = redact_text(final_redaction_summary, self.TEXT)
        expected_text = """**** 1 of 3
PATIENT:*****, ******* - ********** Generated on ********** - ******* ****** ************
******* ******
************
******* *****
Generated on: **********
Patient Data:
Who Name: Mr. ******* ** ***** External ID: ***********
DOB: **********
Sex: Female
History Data:
General Risk Factors:
Hypertension
Exams/Tests:
Lifestyle Tobacco:
Never smoker ( SNOMED-CT:********* ) nonsmoker Status: Never
Coffee:
Alcohol:
not consume alcohol
Recreational Drugs: No history of recreational drug use.
Counseling:
Exercise Patterns:
Hazardous Activities:
Insurance Data:
Primary Insurance Data:
Subscriber Date of Birth:
**********
Secondary Insurance Data:
Subscriber Date of Birth:
**********
Tertiary Insurance Data:
Subscriber Date of Birth:
**********
Billing Information:
Patient Immunization:
Patient Notes:
**********: Admitting diagnoses: Bloody diarrhea Shortness of breath congestive heart failure Discharge diagnosis:
Diverticulitis Valvular insufficiency Congestive Heart Failure HISTORY OF PRESENT ILLNESS: The patient is an
*********** female admitted for evaluation of abdominal pain and bloody stools. The patient has colitis and also
diverticulitis, undergoing treatment. During the hospitalization, the patient complained of shortness of breath, which
http://demo.visolve.com/openemr-demo/interface/patient file/report/custom report.php?p... *********
Page 2 of 3
worsened. The patient underwent an echocardiogram, which shows severe mitral regurgitation and also large pleural
effusion. Per the patient, she has limited activity level with shortness of breath for many years. She also was told that
she has a heart murmur, which was not followed through for a diagnosis. CORONARY RISK FACTORS: History of
hypertension, no history of diabetes mellitus, nonsmoker, cholesterol status unclear, no prior history of coronary artery
disease, and family history noncontributory. FAMILY HISTORY: Nonsignificant. PAST SURGICAL HISTORY: No
major surgery. MEDICATIONS: Presently on Lasix, potassium supplementation, Levaquin, hydralazine 10 mg b.i.d.,
antibiotic treatments, and thyroid supplementation. ALLERGIES: AMBIEN, CARDIZEM, AND IBUPROFEN.
PERSONAL HISTORY: She is a nonsmoker. Does not consume alcohol. No history of recreational drug use. PAST
MEDICAL HISTORY: Basically GI pathology with diverticulitis, colitis, hypothyroidism, arthritis, questionable
hypertension, no prior history of coronary artery disease, and heart murmur. REVIEW OF SYSTEMS
CONSTITUTIONAL: Weakness, fatigue, and tiredness. HEENT: History of cataract, blurred vision, and hearing
impairment. CARDIOVASCULAR: Shortness of breath and heart murmur. No coronary artery disease.
RESPIRATORY: Shortness of breath. No pneumonia or valley fever. GASTROINTESTINAL: No nausea, vomiting,
hematemesis, or melena. UROLOGICAL: No frequency or urgency. MUSCULOSKELETAL: Arthritis and severe
muscle weakness. SKIN: Nonsignificant. NEUROLOGICAL: No TIA or CVA. No seizure disorder.
ENDOCRINE/HEMATOLOGICAL: As above. PHYSICAL EXAMINATION VITAL SIGNS: Pulse of 84, blood
pressure of 168/74, afebrile, and respiratory rate 16 per minute. HEENT/NECK: Head is atraumatic and normocephalic.
Neck veins flat. No significant carotid bruits appreciated. LUNGS: Air entry bilaterally fair. No obvious rales or
wheezes. HEART: PMI displaced. S1, S2 with systolic murmur at the precordium, grade 2/6. ABDOMEN: Soft and
nontender. EXTREMITIES: Chronic skin changes. Feeble pulses distally. No clubbing or cyanosis. DIAGNOSTIC
DATA: EKG: Normal sinus rhythm. No acute ST-T changes. Echocardiogram was reviewed showing severe mitral
regurgitation. LABORATORY DATA: H&H 13 and 39. BUN and creatinine within normal limits. Potassium within
normal limits. BNP 9290. IMPRESSION: 1. The patient admitted for gastrointestinal pathology, under working
treatment. 2. History of prior heart murmur with Echocardiogram findings as above. Basically revealed normal left
ventricular function with left atrial enlargement, large pleural effusion, and severe mitral regurgitation and tricuspid
regurgitation. Discharge Recommendations: 1. Finish antibiotics, High fiber diet for diverticulitis. PCP follow up in 30
days 2. From cardiac standpoint, conservative treatment. Possibility of a transesophageal echocardiogram to assess
valvular insufficiency adequately well discussed extensively. After extensive discussion, given her age **, limited
activity level, and no intention of undergoing any treatment in this regard from a surgical standpoint, the patient does
not wish to proceed with a transesophageal echocardiogram. 3. Based on the above findings, we will treat her cardiac
disease medically with ACE inhibitors and diuretics and see how she fares. She has a normal LV function.
Patient Transactions:
::::::::::::::::::::****************************
Patient Communication sent:
Recurrent Appointments:
None
Issues
Allergies:
AMBIEN:
CARDIZEM:
IBUPROFEN:
Medications:
Lasix:
potassium supplementation:
Levaquin:
hydralazine 10 mg b.i.d.:
antibiotic treatments:
thyroid supplementation:
.....................................................................................................................................
http://demo.visolve.com/openemr-demo/interface/patient file/report/custom report.php?p... *********
Page 3 of 3
New Patient Encounter
(**********) Provider: Administrator Administrator
Facility: ******* ******
Reason: HISTORY OF PRESENT ILLNESS: The patient is an *********** female admitted for evaluation of abdominal
pain and bloody stools. The patient has colitis and also diverticulitis, undergoing treatment. During the hospitalization, the
patient complained of shortness of breath, which worsened. The patient underwent an echocardiogram, which shows
severe mitral regurgitation and also large pleural effusion. Per the patient, she has limited activity level with shortness of
breath for many years. She also was told that she has a heart murmur, which was not followed through for a diagnosis.
Review Of Systems
(**********)
Weakness: Fatigue:
YES
YES
Shortness Of Breath: History Murmur:
YES
YES
Vomiting:
YES
Hematemesis:
YES
TIA: NO
Vitals
(**********)
Blood Pressure: 168/74 Pulse: 84 per min Respiration: 16 per min
Review of Systems Checks
(**********)
Cataracts: yes Blurred Vision: yes Shortness Of Breath: yes
Signature:
http://demo.visolve.com/openemr-demo/interface/patient file/report/custom report.php?p... *********
"""
        self.assertEqual(carolyn_blose_safe_harbor_text, expected_text)

    def test_lds_date(self):
        final_redaction_summary = global_redact_and_filter(
            demographic_json=self.DEMOGRAPHICS_JSON,
            phi_token_json=self.PHI_TOKENS,
            machine_annotation=self.ANNOTATIONS,
            phi_categories_to_include=DeidGroupings.LDS_DATE.value
        )
        carolyn_blose_lds_date = redact_text(final_redaction_summary, self.TEXT)
        expected_text = """**** 1 of 3
PATIENT:*****, ******* - ********** Generated on 2017-07-21 - ******* ****** ************
******* ******
************
******* *****
Generated on: 2017-07-21
Patient Data:
Who Name: Mr. ******* ** ***** External ID: ***********
DOB: **********
Sex: Female
History Data:
General Risk Factors:
Hypertension
Exams/Tests:
Lifestyle Tobacco:
Never smoker ( SNOMED-CT:********* ) nonsmoker Status: Never
Coffee:
Alcohol:
not consume alcohol
Recreational Drugs: No history of recreational drug use.
Counseling:
Exercise Patterns:
Hazardous Activities:
Insurance Data:
Primary Insurance Data:
Subscriber Date of Birth:
0000-00-00
Secondary Insurance Data:
Subscriber Date of Birth:
0000-00-00
Tertiary Insurance Data:
Subscriber Date of Birth:
0000-00-00
Billing Information:
Patient Immunization:
Patient Notes:
2017-07-19: Admitting diagnoses: Bloody diarrhea Shortness of breath congestive heart failure Discharge diagnosis:
Diverticulitis Valvular insufficiency Congestive Heart Failure HISTORY OF PRESENT ILLNESS: The patient is an
*********** female admitted for evaluation of abdominal pain and bloody stools. The patient has colitis and also
diverticulitis, undergoing treatment. During the hospitalization, the patient complained of shortness of breath, which
http://demo.visolve.com/openemr-demo/interface/patient file/report/custom report.php?p... 7/21/2017
Page 2 of 3
worsened. The patient underwent an echocardiogram, which shows severe mitral regurgitation and also large pleural
effusion. Per the patient, she has limited activity level with shortness of breath for many years. She also was told that
she has a heart murmur, which was not followed through for a diagnosis. CORONARY RISK FACTORS: History of
hypertension, no history of diabetes mellitus, nonsmoker, cholesterol status unclear, no prior history of coronary artery
disease, and family history noncontributory. FAMILY HISTORY: Nonsignificant. PAST SURGICAL HISTORY: No
major surgery. MEDICATIONS: Presently on Lasix, potassium supplementation, Levaquin, hydralazine 10 mg b.i.d.,
antibiotic treatments, and thyroid supplementation. ALLERGIES: AMBIEN, CARDIZEM, AND IBUPROFEN.
PERSONAL HISTORY: She is a nonsmoker. Does not consume alcohol. No history of recreational drug use. PAST
MEDICAL HISTORY: Basically GI pathology with diverticulitis, colitis, hypothyroidism, arthritis, questionable
hypertension, no prior history of coronary artery disease, and heart murmur. REVIEW OF SYSTEMS
CONSTITUTIONAL: Weakness, fatigue, and tiredness. HEENT: History of cataract, blurred vision, and hearing
impairment. CARDIOVASCULAR: Shortness of breath and heart murmur. No coronary artery disease.
RESPIRATORY: Shortness of breath. No pneumonia or valley fever. GASTROINTESTINAL: No nausea, vomiting,
hematemesis, or melena. UROLOGICAL: No frequency or urgency. MUSCULOSKELETAL: Arthritis and severe
muscle weakness. SKIN: Nonsignificant. NEUROLOGICAL: No TIA or CVA. No seizure disorder.
ENDOCRINE/HEMATOLOGICAL: As above. PHYSICAL EXAMINATION VITAL SIGNS: Pulse of 84, blood
pressure of 168/74, afebrile, and respiratory rate 16 per minute. HEENT/NECK: Head is atraumatic and normocephalic.
Neck veins flat. No significant carotid bruits appreciated. LUNGS: Air entry bilaterally fair. No obvious rales or
wheezes. HEART: PMI displaced. S1, S2 with systolic murmur at the precordium, grade 2/6. ABDOMEN: Soft and
nontender. EXTREMITIES: Chronic skin changes. Feeble pulses distally. No clubbing or cyanosis. DIAGNOSTIC
DATA: EKG: Normal sinus rhythm. No acute ST-T changes. Echocardiogram was reviewed showing severe mitral
regurgitation. LABORATORY DATA: H&H 13 and 39. BUN and creatinine within normal limits. Potassium within
normal limits. BNP 9290. IMPRESSION: 1. The patient admitted for gastrointestinal pathology, under working
treatment. 2. History of prior heart murmur with Echocardiogram findings as above. Basically revealed normal left
ventricular function with left atrial enlargement, large pleural effusion, and severe mitral regurgitation and tricuspid
regurgitation. Discharge Recommendations: 1. Finish antibiotics, High fiber diet for diverticulitis. PCP follow up in 30
days 2. From cardiac standpoint, conservative treatment. Possibility of a transesophageal echocardiogram to assess
valvular insufficiency adequately well discussed extensively. After extensive discussion, given her age **, limited
activity level, and no intention of undergoing any treatment in this regard from a surgical standpoint, the patient does
not wish to proceed with a transesophageal echocardiogram. 3. Based on the above findings, we will treat her cardiac
disease medically with ACE inhibitors and diuretics and see how she fares. She has a normal LV function.
Patient Transactions:
::::::::::::::::::::****************************
Patient Communication sent:
Recurrent Appointments:
None
Issues
Allergies:
AMBIEN:
CARDIZEM:
IBUPROFEN:
Medications:
Lasix:
potassium supplementation:
Levaquin:
hydralazine 10 mg b.i.d.:
antibiotic treatments:
thyroid supplementation:
.....................................................................................................................................
http://demo.visolve.com/openemr-demo/interface/patient file/report/custom report.php?p... 7/21/2017
Page 3 of 3
New Patient Encounter
(2017-07-19) Provider: Administrator Administrator
Facility: ******* ******
Reason: HISTORY OF PRESENT ILLNESS: The patient is an *********** female admitted for evaluation of abdominal
pain and bloody stools. The patient has colitis and also diverticulitis, undergoing treatment. During the hospitalization, the
patient complained of shortness of breath, which worsened. The patient underwent an echocardiogram, which shows
severe mitral regurgitation and also large pleural effusion. Per the patient, she has limited activity level with shortness of
breath for many years. She also was told that she has a heart murmur, which was not followed through for a diagnosis.
Review Of Systems
(2017-07-19)
Weakness: Fatigue:
YES
YES
Shortness Of Breath: History Murmur:
YES
YES
Vomiting:
YES
Hematemesis:
YES
TIA: NO
Vitals
(2017-07-19)
Blood Pressure: 168/74 Pulse: 84 per min Respiration: 16 per min
Review of Systems Checks
(2017-07-19)
Cataracts: yes Blurred Vision: yes Shortness Of Breath: yes
Signature:
http://demo.visolve.com/openemr-demo/interface/patient file/report/custom report.php?p... 7/21/2017
"""
        self.assertEqual(carolyn_blose_lds_date, expected_text)

    def test_lds_provider_date(self):
        final_redaction_summary = global_redact_and_filter(
            demographic_json=self.DEMOGRAPHICS_JSON,
            phi_token_json=self.PHI_TOKENS,
            machine_annotation=self.ANNOTATIONS,
            phi_categories_to_include=DeidGroupings.LDS_DATE_PROVIDER.value
        )
        carolyn_blose_lds_date_provider = redact_text(final_redaction_summary, self.TEXT)
        expected_text = """Page 1 of 3
PATIENT:*****, ******* - ********** Generated on 2017-07-21 - ViSolve Clinic ************
ViSolve Clinic
************
******* *****
Generated on: 2017-07-21
Patient Data:
Who Name: Mr. ******* ** ***** External ID: ***********
DOB: **********
Sex: Female
History Data:
General Risk Factors:
Hypertension
Exams/Tests:
Lifestyle Tobacco:
Never smoker ( SNOMED-CT:********* ) nonsmoker Status: Never
Coffee:
Alcohol:
not consume alcohol
Recreational Drugs: No history of recreational drug use.
Counseling:
Exercise Patterns:
Hazardous Activities:
Insurance Data:
Primary Insurance Data:
Subscriber Date of Birth:
0000-00-00
Secondary Insurance Data:
Subscriber Date of Birth:
0000-00-00
Tertiary Insurance Data:
Subscriber Date of Birth:
0000-00-00
Billing Information:
Patient Immunization:
Patient Notes:
2017-07-19: Admitting diagnoses: Bloody diarrhea Shortness of breath congestive heart failure Discharge diagnosis:
Diverticulitis Valvular insufficiency Congestive Heart Failure HISTORY OF PRESENT ILLNESS: The patient is an
*********** female admitted for evaluation of abdominal pain and bloody stools. The patient has colitis and also
diverticulitis, undergoing treatment. During the hospitalization, the patient complained of shortness of breath, which
http://demo.visolve.com/openemr-demo/interface/patient file/report/custom report.php?p... 7/21/2017
Page 2 of 3
worsened. The patient underwent an echocardiogram, which shows severe mitral regurgitation and also large pleural
effusion. Per the patient, she has limited activity level with shortness of breath for many years. She also was told that
she has a heart murmur, which was not followed through for a diagnosis. CORONARY RISK FACTORS: History of
hypertension, no history of diabetes mellitus, nonsmoker, cholesterol status unclear, no prior history of coronary artery
disease, and family history noncontributory. FAMILY HISTORY: Nonsignificant. PAST SURGICAL HISTORY: No
major surgery. MEDICATIONS: Presently on Lasix, potassium supplementation, Levaquin, hydralazine 10 mg b.i.d.,
antibiotic treatments, and thyroid supplementation. ALLERGIES: AMBIEN, CARDIZEM, AND IBUPROFEN.
PERSONAL HISTORY: She is a nonsmoker. Does not consume alcohol. No history of recreational drug use. PAST
MEDICAL HISTORY: Basically GI pathology with diverticulitis, colitis, hypothyroidism, arthritis, questionable
hypertension, no prior history of coronary artery disease, and heart murmur. REVIEW OF SYSTEMS
CONSTITUTIONAL: Weakness, fatigue, and tiredness. HEENT: History of cataract, blurred vision, and hearing
impairment. CARDIOVASCULAR: Shortness of breath and heart murmur. No coronary artery disease.
RESPIRATORY: Shortness of breath. No pneumonia or valley fever. GASTROINTESTINAL: No nausea, vomiting,
hematemesis, or melena. UROLOGICAL: No frequency or urgency. MUSCULOSKELETAL: Arthritis and severe
muscle weakness. SKIN: Nonsignificant. NEUROLOGICAL: No TIA or CVA. No seizure disorder.
ENDOCRINE/HEMATOLOGICAL: As above. PHYSICAL EXAMINATION VITAL SIGNS: Pulse of 84, blood
pressure of 168/74, afebrile, and respiratory rate 16 per minute. HEENT/NECK: Head is atraumatic and normocephalic.
Neck veins flat. No significant carotid bruits appreciated. LUNGS: Air entry bilaterally fair. No obvious rales or
wheezes. HEART: PMI displaced. S1, S2 with systolic murmur at the precordium, grade 2/6. ABDOMEN: Soft and
nontender. EXTREMITIES: Chronic skin changes. Feeble pulses distally. No clubbing or cyanosis. DIAGNOSTIC
DATA: EKG: Normal sinus rhythm. No acute ST-T changes. Echocardiogram was reviewed showing severe mitral
regurgitation. LABORATORY DATA: H&H 13 and 39. BUN and creatinine within normal limits. Potassium within
normal limits. BNP 9290. IMPRESSION: 1. The patient admitted for gastrointestinal pathology, under working
treatment. 2. History of prior heart murmur with Echocardiogram findings as above. Basically revealed normal left
ventricular function with left atrial enlargement, large pleural effusion, and severe mitral regurgitation and tricuspid
regurgitation. Discharge Recommendations: 1. Finish antibiotics, High fiber diet for diverticulitis. PCP follow up in 30
days 2. From cardiac standpoint, conservative treatment. Possibility of a transesophageal echocardiogram to assess
valvular insufficiency adequately well discussed extensively. After extensive discussion, given her age **, limited
activity level, and no intention of undergoing any treatment in this regard from a surgical standpoint, the patient does
not wish to proceed with a transesophageal echocardiogram. 3. Based on the above findings, we will treat her cardiac
disease medically with ACE inhibitors and diuretics and see how she fares. She has a normal LV function.
Patient Transactions:
::::::::::::::::::::****************************
Patient Communication sent:
Recurrent Appointments:
None
Issues
Allergies:
AMBIEN:
CARDIZEM:
IBUPROFEN:
Medications:
Lasix:
potassium supplementation:
Levaquin:
hydralazine 10 mg b.i.d.:
antibiotic treatments:
thyroid supplementation:
.....................................................................................................................................
http://demo.visolve.com/openemr-demo/interface/patient file/report/custom report.php?p... 7/21/2017
Page 3 of 3
New Patient Encounter
(2017-07-19) Provider: Administrator Administrator
Facility: ViSolve Clinic
Reason: HISTORY OF PRESENT ILLNESS: The patient is an *********** female admitted for evaluation of abdominal
pain and bloody stools. The patient has colitis and also diverticulitis, undergoing treatment. During the hospitalization, the
patient complained of shortness of breath, which worsened. The patient underwent an echocardiogram, which shows
severe mitral regurgitation and also large pleural effusion. Per the patient, she has limited activity level with shortness of
breath for many years. She also was told that she has a heart murmur, which was not followed through for a diagnosis.
Review Of Systems
(2017-07-19)
Weakness: Fatigue:
YES
YES
Shortness Of Breath: History Murmur:
YES
YES
Vomiting:
YES
Hematemesis:
YES
TIA: NO
Vitals
(2017-07-19)
Blood Pressure: 168/74 Pulse: 84 per min Respiration: 16 per min
Review of Systems Checks
(2017-07-19)
Cataracts: yes Blurred Vision: yes Shortness Of Breath: yes
Signature:
http://demo.visolve.com/openemr-demo/interface/patient file/report/custom report.php?p... 7/21/2017
"""
        self.assertEqual(carolyn_blose_lds_date_provider, expected_text)
