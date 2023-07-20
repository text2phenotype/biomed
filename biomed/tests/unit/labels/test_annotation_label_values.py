import enum
import unittest

from text2phenotype.constants.features.label_types import (
    AllergyLabel,
    CancerLabel,
    DemographicEncounterLabel,
    DisabilityLabel,
    LabLabel,
    MedLabel,
    PHILabel,
    ProblemLabel,
    SignSymptomLabel,
    GeneticsLabel
)


class TestLabel(unittest.TestCase):
    def db_range_of_ints(self, labelEnum: enum):
        i = 0
        for entry in labelEnum:
            if entry.value.column_index != i:
                return False
            i += 1
        return True

    def test_demographic_label(self):
        self.assertTrue(isinstance(DemographicEncounterLabel.to_dict(), dict))
        self.assertTrue(self.db_range_of_ints(DemographicEncounterLabel))
        self.assertEqual(DemographicEncounterLabel.na.value.column_index, 0)
        self.assertEqual(DemographicEncounterLabel.ssn.value.column_index, 1)
        self.assertEqual(DemographicEncounterLabel.mrn.value.column_index, 2)
        self.assertEqual(DemographicEncounterLabel.pat_first.value.column_index, 3)
        self.assertEqual(DemographicEncounterLabel.pat_middle.value.column_index, 4)
        self.assertEqual(DemographicEncounterLabel.pat_last.value.column_index, 5)
        self.assertEqual(DemographicEncounterLabel.pat_initials.value.column_index, 6)
        self.assertEqual(DemographicEncounterLabel.pat_age.value.column_index, 7)
        self.assertEqual(DemographicEncounterLabel.pat_street.value.column_index, 8)
        self.assertEqual(DemographicEncounterLabel.pat_zip.value.column_index, 9)
        self.assertEqual(DemographicEncounterLabel.pat_city.value.column_index, 10)
        self.assertEqual(DemographicEncounterLabel.pat_state.value.column_index, 11)
        self.assertEqual(DemographicEncounterLabel.pat_phone.value.column_index, 12)
        self.assertEqual(DemographicEncounterLabel.pat_email.value.column_index, 13)
        self.assertEqual(DemographicEncounterLabel.insurance.value.column_index, 14)
        self.assertEqual(DemographicEncounterLabel.facility_name.value.column_index, 15)
        self.assertEqual(DemographicEncounterLabel.dr_first.value.column_index, 16)
        self.assertEqual(DemographicEncounterLabel.dr_middle.value.column_index, 17)
        self.assertEqual(DemographicEncounterLabel.dr_last.value.column_index, 18)
        self.assertEqual(DemographicEncounterLabel.dr_initials.value.column_index, 19)
        self.assertEqual(DemographicEncounterLabel.dr_street.value.column_index, 20)
        self.assertEqual(DemographicEncounterLabel.dr_zip.value.column_index, 21)
        self.assertEqual(DemographicEncounterLabel.dr_city.value.column_index, 22)
        self.assertEqual(DemographicEncounterLabel.dr_state.value.column_index, 23)
        self.assertEqual(DemographicEncounterLabel.dr_phone.value.column_index, 24)
        self.assertEqual(DemographicEncounterLabel.dr_fax.value.column_index, 25)
        self.assertEqual(DemographicEncounterLabel.dr_email.value.column_index, 26)
        self.assertEqual(DemographicEncounterLabel.dr_id.value.column_index, 27)
        self.assertEqual(DemographicEncounterLabel.dr_org.value.column_index, 28)
        self.assertEqual(DemographicEncounterLabel.sex.value.column_index, 29)
        self.assertEqual(DemographicEncounterLabel.dob.value.column_index, 30)

    def test_deid_label(self):
        self.assertTrue(isinstance(PHILabel.to_dict(), dict))
        self.assertTrue(self.db_range_of_ints(PHILabel))
        self.assertEqual(PHILabel.na.value.column_index, 0)
        self.assertEqual(PHILabel.date.value.column_index, 1)
        self.assertEqual(PHILabel.hospital.value.column_index, 2)
        self.assertEqual(PHILabel.age.value.column_index, 3)
        self.assertEqual(PHILabel.street.value.column_index, 4)
        self.assertEqual(PHILabel.zip.value.column_index, 5)
        self.assertEqual(PHILabel.city.value.column_index, 6)
        self.assertEqual(PHILabel.state.value.column_index, 7)
        self.assertEqual(PHILabel.country.value.column_index, 8)
        self.assertEqual(PHILabel.location_other.value.column_index, 9)
        self.assertEqual(PHILabel.phone.value.column_index, 10)
        self.assertEqual(PHILabel.url.value.column_index, 11)
        self.assertEqual(PHILabel.fax.value.column_index, 12)
        self.assertEqual(PHILabel.email.value.column_index, 13)
        self.assertEqual(PHILabel.idnum.value.column_index, 14)
        self.assertEqual(PHILabel.bioid.value.column_index, 15)
        self.assertEqual(PHILabel.organization.value.column_index, 16)
        self.assertEqual(PHILabel.profession.value.column_index, 17)
        self.assertEqual(PHILabel.patient.value.column_index, 18)
        self.assertEqual(PHILabel.doctor.value.column_index, 19)
        self.assertEqual(PHILabel.medicalrecord.value.column_index, 20)
        self.assertEqual(PHILabel.username.value.column_index, 21)
        self.assertEqual(PHILabel.device.value.column_index, 22)
        self.assertEqual(PHILabel.healthplan.value.column_index, 23)

    def test_allergy_label(self):
        self.assertTrue(isinstance(AllergyLabel.to_dict(), dict))
        self.assertTrue(self.db_range_of_ints(AllergyLabel))
        self.assertEqual(AllergyLabel.na.value.column_index, 0)
        self.assertEqual(AllergyLabel.allergy.value.column_index, 1)

    def test_diagnosis_label(self):
        self.assertTrue(isinstance(ProblemLabel.to_dict(), dict))
        self.assertTrue(self.db_range_of_ints(ProblemLabel))
        self.assertEqual(ProblemLabel.na.value.column_index, 0)
        self.assertEqual(ProblemLabel.diagnosis.value.column_index, 1)
        self.assertEqual(ProblemLabel.problem.value.column_index, 2)

    def test_med_label(self):

        self.assertTrue(isinstance(MedLabel.to_dict(), dict))
        self.assertTrue(self.db_range_of_ints(MedLabel))
        self.assertEqual(MedLabel.na.value.column_index, 0)
        self.assertEqual(MedLabel.med.value.column_index, 1)

    def test_lab_label(self):
        self.assertTrue(isinstance(LabLabel.to_dict(), dict))

        self.assertTrue(self.db_range_of_ints(LabLabel))
        self.assertEqual(LabLabel.na.value.column_index, 0)
        self.assertEqual(LabLabel.lab.value.column_index, 1)
        self.assertEqual(LabLabel.lab_value.value.column_index, 2)
        self.assertEqual(LabLabel.lab_unit.value.column_index, 3)

    def test_signsymptom_label(self):
        self.assertTrue(isinstance(SignSymptomLabel.to_dict(), dict))

        self.assertTrue(self.db_range_of_ints(SignSymptomLabel))
        self.assertEqual(SignSymptomLabel.na.value.column_index, 0)
        self.assertEqual(SignSymptomLabel.signsymptom.value.column_index, 1)

    def test_disability_label(self):
        self.assertTrue(isinstance(DisabilityLabel.to_dict(), dict))

        self.assertTrue(self.db_range_of_ints(DisabilityLabel))
        self.assertEqual(DisabilityLabel.na.value.column_index, 0)
        self.assertEqual(DisabilityLabel.diagnosis.value.column_index, 1)
        self.assertEqual(DisabilityLabel.procedure.value.column_index, 2)
        self.assertEqual(DisabilityLabel.finding.value.column_index, 3)
        self.assertEqual(DisabilityLabel.physical_exam.value.column_index, 4)
        self.assertEqual(DisabilityLabel.device.value.column_index, 5)
        self.assertEqual(DisabilityLabel.signsymptom.value.column_index, 6)

    def test_cancer_label(self):
        self.assertTrue(isinstance(CancerLabel.to_dict(), dict))

        self.assertTrue(self.db_range_of_ints(CancerLabel))
        self.assertEqual(CancerLabel.na.value.column_index, 0)
        self.assertEqual(CancerLabel.topography_primary.value.column_index, 1)
        self.assertEqual(CancerLabel.topography_metastatic.value.column_index, 2)
        self.assertEqual(CancerLabel.morphology.value.column_index, 3)
        self.assertEqual(CancerLabel.behavior.value.column_index, 4)
        self.assertEqual(CancerLabel.grade.value.column_index, 5)
        self.assertEqual(CancerLabel.stage.value.column_index, 6)

    def test_genetics_label(self):
        self.assertTrue(isinstance(GeneticsLabel.to_dict(), dict))

        self.assertTrue(self.db_range_of_ints(GeneticsLabel))
        self.assertEqual(GeneticsLabel.na.value.column_index, 0)
        self.assertEqual(GeneticsLabel.gene.value.column_index, 1)
        self.assertEqual(GeneticsLabel.gene_interpretation.value.column_index, 2)

