import unittest

from biomed.demographic.demographic import get_demographic_tokens
from text2phenotype.constants.features.label_types import PHILabel, DemographicEncounterLabel
from text2phenotype.tasks.task_enums import TaskOperation

from biomed.common.helpers import annotation_helper
from biomed.deid.deid import get_phi_tokens
from biomed.deid.utils import demographics_chunk_to_deid
from biomed.reassembler.reassemble_functions import reassemble_summary_chunk_results


class TestBiomed170(unittest.TestCase):

    def test_MRN_from_CCDA(self):

        text = """
        Continuity of Care Document 06/16/2017 -- Campos,Ricardo
        Patient	Ricardo Campos
        Date of birth	January 2, 1963
        Sex	Male
        Race	Other Race
        Ethnicity	Hispanic or Latino
        Contact info	Primary Home:
        123 MONTGOMERY BLVD NE
        Las Cruces, NM 88007, US
        Tel: (575) 123-4567
        Patient IDs	123456789 2.16.840.1.113883.1.13.99999.1
        Document Id	280004 2.16.840.1.113883.1.13.99999.999362
        Document Created:	July 16, 2017, 11:44:46, EST
        Performer (primary care physician)	Miranda Bailey, MD
        Author	Millennium Clinical Document Generator
        Contact info
        """
        mrn, person, street, phone, hipaa, date = [], [], [], [], [], []
        tokens,  vectors = annotation_helper(text, {TaskOperation.phi_tokens, TaskOperation.demographics})

        phi_tokens = get_phi_tokens(tokens, vectors=vectors)
        dem_tokens = get_demographic_tokens(
            tokens,
            vectors=vectors)
        phi_resp = reassemble_summary_chunk_results([
            ([0, len(text)],  phi_tokens),
            (
                [0, len(text)],
                demographics_chunk_to_deid(
                    dem_tokens[DemographicEncounterLabel.get_category_label().persistent_label]).to_json())
        ])
        phi_tokens = phi_resp[PHILabel.get_category_label().persistent_label]

        for phi in phi_tokens:
            hipaa.append(phi['text'])

            if phi['label'] in [PHILabel.medicalrecord.name]:
                mrn.append(phi['text'])

            elif phi['label'] in [PHILabel.doctor.name, PHILabel.patient.name]:
                person.append(phi['text'])

            elif phi['label'] in [PHILabel.street.name]:
                street.append(phi['text'])

            elif phi['label'] in [PHILabel.phone.name]:
                phone.append(phi['text'])

            elif phi['label'] in [PHILabel.date.name]:
                date.append(phi['text'])

        self.assertIn('06/16/2017', hipaa)
        self.assertIn('Ricardo', person)
        self.assertIn('Campos', person)
        self.assertIn('Miranda', person)
        self.assertIn('Bailey', person)
        self.assertIn('123456789', hipaa)
        self.assertIn('123', street)
        self.assertIn('575', phone)
        self.assertIn('123-4567', phone)
        self.assertIn('January', hipaa)
        self.assertIn('2', hipaa)
        self.assertIn('July', hipaa)
        self.assertIn('16', hipaa)
        self.assertIn('123456789', hipaa)
        self.assertIn('2.16.840.1.113883.1.13.99999.1', hipaa)
