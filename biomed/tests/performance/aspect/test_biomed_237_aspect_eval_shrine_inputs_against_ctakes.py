import os
import unittest

from feature_service.nlp.nlp_reader import ClinicalReader, ResultType
from text2phenotype.common import common
from text2phenotype.common.log import operations_logger

from feature_service.nlp import autocode

from biomed.tests.samples import SHRINE_LOINC_BSV
from biomed.tests.samples import SHRINE_RXNORM_BSV, SHRINE_NDFRT_BSV
from biomed.tests.samples import SHRINE_ICD9_BSV, SHRINE_ICD10_BSV, SHRINE_RACE_BSV
from biomed.tests.samples import VACCINE_CSX_BSV


class TestBiomed237(unittest.TestCase):

    def setUp(self):
        self.do_output = os.environ.get('TEST_OUTPUT', 0)

    def list_samples(self, file_bsv):
        return [line.split('|') for line in common.read_text(file_bsv).splitlines()]

    def progress(self, success: int, failed: int):
        total = success + failed
        prct = success / total
        operations_logger.info(f"{prct}% | {total} | {success} | {failed}")

    def list2bsv(self, from_list) -> str:
        return '\n'.join(from_list)

    def test_biomed_237_shrine_curated_samples(self):
        """
        McMurry et al:
        http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0055811

        Matvey et al
        https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3041416/
        """
        self.test_demographics()
        self.test_loinc()

        self.test_ndfrt()
        self.test_rxnorm()

        self.test_icd9()
        self.test_icd10()

    def test_loinc(self):
        """
        Input: 275 common lab tests
        """
        self.eval_source(SHRINE_LOINC_BSV, 'loinc')

    def test_ndfrt(self):
        """
        484 Medication Group Categories like "ANTIHISTAMINES", "CARDIOVASCULAR MEDICATIONS"
        SHRINE curated list of Medication names from 5 hospitals,
        must have at least one medication common in all 5 hospital settings.
        """
        self.eval_source(SHRINE_NDFRT_BSV, 'ndfrt')

    def test_rxnorm(self):
        """
        14,248 Medications in RXNORM or NDFRT
        Meds 'mapped' to a local terminology code using ingredients, at least once in 5 harvard hospital locations.
        """
        self.eval_source(SHRINE_RXNORM_BSV, 'rxnorm')

    def test_demographics(self):
        """
        CDC Race/Ethnicity code sets
        https://gettext2phenotype.atlassian.net/browse/BIOMED-124
        """
        self.eval_source(SHRINE_RACE_BSV, 'race')

    def test_icd9(self):
        """
        49,532 unique diagnosis texts
        17,794 unique diagnosis codes
        ICD9 Diagnoses recorded at least once in 5 harvard hospitals
        """
        self.eval_source(SHRINE_ICD9_BSV, 'icd9')

    def test_icd10(self):
        """
        92,037  unique diagnosis codes
        102,735 unique diagnosis texts
        ICD10 Diagnoses recorded at least once in 5 harvard hospitals
        """
        self.eval_source(SHRINE_ICD10_BSV, 'icd10')

    def test_cvx(self):
        """
        https://www2a.cdc.gov/vaccines/iis/iisstandards/vaccines.asp?rpt=vg
        """
        self.eval_source(VACCINE_CSX_BSV, 'cvx')

    def eval_source(self, source_file_bsv, vocab_name):
        """
        :param source_file_bsv: BSV file format with code|text
        :param vocab_name: name of vocab like "NDFRT"
        """
        return [self.eval_source_target(source_file_bsv,
                                        f"{vocab_name}_drug_ner",
                                        autocode.drug_ner,
                                        ResultType.drug_ner),
                self.eval_source_target(source_file_bsv,
                                        f"{vocab_name}_lab_value",
                                        autocode.lab_value,
                                        ResultType.lab_value),
                self.eval_source_target(source_file_bsv,
                                        f"{vocab_name}_hepc_lab_value",
                                        autocode.hepc_lab_value,
                                        ResultType.lab_value),
                self.eval_source_target(source_file_bsv,
                                        f"{vocab_name}_hepc_clinical",
                                        autocode.hepc_lab_value,
                                        ResultType.clinical),
                self.eval_source_target(source_file_bsv,
                                        f"{vocab_name}_clinical",
                                        autocode.clinical,
                                        ResultType.clinical)]

    def eval_source_target(self, source_file_bsv, batch_name, pipeline, result_type):
        """
        """
        operations_logger.info(f"{batch_name}")

        exact, partial, missed = list(), list(), list()

        rxnorm = self.list_samples(source_file_bsv)
        uniq = set()

        for concept in rxnorm:
            code, text = concept[0], concept[1]

            try:
                if text not in uniq:
                    uniq.add(text)

                    reader = ClinicalReader(text, pipeline, result_type)

                    mentioned = reader.uniq_result_text()
                    preferred = reader.uniq_concept_text()
                    cui_csv = ','.join(reader.uniq_concept_cuis())

                    if text in mentioned or text in preferred:
                        exact.append(f"{code}|{text}|{preferred}|{cui_csv}")
                    else:
                        partial.append(f"{code}|{text}|{preferred}|{cui_csv}")

                    self.progress(len(exact), len(missed) + len(partial))

            except KeyError:
                missed.append(f"{code}|{text}")

        if self.do_output:
            target_file_bsv = os.path.dirname(source_file_bsv)
            target_file_bsv = f"{target_file_bsv}/BIOMED-237.{batch_name}"

            common.write_text(self.list2bsv(exact), f"{target_file_bsv}.exact.bsv")
            common.write_text(self.list2bsv(partial), f"{target_file_bsv}.partial.bsv")
            common.write_text(self.list2bsv(missed), f"{target_file_bsv}.missed.bsv")

        return [exact, partial, missed]
