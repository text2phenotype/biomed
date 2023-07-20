import unittest

from text2phenotype.tasks.task_enums import TaskOperation

from biomed.deid.deid import get_phi_tokens
from biomed.common.helpers import annotation_helper

# https://www.ncbi.nlm.nih.gov/pubmed/13357543
#
TEXT = """
Epithelia noted for their water transport have been studied by electron microscopy with particular emphasis upon basal specializations. 
Epithelia of the submaxillary gland, choroid plexus, and ciliary body are described in this article, and compared with previous observations on the kidney. 
The basal surface of all these epithelia is tremendously expanded by folds which penetrate deeply into the cytoplasm. 
In the submaxillary gland this is particularly notable in cells of the serous alveoli and in the secretory ducts. 
In these instances the folds have a fairly regular distribution and have a marked tendency to turn back upon themselves and so form repeating S-shaped patterns. 
In the choroid plexus the penetrating basal folds are limited to the lateral regions of each ependymal cell where they blend with the intercellular membranes that are also folded. 
In the epithelium of the ciliary body it is the inner layer that is specialized. 
The surface adjacent to the cavity of the eye penetrates irregularly, nearly through the full depth of the cell layer. 
The exposed surface is, in a fundamental sense, the basal surface of this epithelial layer. 
It is apparent that the pattern of folding is quite distinctive in the different epithelia. 
Therefore, the specializations should be regarded as analogous rather than homologous. 
Topographic considerations presumably limit the manner in which basal cell surfaces might be expanded. 
Penetrating folds would seem to represent almost the only possible solution.        
"""


class TestBiomed277(unittest.TestCase):

    def assertNoPHI(self, text: str):
        tokens, vectors = annotation_helper(text, {TaskOperation.phi_tokens})
        phi_res = get_phi_tokens(tokens, vectors=vectors)
        phi = phi_res['PHI']
        phi_certain = [p for p in phi if p['score'] > .5]
        self.assertEqual(0, len(phi_certain))
        self.assertLessEqual(len(phi), 2)

    def test_specificity_pubmed_paragraph_no_section(self):
        """
        DEID Text without a section header in paragraph format.
        """
        self.assertNoPHI(TEXT)

    def test_specificity_pubmed_paragraph_with_section_headers(self):
        """
        DEID Text with common PHI and non-PHI section headers
        """
        self.assertNoPHI(f"SSN:\n{TEXT}")
        self.assertNoPHI(f"DEMOGRAPHICS:\n{TEXT}")

        self.assertNoPHI(f"PROBLEM LIST:\n{TEXT}")
        self.assertNoPHI(f"MEDICATION LIST:\n{TEXT}")

    def test_specificity_pubmed_runon_sentence_no_section(self):
        """
        DEID Text without line breaks
        """
        self.assertNoPHI(TEXT.replace('\n', ''))
