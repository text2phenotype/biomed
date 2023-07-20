import unittest
from text2phenotype.apiclients.feature_service import FeatureServiceClient

from biomed.constants.constants import ModelType
from biomed.meta.ensembler import Ensembler
from biomed.imaging_finding.imaging_finding import imaging_and_findings


class TestImagingFindingModel(unittest.TestCase):
    txt = """
Chest XRay - bilateral infiltrates, ground glass opacities, diffuse opacities
Chest CT - diffuse opacities
"""

    def test_imaging_finding(self):
        annotation, vectorization = FeatureServiceClient().annotate_vectorize(text=self.txt)
        res = imaging_and_findings(tokens=annotation, vectors=vectorization, text=self.txt)
        findings =  res['Findings']
        imaging_studies = res['DiagnosticImaging']

        # test findings
        required_findings = {'infiltrates', 'opacities'}
        acceptable_findings = {'bilateral', 'ground', 'glass', 'diffuse'}.union(required_findings)
        pred_finding_terms = set()
        for find in findings:
            for val in find['text'].split():
                pred_finding_terms.add(val.strip(' ,.'))
        self.assertEqual(len(required_findings.difference(pred_finding_terms)), 0)
        self.assertEqual(len(pred_finding_terms.difference(acceptable_findings)), 0)
        # test imaging
        required_diagnostic_imaging = {'Chest', 'XRay', 'CT'}
        pred_imaging_terms = set()
        for find in imaging_studies:
            for val in find['text'].split():
                pred_imaging_terms.add(val.strip(' ,.'))
        self.assertEqual(len(required_diagnostic_imaging.difference(pred_imaging_terms)), 0)
