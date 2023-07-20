import unittest

from text2phenotype.constants.features import  AllergyLabel, MedLabel

from biomed.common.biomed_ouput import SummaryOutput, MedOutput
from biomed.common.aspect_response import AspectResponse
from biomed.common.biomed_summary import FullSummaryResponse


class TestBiomed1285(unittest.TestCase):
    INPUT = FullSummaryResponse([AspectResponse(AllergyLabel.get_category_label().persistent_label,  [
        SummaryOutput(text='Ibuprofen', umlsConcept={'cui': 'C0020740', 'code': '5640'}, range=[12, 25],
                     label=AllergyLabel.allergy.value.persistent_label),
        SummaryOutput(text='Pollen', label=AllergyLabel.allergy.value.persistent_label, range=[89, 95])
    ]),
        AspectResponse(MedLabel.get_category_label().persistent_label, [
            MedOutput(text='ibuprofen', range=[34, 46],
                         label=MedLabel.med.value.persistent_label),
            MedOutput(text='hello', umlsConcept={'cui': 'C0020740'}, range=[231, 240],
                         label=MedLabel.med.value.persistent_label),
            MedOutput(text='abs', umlsConcept={'code': '5640'}, range=[341, 240],
                         label=MedLabel.med.value.persistent_label)
        ])])

    def test_remove_allergies_from_meds(self):
        test_input = self.INPUT
        test_input.remove_allergies_from_meds()
        # test all meds removed
        self.assertEqual(test_input.to_json().get(MedLabel.get_category_label().persistent_label), [])

        expected = [
                    SummaryOutput(text='Ibuprofen', umlsConcept={'cui': 'C0020740', 'code': '5640'}, range=[12, 25],
                                 label=AllergyLabel.allergy.value.persistent_label).to_dict(),
                    SummaryOutput(text='ibuprofen', range=[34, 46],
                                  label=AllergyLabel.allergy.value.persistent_label).to_dict(),
                    SummaryOutput(text='Pollen', label=AllergyLabel.allergy.value.persistent_label,
                                  range=[89, 95]).to_dict(),
                    SummaryOutput(text='hello', umlsConcept={'cui': 'C0020740'}, range=[231, 240],
                                  label=AllergyLabel.allergy.value.persistent_label).to_dict(),
                    SummaryOutput(text='abs', umlsConcept={'code': '5640'}, range=[341, 240],
                                 label=AllergyLabel.allergy.value.persistent_label).to_dict()

                    ]
        actual = test_input.to_json()['Allergy']
        self.assertListEqual(actual, expected)
