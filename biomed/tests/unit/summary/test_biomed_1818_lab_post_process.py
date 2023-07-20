import unittest
from biomed.common.aspect_response import LabResponse
from biomed.common.biomed_ouput import LabOutput
from text2phenotype.constants.features import LabLabel


class TestBiomed1818LabUnderstanding(unittest.TestCase):
    TEXT = """Hemoglobin A1c 4.3 g/dl (12.9-18.4) 
    WBC 2.43 x 103/ul

    """
    LAB_OUTPUT = LabResponse(
        'Lab',
        [LabOutput(**{
            'text': 'Hemoglobin', 'range': [0, 10], 'lstm_prob': 0.98,
            'label': 'lab', 'polarity': None, 'code': 'LP32067-8', 'cui': 'C0019046',
            'tui': 'T034', 'vocab': 'LNC', 'preferredText': 'HEMOGLOBIN',
            'labValue': ['12.9', 25, 29], 'labUnit': [], 'labInterp': None})
         ])

    def test_get_summary_labs(self):
        self.LAB_OUTPUT.response_list.append(LabOutput(text='g/dl', label=LabLabel.lab_unit.value.persistent_label,
                                                       range=[19, 23], lstm_prob=0.98
                                                       ))
        self.LAB_OUTPUT.post_process(self.TEXT)
        self.assertEqual(self.LAB_OUTPUT.response_list[0].labUnit.to_output_list(), ['g/dl', 19, 23])
