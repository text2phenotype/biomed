import datetime
from unittest import TestCase
from biomed.common.biomed_ouput import MedOutput
from biomed.common.aspect_response import AspectResponse


class TestBiomed1405(TestCase):
    def test_merge_nearby_drug(self):
        input = AspectResponse('Medication', [MedOutput(text='hydralazine', label='medication', range=[0, 11],
                                          attributes={'polarity': 'positive',
                                                      'medFrequencyNumber': ['2.0', 7, 9],
                                                      'medFrequencyUnit': ['day', 8, 12],
                                                      'medStrengthNum': ['10', 11, 13],
                                                      'medStrengthUnit': ['mg', 14, 16],
                                                      }, lstm_prob=0.973,
                                          umlsConcept={'code': '5470',
                                                       'cui': 'C0020223',
                                                       'tui': 'T109',
                                                       'codingScheme': 'RXNORM',
                                                       'preferredText': 'Hydralazine',
                                                       },
                                                    date=datetime.date(year=2020, day=19, month=11)
),
                                MedOutput(text='10', label='medication', range=[12, 14],
                                          date=datetime.date(year=2020, day=18, month=11),
                                          attributes={'polarity': 'positive', 'medDosage': '10'}, lstm_prob=0.976)])

        input.merge_nearby_terms(text='hydralazine 10')
        expected = dict(text='hydralazine 10',
                        label='medication',
                        range=[0, 14],
                        score=0.976,
                        polarity='positive',
                        medFrequencyNumber=[2.0, 7, 9],
                        medFrequencyUnit=['day', 8, 12],
                        medStrengthNum=[10.0, 11, 13],
                        medStrengthUnit=['mg', 14, 16],
                        code='5470',
                        cui='C0020223',
                        tui='T109',
                        vocab='RXNORM',
                        date='2020-11-19',
                        preferredText='Hydralazine',
                        page=None)
        self.assertDictEqual(expected, input.response_list[0].to_dict())
