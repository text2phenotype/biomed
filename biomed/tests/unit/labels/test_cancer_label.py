import unittest

from text2phenotype.constants.features.label_types import CancerLabel


class TestCancerLabel(unittest.TestCase):
    def test_non_behavior_labels(self):
        for label in [CancerLabel[label] for label in CancerLabel.__members__
                      if label != CancerLabel.behavior.value.persistent_label]:
            self.assertIs(label, CancerLabel.from_brat(label.value.persistent_label))

    def test_behavior_labels(self):
        for behavior in CancerLabel.behavior.value.alternative_text_labels:
            self.assertIs(CancerLabel.behavior, CancerLabel.from_brat(behavior))


if __name__ == '__main__':
    unittest.main()
