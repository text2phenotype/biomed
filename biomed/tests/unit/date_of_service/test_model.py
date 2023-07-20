import unittest

from biomed.date_of_service.model import DOSModel


class DOSModelTest(unittest.TestCase):
    __model = DOSModel()

    def test_date_identified(self):
        text = """Cardiology From 1/1/2004 through 12/29/2014
08/15/2014 09:03
12 Lead ECG
Final
Report Status: Final
Report Number: 2722171
Type : 12 Lead ECG
Date: 08/15/2014 09:03
Ordering Provider: CHIOCCA, ENNIO
Reviewed by: LEWIS, M.D., ELDRIN F.
Ventricular Rate 60 BPM
Atrial Rate 60 BPM
P-R Interval 148 ms
QRS Duration 100 ms
QT 390 ms
QTC 390 ms
P Axis 48 degrees
R Axis 84 degrees
T Axis 49 degrees
Normal sinus rhythm
Normal ECG
No previous ECGS available
Confirmed by LEWIS, M.D., ELDRIN F. (225) on 8/19/2014 12:11:08 PM
Printed: 12/29/2014 12:37 PM
Page 2 of 70
Partners HealthCare System, Inc.
BRIGHAM & WOMEN'S HOSPITAL
A Teaching Affiliate of Harvard Medical School
75 Francis Street, Boston, Massachusetts 02115
MRN:
(BWH)
KEATING,STEVEN
Date of Birth: 04/29/1988
Age: 26 yrs. Sex: M
Discharge Reports From 1/1/2004 through 12/29/2014
08/19/2014 06:22
Discharge Summary
Final
Ennio A. Chiocca, M.D. Ph.D.
BWH BRIGHAM AND WOMEN'S HOSPITAL
A Teaching Affiliate of Harvard Medical School
75 Francis Street, Boston, Massachusetts 02115
Admission: 8/19/2014
Discharge: 8/21/2014
Discharge Summary
FINAL
"""

        predictions = self.__model.predict(text)
        self.assertGreater(len(predictions), 0)

    def test_regression1(self):
        problem_text = """Johnson-Smith, Michael Stephen
Expired Tidal... (mL) [528] [581] [509] [541] [534] [513]

Resp Rate C02 [14] [12] [12] [12] [12] [12]
1545 1660 1615 1630 1645 1760 1715

"""
        self.__test_predicted_indices(problem_text)

    def __test_predicted_indices(self, text):
        for prediction in self.__model.predict(text):
            self.assertEqual(prediction.text, text[prediction.span[0]:prediction.span[1]])


if __name__ == '__main__':
    unittest.main()
