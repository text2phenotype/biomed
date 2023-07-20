import unittest

from text2phenotype.entity.attributes import TextSpan


class TestBiomed503(unittest.TestCase):

    def assertGuardSpan(self, text_span: TextSpan) -> None:
        """
        :param text_span: TextSpan object to test
        :return: None
        """
        caught = False
        try:
            text_span.guard_span()
        except Exception:
            caught = True

        self.assertTrue(caught)

    def test_text_span_guard(self):

        self.assertGuardSpan(TextSpan(['text', None, 0]))  # start is None
        self.assertGuardSpan(TextSpan(['text', 0, None]))  # stop is None
        self.assertGuardSpan(TextSpan(['text', 0, 0]))  # start == stop
        self.assertGuardSpan(TextSpan(['text', -1, 0]))  # start negative
        self.assertGuardSpan(TextSpan(['text', 0, -1]))  # stop negative

        self.assertGuardSpan(TextSpan([None, 0]))  # start is None
        self.assertGuardSpan(TextSpan([0, None]))  # stop is None
        self.assertGuardSpan(TextSpan([0, 0]))  # start == stop
        self.assertGuardSpan(TextSpan([-1, 0]))  # start negative
        self.assertGuardSpan(TextSpan([0, -1]))  # stop  negative

        self.assertGuardSpan(TextSpan(['a', 0]))  # start is invalid
        self.assertGuardSpan(TextSpan([0, 'b']))  # stop is invalid
        self.assertGuardSpan(TextSpan([10, 10]))  # start == stop

    def test_text_span_equal(self):

        a = TextSpan([0, 2])
        b = TextSpan([0, 2])

        self.assertEqual(a, b)

        a = TextSpan([1, 2])
        b = TextSpan([1, 2])

        self.assertEqual(a, b)

        a = TextSpan([10, 20])
        b = TextSpan([10, 20])

        self.assertEqual(a, b)

        a = TextSpan([10, 200])
        b = TextSpan([10, 200])

        self.assertEqual(a, b)

    def test_text_span_sort(self):

        a = TextSpan([0, 1])
        b = TextSpan([0, 2])
        c = TextSpan([0, 3])
        d = TextSpan([1, 2])
        e = TextSpan([1, 3])

        # order expected
        expected = [a, b, c, d, e]

        actual = sorted(expected)

        self.assertEqual(actual[0].start, 0)
        self.assertEqual(actual[0].stop, 1)

        self.assertEqual(actual[1].start, 0)
        self.assertEqual(actual[1].stop, 2)

        self.assertEqual(actual[2].start, 0)
        self.assertEqual(actual[2].stop, 3)

        self.assertEqual(actual[3].start, 1)
        self.assertEqual(actual[3].stop, 2)

        self.assertEqual(actual[4].start, 1)
        self.assertEqual(actual[4].stop, 3)

        self.assertEqual(expected, actual)

        # out of order
        self.assertEqual(expected, sorted([e, d, c, b, a]))
        self.assertEqual(expected, sorted([d, e, a, b, c]))
        self.assertEqual(expected, sorted([c, d, b, e, a]))
        self.assertEqual(expected, sorted([b, a, c, d, e]))
        self.assertEqual(expected, sorted([a, b, c, d, e]))

        f1 = {'feature-1': 'value1', 'polarity': 'positive', 'vocab': 'ICD9CM'}
        f2 = {'feature-2': 'value2', 'polarity': 'negative', 'vocab': 'RXNORM'}
        f3 = {'feature-3': 'value3'}
        f4 = {'feature-4': None}
        f5 = {'feature-5': ['match_type_first', 'match_type_second']}

        # dictionary form
        expected = [{'span': a, 'match': f1},
                    {'span': b, 'match': f2},
                    {'span': c, 'match': f3},
                    {'span': d, 'match': f4},
                    {'span': e, 'match': f5}]

        from operator import itemgetter

        expected = sorted(expected, key=itemgetter('span'))

        # dictionary form
        actual = [{'span': a, 'match': f1},
                  {'span': b, 'match': f2},
                  {'span': c, 'match': f3},
                  {'span': d, 'match': f4},
                  {'span': e, 'match': f5}]

        actual = sorted(actual, key=itemgetter('span'))

        self.assertEqual(expected, actual)

    def test_tuple(self):

        a = (0, 1)
        b = (0, 2)
        c = (0, 3)
        d = (1, 2)
        e = (1, 3)

        val = {'foo': 'bar'}

        # dictionary form
        expected = {a: val,
                    b: val,
                    c: val,
                    d: val,
                    e: val}

        expected = sorted(expected.items())

        # dictionary form
        actual = {d: val,
                  c: val,
                  e: val,
                  a: val,
                  b: val}

        actual = sorted(actual.items())

        self.assertEqual(expected, actual)
