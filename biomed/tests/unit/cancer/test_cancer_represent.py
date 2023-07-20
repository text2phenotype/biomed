import unittest

from biomed.cancer.cancer_represent import Stage, TStage, NStage, MStage, ClinicalStage


class TestStage(unittest.TestCase):
    def test(self):
        inputs = {
            self.__make_stage(TStage.A, NStage.X, MStage.X, ClinicalStage.ZERO_A): ["Stage Oa (pTa,NX, MX"],
            self.__make_stage(None, None, None, ClinicalStage.IA1): ["ia1"],
            self.__make_stage(None, None, None, ClinicalStage.IA2): ["ia2"],
            self.__make_stage(None, None, None, ClinicalStage.IB1): ["ib1", "stage ib1", "stage 1b1"],
            self.__make_stage(None, None, None, ClinicalStage.IB2): ["stage IB2", "ib2", "stage ib2"],
            self.__make_stage(None, None, None, ClinicalStage.IB3): ["ib3"],
            self.__make_stage(None, None, None, ClinicalStage.IIA): ["iia", "stage iia"],
            self.__make_stage(None, None, None, ClinicalStage.IIA2): ["iia2"],
            self.__make_stage(None, None, None, ClinicalStage.IIB): ["iib", "stage iib"],
            self.__make_stage(None, None, None, ClinicalStage.IIC): ["iic"],
            self.__make_stage(None, None, None, ClinicalStage.III): ["STAGE III", "III", "iii", " stage iii"],
            self.__make_stage(None, None, None, ClinicalStage.IIIA): ["IIIa", "iiia", "stage iiia", "iii-a"],
            self.__make_stage(None, None, None, ClinicalStage.IIIB): ["IIIB", "iiib", "stage iiib"],
            self.__make_stage(None, None, None, ClinicalStage.IIIC): ["stage IIIC", "iiic"],
            self.__make_stage(None, None, None, ClinicalStage.IIIC1): ["iiic1"],
            self.__make_stage(None, None, None, ClinicalStage.IV): ["IV", "iv", "stage iv", "stage iv gej"],
            self.__make_stage(None, None, None, ClinicalStage.IVB): ["ivb", "stage ivb"],
            # self.__make_stage(None, None, None, ClinicalStage.IVC): ["ivc", "stage ivc"],
            self.__make_stage(None, None, MStage.X, None): ["mx"],
            self.__make_stage(None, None, MStage.ZERO, None): ["m0", "mo"],
            self.__make_stage(None, None, MStage.ONE, None): ["m1", "m1a", "m1b", "m1c", "m1d", "vm1c"],
            self.__make_stage(None, None, MStage.ONE, ClinicalStage.IV): ["iv m1", "iv m1b", "iv m1c", "iv m1d",
                                                                          "iv, m1b", "iv, m1c", "stage iv m1b",
                                                                          "stage iv m1c", "stage iv m1d",
                                                                          "stage iv, m1c"],
            self.__make_stage(None, NStage.X, None, None): ["nx"],
            self.__make_stage(None, NStage.X, MStage.X, None): ["nx mx"],
            self.__make_stage(None, NStage.X, MStage.ZERO, None): ["nx mo"],
            self.__make_stage(None, NStage.X, MStage.ONE, None): ["nx m1", "nx m1b"],
            self.__make_stage(None, NStage.ZERO, None, None): ["n0"],
            self.__make_stage(None, NStage.ONE, None, None): [".n1", "¢n1a", "cn1", "n1", "n1a", "n1b", "n1c", "pn1a", "pn1b", "pn1c"],
            self.__make_stage(None, NStage.ONE, MStage.X, None): ["n1 mx", "ypn1 mx"],
            self.__make_stage(None, NStage.ONE, MStage.ZERO, None): ["n1 m0", "n1 mo", "n1, m0", "n1, mo"],
            self.__make_stage(None, NStage.ONE, MStage.ONE, None): ["n1 (m1", "n1 m1", "n1, m1", "n1, m1a", "n1b m1"],
            self.__make_stage(None, NStage.ONE, MStage.ONE, ClinicalStage.IV): ["iv, n1b m1"],
            self.__make_stage(None, NStage.TWO, None, None): ["n2", "n2¢", "n2a", "n2b", "n2c", "n2p", "pn2a"],
            self.__make_stage(None, NStage.TWO, MStage.X, None): ["n2 mx"],
            self.__make_stage(None, NStage.TWO, MStage.ZERO, None): ["n2 m0", "n2 mo", "n2b m0", "n2c m0"],
            self.__make_stage(None, NStage.TWO, MStage.ZERO, ClinicalStage.IIIA): ["n2 mo stage iiia"],
            self.__make_stage(None, NStage.TWO, MStage.ONE, None): ["n2 m1", "n2, m1", "n2b m1", "n2c m1"],
            self.__make_stage(None, NStage.TWO, MStage.ONE, ClinicalStage.IV): ["n2 m1 stage iv"],
            self.__make_stage(None, NStage.THREE, None, None): ["n3", "n3a", "n3b", "n3c", "pn3b"],
            self.__make_stage(None, NStage.THREE, MStage.ZERO, None): ["n3 m0", "n3 mo"],
            self.__make_stage(None, NStage.THREE, MStage.ONE, None): ["n3 m1", "n3 m1b", "n3, m1", "n3b m1"],

            self.__make_stage(TStage.X, NStage.ONE, MStage.ONE, ClinicalStage.IV): ["iv txn1m1", "stage iv txn1m1"],

            self.__make_stage(TStage.ZERO, None, None, None): ["ct0"],
            self.__make_stage(TStage.ZERO, NStage.ZERO, MStage.ZERO, None): ["ypT0 N0 (i-) M0;"],
            self.__make_stage(TStage.ZERO, NStage.TWO, None, None): ["t0 n2a"],

            self.__make_stage(TStage.ONE, None, None, None): ["ct1¢", "ct1a", "ct1c", "pt1", "pt1a", "pt1b", "pt1c",
                                                              "t1", "t1a", "t1b", "yt1a"],
            self.__make_stage(TStage.ONE, NStage.X, None, None): ["t1 nx"],
            self.__make_stage(TStage.ONE, NStage.X, MStage.X, None): ["pt1b/nx/mx", "t1a/nx/mx", "T1, NX, MX"],
            self.__make_stage(TStage.ONE, NStage.ZERO, MStage.X, None): ["T1N0Mx", "t1bnomx"],
            self.__make_stage(TStage.ONE, NStage.ZERO, MStage.ZERO, None): ["t1ccnomo", "t1cnomo"],
            self.__make_stage(TStage.ONE, NStage.ZERO, MStage.ZERO, ClinicalStage.I): ["Stage I (pT1a pN0 M0)"],
            self.__make_stage(TStage.ONE, NStage.ZERO, MStage.ONE, None): ["t1bnom1b"],
            self.__make_stage(TStage.ONE, NStage.ONE, None, None): ["ct1c n1", "pt1 cn1", "pt1 n1", "pt1¢ n1",
                                                                    "pt1,n1", "pt1¢n1a", "pt1c n1a", "t1c n1"],
            self.__make_stage(TStage.ONE, NStage.ONE, MStage.ZERO, None): ["ct1cn1mo0", "pt1 n1 m0", "t1c n1 m0"],
            self.__make_stage(TStage.ONE, NStage.ONE, MStage.ZERO, ClinicalStage.IIA): ["iia (t1¢ n1 m0"],
            self.__make_stage(TStage.ONE, NStage.ONE, MStage.ONE, None): ["t1/n1/m1"],
            self.__make_stage(TStage.ONE, NStage.TWO, None, None): ["pt1b pn2", "t1 n2b", "ypt1 n2"],
            self.__make_stage(TStage.ONE, NStage.TWO, MStage.ZERO, None): ["pt1¢cn2amo", "t1n2mo"],
            self.__make_stage(TStage.ONE, NStage.THREE, None, None): ["pt1b n3b", "pt1b, n3c"],
            self.__make_stage(TStage.ONE, NStage.THREE, MStage.X, None): ["t1/n3/mx"],
            self.__make_stage(TStage.ONE, NStage.THREE, None, ClinicalStage.IV): ["iv (pt1b, n3c"],

            self.__make_stage(TStage.TWO, None, None, None): ["ct2", "-CT2", "pt2", "pt2a", "pt2b", "pt2c", "rct2",
                                                              "t2"],
            self.__make_stage(TStage.TWO, None, None, ClinicalStage.IIIA): ["iii-a (pt2"],
            self.__make_stage(TStage.TWO, None, None, ClinicalStage.IIB): ["iib/pt2"],
            self.__make_stage(TStage.TWO, NStage.X, None, None): ["T2aNx", "pt2 nx", "t2a nx", "pT2, PNX"],
            self.__make_stage(TStage.TWO, NStage.X, MStage.ZERO, None): ["t2nxmo"],
            self.__make_stage(TStage.TWO, NStage.X, MStage.X, None): ["T2, Nx, Mx"],
            self.__make_stage(TStage.TWO, NStage.ZERO, None, None): ["pt2 pn0", "t2ano"],
            self.__make_stage(TStage.TWO, NStage.ZERO, MStage.X, None): ["t2cnomx", "t2nomx"],
            self.__make_stage(TStage.TWO, NStage.ZERO, MStage.ZERO, None): ["stage cT2 N0 M0", "t2anomo"],
            self.__make_stage(TStage.TWO, NStage.ZERO, MStage.ONE, ClinicalStage.IV): ["Stage IV (cT2, cN0, pM1)"],
            self.__make_stage(TStage.TWO, NStage.ONE, None, None): ["ct2 n1", "ct2.n1", "pt2 n1", "pt2 pn1", "t2 n1",
                                                                    "t2bn1", "t2n1"],
            self.__make_stage(TStage.TWO, NStage.ONE, MStage.X, None): ["t2 n1 mx", "ypt2 ypn1 mx"],
            self.__make_stage(TStage.TWO, NStage.ONE, MStage.ZERO, None): ["ct2 n1 m0", "ct2 n1 mo", "pt2 n1 m0", "pt2. n1 m0"],
            self.__make_stage(TStage.TWO, NStage.ONE, MStage.ONE, None): ["ct2 n1 m1", "ct2¢n1¢m1", "t2 n1 m1"],
            self.__make_stage(TStage.TWO, NStage.TWO, None, None): ["ct2 n2", "ct2 n2a", "ct2a n2", "pt2 n2e",
                                                                    "pt2 pn2a", "t2 n2", "t2 n2b", "t2 n2c", "ypt2 n2",
                                                                    "ypt2 n2a"],
            self.__make_stage(TStage.TWO, NStage.TWO, None, ClinicalStage.IIIA): ["iiia t2 n2"],
            self.__make_stage(TStage.TWO, NStage.TWO, MStage.X, None): ["T2 N2 MX", "pt2 pn2a mx", "t2/n2/mx",
                                                                        "ypt2 pn2 mx"],
            self.__make_stage(TStage.TWO, NStage.TWO, MStage.ZERO, None): ["pt2an2mo", "t2 n2 m0"],
            self.__make_stage(TStage.TWO, NStage.THREE, None, None): ["ct2 cn3", "ct2 n3a", "ct2 n3c", "t2 n3",
                                                                      "t2 n3b", "t2n3"],

            self.__make_stage(TStage.THREE, None, None, None): ["ct3", "pt3", "pt3a", "pt3b", "t3a", "ut3", "ypt3",
                                                                "pt3a"],
            self.__make_stage(TStage.THREE, None, MStage.ONE, ClinicalStage.IV): ["iv (pt3a m1"],
            self.__make_stage(TStage.THREE, NStage.X, MStage.X, None): ["pt3c/nx/mx"],
            self.__make_stage(TStage.THREE, NStage.ZERO, None, None): ["t3no", "ut3n0"],
            self.__make_stage(TStage.THREE, NStage.ZERO, MStage.X, None): ["T3bNoMX"],
            self.__make_stage(TStage.THREE, NStage.ZERO, MStage.ZERO, None): ["t3nomo"],
            self.__make_stage(TStage.THREE, NStage.ZERO, MStage.ONE, ClinicalStage.IV): ["iv t3nom1b"],
            self.__make_stage(TStage.THREE, NStage.ONE, None, None): ["pT3pN1b", "ct3 cn1", "ct3 n1", "ot3n1b",
                                                                      "pt3 n1", "pt3b n1s", "t3 n1", "t3n1", "t3a n1a",
                                                                      "ut3 n1"],
            self.__make_stage(TStage.THREE, NStage.ONE, None, ClinicalStage.IIIA): ["T3N1 Stage IIIa"],
            self.__make_stage(TStage.THREE, NStage.ONE, MStage.X, None): ["pt3/n1/mx", "t3 n1 mx", "t3/n1/mx",
                                                                          "t3n1mx"],
            self.__make_stage(TStage.THREE, NStage.ONE, MStage.ZERO, None): ["ct3 n1 m0", "t3 n1 m0"],
            self.__make_stage(TStage.THREE, NStage.ONE, MStage.ONE, None): ["pt3b, n1, m1b", "ypt3b n1 m1", "t3nlamla"],
            self.__make_stage(TStage.THREE, NStage.TWO, None, None): ["¢t3 n2", "ct3 n2", "pt3 n2", "pt3, pn2c",
                                                                      "pt3b pn2a", "t3 n2", "ut3 n2", "yt3, n2a"],
            self.__make_stage(TStage.THREE, NStage.TWO, MStage.X, None): ["t3 n2 mx", "t3n2mx"],
            self.__make_stage(TStage.THREE, NStage.TWO, MStage.ZERO, None): ["mrT3 N2 M0", "ct3 cn2 mo", "t3n2m0"],
            self.__make_stage(TStage.THREE, NStage.TWO, MStage.ONE, None): ["ct3 cn2 cm1a", "pt3 n2p m1", "t3n2bmla"],
            self.__make_stage(TStage.THREE, NStage.TWO, MStage.ONE, ClinicalStage.IV): ["iv t3n2m1"],
            self.__make_stage(TStage.THREE, NStage.THREE, None, None): ["t3a n3", "ypt3 n3"],
            self.__make_stage(TStage.THREE, NStage.THREE, MStage.ONE, None): ["ct3 n3 m1b"],
            self.__make_stage(TStage.THREE, NStage.THREE, MStage.ONE, ClinicalStage.IV): ["T3N3M1b Stage IV"],

            self.__make_stage(TStage.FOUR, None, None, None): ["ct4", "ct4a", "ct4b", "ct4d", "pt4", "pt4b", "pt4a",
                                                               "t4b"],
            self.__make_stage(TStage.FOUR, NStage.X, None, None): ["stage t4b nx", "t4b nx"],
            self.__make_stage(TStage.FOUR, NStage.ONE, None, None): ["pt4a pn1b", "t4b n1", "t4b n1b"],
            self.__make_stage(TStage.FOUR, NStage.ONE, MStage.ZERO, ClinicalStage.IIIA): ["iiia t4an1m0",
                                                                                          "iiia t4an1mo0"],
            self.__make_stage(TStage.FOUR, NStage.ONE, MStage.ONE, None): ["ct4b n1 m1", "t4bn1bm1"],
            self.__make_stage(TStage.FOUR, NStage.TWO, None, None): ["ct4a n2", "ct4a n2c", "pn2a ct4b", "pt4, n2",
                                                                     "pt4b pn2b", "t4a n2c", "t4b n2b"],
            self.__make_stage(TStage.FOUR, NStage.TWO, MStage.ZERO, None): ["ct4a n2c m0", "t4an2bm0"],
            self.__make_stage(TStage.FOUR, NStage.THREE, None, None): ["ct4 cn3", "pt4 n3", "pt4b, pn3b"],
            self.__make_stage(TStage.FOUR, NStage.THREE, None, ClinicalStage.IIIC): ["iiic (t4b n3"],
            self.__make_stage(TStage.FOUR, NStage.THREE, MStage.X, None): ["pT4a pN3 pMx"],
            self.__make_stage(TStage.FOUR, NStage.THREE, MStage.ZERO, ClinicalStage.IV): ["T4N3M0Stage IV"],

            self.__make_stage(TStage.A, None, None, None): ["stage TA"],
            self.__make_stage(TStage.A, NStage.X, None, None): ["pTa PNX"],
            self.__make_stage(TStage.A, NStage.X, MStage.X, None): ["Ta, NX, MX", "pTa Nx Mx", "pTa, PNX, PMX"]
        }

        failures = []
        for stage, raw_strings in inputs.items():
            for raw_str in raw_strings:
                try:
                    self.__test(raw_str, stage)
                except Exception as e:
                    failures.append(f'{raw_str}: {e}')

        if failures:
            raise Exception('\n'.join(failures))

    @staticmethod
    def __make_stage(t: TStage, n: NStage, m: MStage, clinical: ClinicalStage):
        s = Stage()
        s.T = t
        s.N = n
        s.M = m
        s.clinical = clinical
        return s

    def __test(self, raw_str: str, exp_stage: Stage):
        obs_stage = Stage.from_string(raw_str)

        self.assertEqual(exp_stage.T, obs_stage.T)
        self.assertEqual(exp_stage.N, obs_stage.N)
        self.assertEqual(exp_stage.M, obs_stage.M)
        self.assertEqual(exp_stage.clinical, obs_stage.clinical)


if __name__ == '__main__':
    unittest.main()
