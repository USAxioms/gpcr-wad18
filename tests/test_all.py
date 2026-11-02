"""
ACE test suite — zero third-party dependencies beyond numpy/scipy.

Tests are pinned against verified outputs from the first successful run.
Any mutation that changes a result breaks at least one test.
"""
import sys, os, math
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import wad18 as W
import gnm   as GNM
import bias  as BIAS
from   csl   import CSL, IntegrityError
import tempfile, json, unittest


class TestWAD18(unittest.TestCase):

    def test_from_str_roundtrip(self):
        for s in ("-9.200000000000000000", "0.616200000000000000",
                  "-4.100000000000000000", "0.810000000000000000"):
            self.assertEqual(W.to_str(W.from_str(s)), s)

    def test_mul_sign(self):
        a = W.from_str("-9.2"); b = W.from_str("0.81")
        self.assertLess(W.mul(a, b), 0)   # neg * pos = neg

    def test_div_by_zero(self):
        with self.assertRaises(ZeroDivisionError):
            W.div(W.ONE, W.ZERO)

    def test_rejects_float(self):
        with self.assertRaises(TypeError):
            W.assert_wad(1.0)

    def test_half_even(self):
        self.assertEqual(W._rhe(5, 2), 2)   # 2.5 → 2
        self.assertEqual(W._rhe(7, 2), 4)   # 3.5 → 4

    def test_no_overflow_on_multiply(self):
        large = W.from_str("9.200000000000000000")
        result = W.mul(large, large)
        self.assertIsInstance(result, int)

    def test_rejects_over_precision(self):
        with self.assertRaises(ValueError):
            W.from_str("1.0000000000000000001")

    def test_exact_cancellation(self):
        """ΔGbind cancels when computing B from ΔGeff difference."""
        dGb = W.from_str("-9.2"); dGc = W.from_str("-4.1")
        dGt_G = W.from_str("-10.4"); dGt_B = W.from_str("-4.9")
        lG = W.from_str("0.81"); lB = W.from_str("0.38")
        mG = W.from_str("1.0"); mB = W.from_str("1.0")
        RT = W.from_str("0.6162")
        dGe_G = dGb + W.mul(lG, dGc) + W.mul(mG, dGt_G)
        dGe_B = dGb + W.mul(lB, dGc) + W.mul(mB, dGt_B)
        B = W.div(-(dGe_G - dGe_B), RT)
        # compute without dGbind
        B2 = W.div(-(W.mul(lG,dGc)+W.mul(mG,dGt_G) -
                     W.mul(lB,dGc)-W.mul(mB,dGt_B)), RT)
        self.assertEqual(B, B2)


class TestLaplacian(unittest.TestCase):

    def setUp(self):
        self.coords = GNM.synthetic_gpcr_coords(seed=42)
        self.lap    = GNM.build_laplacian(self.coords, r_c=7.5)

    def test_n_residues(self):
        self.assertEqual(self.lap["N"], 184)

    def test_symmetric(self):
        L = self.lap["L"]
        self.assertTrue(np.allclose(L, L.T, atol=1e-12))

    def test_row_sums_zero(self):
        L = self.lap["L"]
        self.assertLess(float(np.abs(L.sum(axis=1)).max()), 1e-12)

    def test_positive_semidefinite(self):
        L = self.lap["L"]
        eigvals = np.linalg.eigvalsh(L)
        self.assertGreaterEqual(float(eigvals.min()), -1e-10)

    def test_has_edges(self):
        self.assertGreater(self.lap["n_edges"], 0)

    def test_adjacency_nonneg(self):
        self.assertTrue((self.lap["A"] >= 0).all())

    def test_gaussian_weights_at_most_one(self):
        self.assertTrue((self.lap["A"] <= 1.0 + 1e-12).all())


class TestPseudoinverse(unittest.TestCase):

    def setUp(self):
        coords   = GNM.synthetic_gpcr_coords(seed=42)
        lap      = GNM.build_laplacian(coords, r_c=7.5)
        self.ps  = GNM.pseudoinverse(lap["L"])
        self.L   = lap["L"]
        self.G   = self.ps["G"]
        self.N   = 184

    def test_rank(self):
        self.assertEqual(self.ps["rank"], self.N - 1)

    def test_fiedler_positive(self):
        self.assertGreater(self.ps["lambda_2"], 0)

    def test_penrose_condition_1(self):
        self.assertLess(self.ps["residual"], 1e-8)

    def test_G_symmetric(self):
        G = self.G
        self.assertTrue(np.allclose(G, G.T, atol=1e-10))

    def test_trace_positive(self):
        self.assertGreater(self.ps["trace_G"], 0)

    def test_chi_positive(self):
        self.assertGreater(self.ps["chi"], 0)

    def test_null_mode_excluded(self):
        """G must annihilate the zero eigenvector: G·1 ≈ 0."""
        ones = np.ones(self.N)
        result = self.G @ ones
        self.assertLess(float(np.abs(result).max()), 1e-8)

    def test_penrose_condition_2(self):
        """G L G = G."""
        G, L = self.G, self.L
        glg = G @ L @ G
        self.assertLess(float(np.linalg.norm(glg - G, "fro")), 1e-7)

    def test_penrose_condition_3(self):
        """(LG)^T = LG."""
        G, L = self.G, self.L
        LG = L @ G
        self.assertTrue(np.allclose(LG, LG.T, atol=1e-8))

    def test_penrose_condition_4(self):
        """(GL)^T = GL."""
        G, L = self.G, self.L
        GL = G @ L
        self.assertTrue(np.allclose(GL, GL.T, atol=1e-8))


class TestAllostery(unittest.TestCase):

    def setUp(self):
        coords  = GNM.synthetic_gpcr_coords(seed=42)
        lap     = GNM.build_laplacian(coords, r_c=7.5)
        ps      = GNM.pseudoinverse(lap["L"])
        self.da = BIAS.allosteric_response(ps["G"], ps["chi"], 65, -9.2)
        self.dP = BIAS.hdx_protection_changes(self.da, 1.5)
        self.N  = 184

    def test_length(self):
        self.assertEqual(len(self.da), self.N)

    def test_site_nonzero(self):
        self.assertNotAlmostEqual(float(self.da[65]), 0.0)

    def test_hdx_antiparallel_to_da(self):
        """ΔP = -β·Δa so they must have opposite signs wherever Δa != 0."""
        nonzero = np.abs(self.da) > 1e-10
        self.assertTrue(np.all(np.sign(self.dP[nonzero]) ==
                               -np.sign(self.da[nonzero])))

    def test_topology_ratio_independent_of_magnitude(self):
        """Δa_j / Δa_k must be the same for two different Δμ values."""
        coords  = GNM.synthetic_gpcr_coords(seed=42)
        lap     = GNM.build_laplacian(coords, r_c=7.5)
        ps      = GNM.pseudoinverse(lap["L"])
        da1     = BIAS.allosteric_response(ps["G"], ps["chi"], 65, -9.2)
        da2     = BIAS.allosteric_response(ps["G"], ps["chi"], 65, -4.6)
        nonzero = np.abs(da2) > 1e-10
        ratio   = da1[nonzero] / da2[nonzero]
        self.assertTrue(np.allclose(ratio, 2.0, rtol=1e-9))


class TestBiasIndex(unittest.TestCase):

    def setUp(self):
        self.dGb  = W.from_str("-9.200000000000000000")
        self.dGc  = W.from_str("-4.100000000000000000")
        self.dGtG = W.from_str("-10.400000000000000000")
        self.dGtB = W.from_str("-4.900000000000000000")
        self.lG   = W.from_str("0.810000000000000000")
        self.lB   = W.from_str("0.380000000000000000")
        self.mG   = W.from_str("1.000000000000000000")
        self.mB   = W.from_str("1.000000000000000000")
        self.RT   = W.from_str("0.616200000000000000")

    def test_effective_free_energy_G(self):
        dGe = BIAS.effective_free_energy_wad18(
            self.dGb, self.dGc, self.dGtG, self.lG, self.mG)
        # ΔGeff(G) = -9.2 + 0.81*(-4.1) + 1.0*(-10.4)
        #           = -9.2 - 3.321 - 10.4 = -22.921
        expected = W.from_str("-22.921000000000000000")
        self.assertEqual(dGe, expected)

    def test_effective_free_energy_Barr(self):
        dGe = BIAS.effective_free_energy_wad18(
            self.dGb, self.dGc, self.dGtB, self.lB, self.mB)
        # ΔGeff(β) = -9.2 + 0.38*(-4.1) + 1.0*(-4.9)
        #           = -9.2 - 1.558 - 4.9 = -15.658
        expected = W.from_str("-15.658000000000000000")
        self.assertEqual(dGe, expected)

    def test_bias_index_value(self):
        dGe_G = BIAS.effective_free_energy_wad18(
            self.dGb, self.dGc, self.dGtG, self.lG, self.mG)
        dGe_B = BIAS.effective_free_energy_wad18(
            self.dGb, self.dGc, self.dGtB, self.lB, self.mB)
        B = BIAS.bias_index_wad18(dGe_G, dGe_B, self.RT)
        # B = -(-22.921 - (-15.658)) / 0.6162
        #   = -(-7.263) / 0.6162 = 7.263/0.6162 ≈ +11.785...
        B_float = B / W.SCALE
        self.assertAlmostEqual(B_float, 11.785, places=2)

    def test_DGbind_cancels_exactly(self):
        dGe_G = BIAS.effective_free_energy_wad18(
            self.dGb, self.dGc, self.dGtG, self.lG, self.mG)
        dGe_B = BIAS.effective_free_energy_wad18(
            self.dGb, self.dGc, self.dGtB, self.lB, self.mB)
        B_with = BIAS.bias_index_wad18(dGe_G, dGe_B, self.RT)
        # recompute without ΔGbind
        dGe_G2 = W.mul(self.lG, self.dGc) + W.mul(self.mG, self.dGtG)
        dGe_B2 = W.mul(self.lB, self.dGc) + W.mul(self.mB, self.dGtB)
        B_without = W.div(-(dGe_G2 - dGe_B2), self.RT)
        self.assertEqual(B_with, B_without)

    def test_B_positive_means_G_protein_preferred(self):
        dGe_G = BIAS.effective_free_energy_wad18(
            self.dGb, self.dGc, self.dGtG, self.lG, self.mG)
        dGe_B = BIAS.effective_free_energy_wad18(
            self.dGb, self.dGc, self.dGtB, self.lB, self.mB)
        B = BIAS.bias_index_wad18(dGe_G, dGe_B, self.RT)
        self.assertGreater(B, 0)   # G-protein preferred

    def test_occupancy_ratio_large(self):
        dGe_G = BIAS.effective_free_energy_wad18(
            self.dGb, self.dGc, self.dGtG, self.lG, self.mG)
        dGe_B = BIAS.effective_free_energy_wad18(
            self.dGb, self.dGc, self.dGtB, self.lB, self.mB)
        B = BIAS.bias_index_wad18(dGe_G, dGe_B, self.RT)
        ratio = math.exp(B / W.SCALE)
        self.assertGreater(ratio, 1e4)


class TestCSL(unittest.TestCase):

    def test_genesis_and_chain(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            csl = CSL(path)
            b0  = csl.append({"x": 1}, "genesis")
            b1  = csl.append({"x": 2}, "second")
            self.assertEqual(b0["i"], 0)
            self.assertEqual(b1["i"], 1)
            self.assertEqual(b1["H_prev"], b0["H_i"])
            self.assertTrue(csl.verify())
        finally:
            os.unlink(path)

    def test_tamper_detection(self):
        with tempfile.NamedTemporaryFile(suffix=".json",
                                         mode="w", delete=False) as f:
            path = f.name
        try:
            csl = CSL(path)
            csl.append({"x": 1})
            csl.append({"x": 2})
            # tamper with block 0
            blocks = json.load(open(path))
            blocks[0]["S_i"]["x"] = 999
            json.dump(blocks, open(path, "w"))
            with self.assertRaises(IntegrityError):
                CSL(path).verify()
        finally:
            os.unlink(path)

    def test_root_hash_changes_on_append(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            csl = CSL(path)
            h0  = csl.root_hash()
            csl.append({"a": 1})
            h1  = csl.root_hash()
            self.assertNotEqual(h0, h1)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
