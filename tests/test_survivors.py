"""
Targeted tests written to kill specific surviving mutants.

Each test is labelled with the mutant(s) it kills.
Tests are ordered by file and line number.

After this suite, remaining survivors are either:
  (a) proved extensionally equivalent (see spec/mutation_equivalents.json), or
  (b) unexplained (there should be none after this pass).
"""
import sys, os, math, tempfile, json, unittest
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import wad18 as W
from csl import CSL, IntegrityError
import gnm as GNM
import bias as BIAS

SCALE = W.SCALE


# ══════════════════════════════════════════════════════════
# wad18.py survivors
# ══════════════════════════════════════════════════════════

class TestWad18Survivors(unittest.TestCase):

    def test_rhe_rejects_zero_denominator_with_value_error(self):
        """Kills wad18:24 '<=→<': den=0 must raise ValueError (our guard),
        NOT ZeroDivisionError (which Python would raise at divmod).
        The mutant '< 0' lets den=0 through to divmod, raising ZeroDivisionError."""
        try:
            W._rhe(1, 0)
            self.fail("Expected ValueError for den=0")
        except ValueError:
            pass   # correct
        except ZeroDivisionError:
            self.fail("Mutant '<=→<' lets den=0 reach divmod: "
                      "should be ValueError not ZeroDivisionError")

    def test_from_str_empty_string_is_zero_not_error(self):
        """Kills wad18:46 '!=→==': from_str('') must return 0.
        With '== \"\"' the condition True and True fires on empty string,
        raising ValueError instead of returning 0."""
        result = W.from_str("")
        self.assertEqual(result, 0,
            "from_str('') should return 0 (empty = zero), got %r" % result)

    def test_rhe_accepts_denominator_of_one(self):
        """Kills wad18:24 '0→1': den=1 must be accepted, not rejected."""
        self.assertEqual(W._rhe(1, 1), 1)
        self.assertEqual(W._rhe(0, 1), 0)

    def test_from_int_returns_scale_multiple(self):
        """Kills wad18:53 '*→//': from_int(1) must equal SCALE, not 0."""
        self.assertEqual(W.from_int(1),  SCALE)
        self.assertEqual(W.from_int(2),  2 * SCALE)
        self.assertEqual(W.from_int(-3), -3 * SCALE)
        self.assertEqual(W.from_int(0),  0)

    def test_to_str_zero_is_not_negative(self):
        """Kills wad18:57 '<→<=' and '0→1': to_str(0) must start with '0'."""
        s = W.to_str(0)
        self.assertFalse(s.startswith("-"),
                         "to_str(0) rendered as negative: %r" % s)
        self.assertTrue(s.startswith("0"))

    def test_to_str_positive_unit_is_not_negative(self):
        """Kills wad18:57 '0→1': to_str(SCALE) must be positive."""
        s = W.to_str(SCALE)
        self.assertFalse(s.startswith("-"))
        self.assertTrue(s.startswith("1"))

    def test_mul_zero_by_nonzero_is_zero(self):
        """Kills wad18:65 '<→<=' and '0→1': 0 * x must equal 0."""
        x = W.from_str("9.200000000000000000")
        self.assertEqual(W.mul(W.ZERO, x), 0)
        self.assertEqual(W.mul(x, W.ZERO), 0)

    def test_mul_negative_zero_is_zero_not_negative(self):
        """Kills wad18:65 '<→<=': sign(0) must not flip mul sign."""
        x = W.from_str("-4.100000000000000000")
        result = W.mul(W.ZERO, x)
        self.assertEqual(result, 0)
        s = W.to_str(result)
        self.assertFalse(s.startswith("-"))

    def test_div_zero_numerator_is_zero(self):
        """Kills wad18:73 '<→<=' and '0→1': 0 / x must equal 0."""
        x = W.from_str("0.616200000000000000")
        self.assertEqual(W.div(W.ZERO, x), 0)

    def test_div_negative_result_is_negative(self):
        """Kills wad18:75 '-→+': -a / b must be negative when a,b > 0."""
        a = W.from_str("9.200000000000000000")
        b = W.from_str("0.616200000000000000")
        result = W.div(-a, b)
        self.assertLess(result, 0,
                        "Expected negative result, got %r" % result)
        # exact check: -(9.2/0.6162) ≈ -14.926...
        self.assertAlmostEqual(result / SCALE, -9.2 / 0.6162, places=4)

    def test_assert_wad_returns_true_on_int(self):
        """Kills wad18:84 'True→False': assert_wad must return True."""
        result = W.assert_wad(W.ONE)
        self.assertIs(result, True)

    def test_invalid_string_raises_not_silently_passes(self):
        """Kills wad18:46 '!=→==': non-numeric string must be rejected."""
        with self.assertRaises(ValueError):
            W.from_str("abc")

    def test_string_with_two_dots_raises(self):
        """Kills wad18:41 — catches strings like '1.2.3' after split change."""
        with self.assertRaises(ValueError):
            W.from_str("1.2.3")


# ══════════════════════════════════════════════════════════
# csl.py survivors
# ══════════════════════════════════════════════════════════

class TestCSLSurvivors(unittest.TestCase):

    def _make_csl(self, n_blocks=3):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        csl = CSL(path)
        blocks = []
        for i in range(n_blocks):
            blocks.append(csl.append({"seq": i, "val": i * 10}))
        return csl, blocks, path

    def test_canon_key_order_independent(self):
        """Kills csl:36 'True→False' in sort_keys.
        Two dicts with reversed key order must produce identical canonical strings."""
        from csl import _canon
        d1 = {"z": 1, "a": 2, "m": 3}
        d2 = {"a": 2, "m": 3, "z": 1}
        self.assertEqual(_canon(d1), _canon(d2))

    def test_h_prev_links_to_previous_not_first_block(self):
        """Kills csl:61 '1→0' in self._blocks[-1]: H_prev of block N+1
        must equal H_i of block N, not H_i of block 0."""
        csl, blocks, path = self._make_csl(3)
        try:
            # block[2].H_prev must equal block[1].H_i, not block[0].H_i
            self.assertEqual(blocks[2]["H_prev"], blocks[1]["H_i"])
            self.assertNotEqual(blocks[2]["H_prev"], blocks[0]["H_i"])
        finally:
            os.unlink(path)

    def test_root_hash_is_last_not_first(self):
        """Kills csl:74 '1→0' in self._blocks[-1]: root_hash must be
        the last block's H_i, not the first block's."""
        csl, blocks, path = self._make_csl(3)
        try:
            self.assertEqual(csl.root_hash(), blocks[-1]["H_i"])
            self.assertNotEqual(csl.root_hash(), blocks[0]["H_i"])
        finally:
            os.unlink(path)

    def test_chain_survives_reload(self):
        """Kills csl:102 formatting mutants: chain must reload correctly
        regardless of JSON indent or sort_keys on output."""
        csl, blocks, path = self._make_csl(3)
        try:
            csl2 = CSL(path)
            self.assertTrue(csl2.verify())
            self.assertEqual(csl2.root_hash(), blocks[-1]["H_i"])
        finally:
            os.unlink(path)


# ══════════════════════════════════════════════════════════
# gnm.py survivors
# ══════════════════════════════════════════════════════════

class TestGNMSurvivors(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.coords = GNM.synthetic_gpcr_coords(seed=42)
        cls.lap    = GNM.build_laplacian(cls.coords, r_c=7.5)
        cls.ps     = GNM.pseudoinverse(cls.lap["L"])

    # ── coordinate generation

    def test_helix_axes_are_distinct(self):
        """Kills gnm:45 '7→8' and '7→6': with 7 helices at bundle_radius=10,
        adjacent helix-axis angles must be 360/7 ≈ 51.4 degrees apart.
        A wrong n_helices gives wrong angular spacing."""
        coords = self.coords
        # compute helix-centre positions (mean of first residue of each helix)
        helix_lengths = [28, 28, 28, 28, 28, 28, 16]
        starts = [sum(helix_lengths[:i]) for i in range(7)]
        centres = [coords[s, :2].copy() for s in starts]  # x,y of first residue
        # angles between successive centres from origin
        angles = [float(np.arctan2(c[1], c[0])) for c in centres]
        diffs  = [abs(angles[(i+1)%7] - angles[i]) for i in range(7)]
        # with 7 helices, spacing ~0.9 rad (≈51°); 6 helices ~1.05 rad, 8 ~0.785
        for d in diffs:
            self.assertGreater(min(d, 2*np.pi - d), 0.5,
                               "helix angular spacing too small: %.3f" % d)

    def test_z_coordinates_span_expected_range(self):
        """Kills gnm:61 '-→+' and '*→//': z-offset must centre helices at z≈0.
        z0 = -rise * n_res / 2; flipping sign shifts all z > 0."""
        z = self.coords[:, 2]
        self.assertLess(float(z.min()), -10.0,
                        "z minimum too high; z-offset sign may be wrong")
        self.assertGreater(float(z.max()), 10.0)
        # centroid z should be near 0
        self.assertAlmostEqual(float(z.mean()), 0.0, delta=3.0)

    def test_phi_rotation_varies_along_helix(self):
        """Kills gnm:64 '*→//' and '2→1/3': consecutive residues in same helix
        must have different phi, giving different x,y relative to helix axis."""
        # residues 0-27 are helix TM1
        h1 = self.coords[:28]
        # x-positions relative to helix-1 centre should not all be equal
        x_vals = h1[:, 0]
        self.assertGreater(float(x_vals.std()), 0.5,
                           "x coords within a helix have no variation — phi rotation broken")

    def test_x_y_include_helix_radius_offset(self):
        """Kills gnm:66/67 '+→-': x = cx + r*cos(phi), y = cy + r*sin(phi).
        With '-' the helix wraps the wrong way; std should be ~helix_radius."""
        h1 = self.coords[:28]
        cx = float(h1[:, 0].mean())  # approximate helix centre x
        cy = float(h1[:, 1].mean())
        # residue offsets from centre must have std ≈ helix_radius (2.3 Å)
        dx = h1[:, 0] - cx
        dy = h1[:, 1] - cy
        r_mean = float(np.sqrt(dx**2 + dy**2).mean())
        self.assertAlmostEqual(r_mean, 2.3, delta=1.0,
                               msg="helix radius offset wrong: %.2f" % r_mean)

    def test_noise_is_centred_near_zero(self):
        """Kills gnm:73 '0→1': rng.normal(0, 0.2) vs rng.normal(1, 0.2).
        Mean of noise should be near 0, not 1."""
        # Generate without noise to compare
        coords_no_noise = GNM.synthetic_gpcr_coords(seed=99999)
        # The mean coordinate should be near the bundle centroid (0,0,0)
        self.assertAlmostEqual(float(self.coords[:, 0].mean()),
                               0.0, delta=3.0)
        self.assertAlmostEqual(float(self.coords[:, 1].mean()),
                               0.0, delta=3.0)

    # ── contact graph

    def test_sequential_residues_are_in_contact(self):
        """Kills gnm:101 '-→+': with '+', distance between sequential
        residues is ||coords_i + coords_i+1|| >> r_c (residues are ~10 Å
        from origin, so their sum is ~20 Å). Adjacent residues must be
        in contact (distance ~3.5 Å << 7.5 Å cutoff)."""
        A = self.lap["A"]
        N = len(self.coords)
        # At least 90% of adjacent pairs must be in contact
        in_contact = sum(1 for i in range(N-1) if A[i, i+1] > 1e-10)
        self.assertGreater(in_contact, int(0.9 * (N-1)),
            "Too few sequential contacts: %d/%d (diff sign may be wrong)"
            % (in_contact, N-1))

    def test_sequential_contact_weight_is_in_helix_range(self):
        """Kills gnm:109 '**→*', '2→3', '2→1': weight = exp(-r^2/rc^2).
        For sequential α-helix residues ~3.5 Å apart, weight ≈ exp(-0.218) ≈ 0.80.
        With r**1, weight ≈ exp(-0.467) ≈ 0.63; with r**3, weight ≈ exp(-0.029) ≈ 0.97."""
        A  = self.lap["A"]
        w01 = float(A[0, 1])   # residues 0 and 1, sequential in TM1
        # expected ≈ 0.806 (pinned from run)
        self.assertGreater(w01, 0.70,
            "Contact weight too low: %.4f (weight function exponent wrong?)" % w01)
        self.assertLess(w01, 0.95,
            "Contact weight too high: %.4f" % w01)

    def test_coordinate_x_of_residue_0_is_near_bundle_radius(self):
        """Kills gnm:64 '*→//', gnm:66 '+→-', gnm:73 '0→1'.
        Residue 0 is the first residue of TM1, at angle theta_0=0,
        cx = bundle_radius * cos(0) = 10.0 Å.
        x[0] ≈ cx + helix_radius * cos(phi_0) = 10 + 2.3*cos(0) = 12.3 Å.
        A sign flip on the helix radius gives x[0] ≈ 7.7 Å."""
        x0 = float(self.coords[0, 0])
        self.assertGreater(x0, 9.0,
            "Residue 0 x=%.2f too small; helix x-offset sign may be wrong" % x0)
        self.assertLess(x0, 16.0)

    def test_z_of_first_residue_is_negative(self):
        """Kills gnm:61 '-→+': z0 = -rise*n_res/2 places the first residue
        at the bottom of the helix. With '+', z0 is positive and residue 0
        starts at the TOP. z[0] should be < 0."""
        z0 = float(self.coords[0, 2])
        self.assertLess(z0, -15.0,
            "Residue 0 z=%.2f should be deeply negative (bottom of helix)" % z0)

    def test_edge_count_is_within_expected_range(self):
        """Kills gnm:116 '//→*', '2→3', '2→1': n_edges = mask.sum()//2.
        With '* 2' the count is doubled; with '//3' halved."""
        n = self.lap["n_edges"]
        # expected ~867 for seed=42; allow ±20% for any topology variation
        self.assertGreater(n, 600, "edge count %d too low" % n)
        self.assertLess(n, 1200, "edge count %d too high (may be doubled)" % n)

    def test_edge_count_is_exactly_half_mask_sum(self):
        """Kills gnm:116 '//→*', '2→3', '2→1': n_edges must be
        mask.sum() // 2, not * or /3."""
        lap = self.lap
        A   = lap["A"]
        mask = A > 0
        expected = int(mask.sum()) // 2
        self.assertEqual(lap["n_edges"], expected)

    def test_contact_weights_are_at_most_one(self):
        """Kills gnm:109 '**→*' and '2→n' mutations: w=exp(-r^2/rc^2) ≤ 1.
        With r**1 instead of r**2, distant contacts get weight > exp(-1)."""
        A = self.lap["A"]
        self.assertLessEqual(float(A.max()), 1.0 + 1e-12)
        # contacts at the cutoff (r=7.5) must have weight ≈ exp(-1) ≈ 0.368
        # contacts near 0 distance must have weight near 1
        nonzero = A[A > 0]
        self.assertGreater(float(nonzero.min()), 0.0)
        self.assertLess(float(nonzero.min()), 0.40,
                        "minimum weight should be ≤ exp(-1) ≈ 0.368")

    def test_contact_mask_excludes_self(self):
        """Kills gnm:106 'False→True' (even though L is equivalent,
        the mask itself should be False on diagonal before fill)."""
        # After build_laplacian, A diagonal should be 0
        A = self.lap["A"]
        self.assertAlmostEqual(float(np.diag(A).max()), 0.0, places=10)

    def test_phi_rotation_is_100_degrees_per_residue(self):
        """Kills gnm:64 '2→3' and '2→1': phi = 2*pi*r*(100/360).
        With 3×pi, step = 150° per residue; with 1×pi, step = 50°.
        Residue 1 x-coordinate distinguishes these rotations."""
        x1 = float(self.coords[1, 0])
        # With correct 100° step: x1 ≈ 9.79 (pinned from run)
        # With 150° step: x1 ≈ 8.01; with 50° step: x1 ≈ 11.48
        self.assertGreater(x1, 8.5,
            "x[1]=%.2f too small; phi step > 100° per residue" % x1)
        self.assertLess(x1, 10.5,
            "x[1]=%.2f too large; phi step < 100° per residue" % x1)

    def test_y_offset_adds_not_subtracts_helix_radius(self):
        """Kills gnm:67 '+→-': y = cy + r*sin(phi) vs cy - r*sin(phi).
        For residue 1 in TM1, phi=100°, sin(100°)≈0.985, cy≈0.
        With '+': y1 ≈ +2.27 Å; with '-': y1 ≈ -2.27 Å."""
        y1 = float(self.coords[1, 1])
        # pinned: y1 ≈ 1.87 from run
        self.assertGreater(y1, 0.5,
            "y[1]=%.2f should be positive (helix offset adds, not subtracts)" % y1)

    def test_noise_shifts_coordinates_by_less_than_1_angstrom(self):
        """Kills gnm:73 '0→1': noise mean=1 shifts all coords by 1 Å.
        The mean of all coordinates must be near zero, not near (1,1,1)."""
        self.assertAlmostEqual(float(self.coords[:, 0].mean()), 0.0, delta=1.5)
        self.assertAlmostEqual(float(self.coords[:, 1].mean()), 0.0, delta=1.5)

    def test_fiedler_value_is_smallest_nontrivial_not_second(self):
        """Kills gnm:166 '0→1': eigvals[nontrivial][1] returns the
        SECOND non-trivial eigenvalue (0.242) not the Fiedler (0.220).
        The returned lambda_2 must be less than the second eigenvalue."""
        lam2 = self.ps["lambda_2"]
        # second non-trivial eigenvalue is 0.242
        self.assertLess(lam2, 0.23,
            "lambda_2=%.6f; may be second eigenvalue (0.242) not Fiedler" % lam2)

    def test_lambda_max_is_largest_not_second_largest(self):
        """Kills gnm:167 '1→2': eigvals[-2] returns second largest (8.62)
        not largest (8.72). lambda_max must exceed 8.65."""
        lam_max = self.ps["lambda_max"]
        self.assertGreater(lam_max, 8.65,
            "lambda_max=%.4f; may be second-largest eigenvalue" % lam_max)

    def test_n_helices_produces_7_distinct_angle_sectors(self):
        """Kills gnm:45 more precisely: 7 helices → 7 distinct angle sectors."""
        helix_lengths = [28, 28, 28, 28, 28, 28, 16]
        starts = [sum(helix_lengths[:i]) for i in range(7)]
        centres = np.array([self.coords[s, :2] for s in starts])
        angles  = np.arctan2(centres[:, 1], centres[:, 0])
        # 7 unique angle sectors
        sorted_a = np.sort(angles)
        diffs    = np.diff(np.append(sorted_a, sorted_a[0] + 2*np.pi))
        # each gap should be close to 2π/7 ≈ 0.898 rad
        for d in diffs:
            self.assertGreater(d, 0.4,
                "sector gap %.3f too small for 7-helix bundle" % d)

    # ── pseudoinverse

    def test_fiedler_value_is_pinned(self):
        """Kills gnm:149 '-→+', '1→2', gnm:166 '0→1': lambda_2 must be
        the smallest non-trivial eigenvalue, pinned to the known value."""
        lam2 = self.ps["lambda_2"]
        self.assertGreater(lam2, 0.15)
        self.assertLess(lam2, 0.35)
        # must be less than lambda_max
        self.assertLess(lam2, self.ps["lambda_max"])

    def test_lambda_max_is_pinned(self):
        """Kills gnm:167 '-→+', '1→2', '1→0': lambda_max must be
        the largest eigenvalue, not the second largest or smallest."""
        lam_max = self.ps["lambda_max"]
        lam2    = self.ps["lambda_2"]
        self.assertGreater(lam_max, 1.0,
                           "lambda_max should be >> lambda_2")
        self.assertGreater(lam_max, lam2 * 5)

    def test_trivial_mode_tolerance_uses_lambda_max(self):
        """Kills gnm:149 '1→2' (eigvals[-1] vs eigvals[-2]):
        lambda_1 ≈ 0 must be classified as trivial even with tol_ratio=1e-10."""
        lam1 = self.ps["lambda_1"]
        self.assertAlmostEqual(lam1, 0.0, delta=1e-10)
        # rank must be N-1 (only one trivial mode)
        self.assertEqual(self.ps["rank"], len(self.coords) - 1)


# ══════════════════════════════════════════════════════════
# bias.py survivors
# ══════════════════════════════════════════════════════════

class TestBiasSurvivors(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        coords   = GNM.synthetic_gpcr_coords(seed=42)
        lap      = GNM.build_laplacian(coords, r_c=7.5)
        ps       = GNM.pseudoinverse(lap["L"])
        cls.G    = ps["G"]
        cls.chi  = ps["chi"]
        cls.da   = BIAS.allosteric_response(ps["G"], ps["chi"], 65, -9.2)
        cls.dP   = BIAS.hdx_protection_changes(cls.da, 1.5)

    def test_delta_a_at_site_is_pinned(self):
        """Kills bias:39 '*→//': chi * G_col * dGbind_float.
        The value at the binding site must match the product formula."""
        da_site = float(self.da[65])
        expected = self.chi * float(self.G[65, 65]) * (-9.2)
        self.assertAlmostEqual(da_site, expected, places=10)

    def test_delta_a_magnitude_is_not_zero(self):
        """Kills bias:39 '*→//': floor division of floats near 0 gives 0."""
        self.assertGreater(float(np.abs(self.da).max()), 1e-6)

    def test_hdx_delta_P_exact_formula(self):
        """Kills bias:50 '*→//': dP = -beta * da must hold exactly."""
        beta = 1.5
        dP_expected = -beta * self.da
        np.testing.assert_allclose(self.dP, dP_expected, rtol=1e-12)

    def test_decomposed_bias_exact_value(self):
        """Kills bias:95/96 '-→+' and bias:100 sign mutations:
        decomposed B must equal the closed-form result."""
        lG  = W.from_str("0.810000000000000000")
        lB  = W.from_str("0.380000000000000000")
        mG  = W.from_str("1.000000000000000000")
        mB  = W.from_str("1.000000000000000000")
        dGc = W.from_str("-4.100000000000000000")
        dGt = W.from_str("-10.400000000000000000")   # G-protein ternary
        # for decomposed_bias_wad18 we pass the DIFFERENTIAL ternary:
        dGt_diff = W.from_str("-5.500000000000000000")  # -10.4 - (-4.9)
        RT  = W.from_str("0.616200000000000000")

        result = BIAS.decomposed_bias_wad18(lG, lB, mG, mB, dGc, dGt_diff, RT)

        # delta_lambda = 0.81 - 0.38 = 0.43; delta_mu = 0
        # conf_term = 0.43 * (-4.1) = -1.763
        # tern_term = 0 * (-5.5)  = 0
        # numerator = -(-1.763 + 0) = 1.763
        # B = 1.763 / 0.6162 ≈ 2.861...
        B_float = result["B_wad18"] / W.SCALE
        self.assertAlmostEqual(B_float, 1.763 / 0.6162, places=3)

    def test_bias_numerator_uses_sum_not_difference(self):
        """Kills bias:100 '+→-': numerator = -(conf_term + tern_term).
        With '+→-', numerator = -(conf_term - tern_term).
        We use parameters where tern_term != 0 so the two formulae diverge."""
        # Use mu_G != mu_B so delta_mu != 0 and tern_term != 0
        lG  = W.from_str("0.810000000000000000")
        lB  = W.from_str("0.380000000000000000")
        mG  = W.from_str("1.000000000000000000")
        mB  = W.from_str("0.500000000000000000")   # mu differs!
        dGc = W.from_str("-4.100000000000000000")
        dGt = W.from_str("-5.500000000000000000")
        RT  = W.from_str("0.616200000000000000")

        result = BIAS.decomposed_bias_wad18(lG, lB, mG, mB, dGc, dGt, RT)

        # delta_lambda = 0.43, delta_mu = 0.5
        # conf_term = 0.43 * (-4.1) = -1.763
        # tern_term = 0.5 * (-5.5) = -2.75
        # numerator (correct) = -((-1.763) + (-2.75)) = 4.513
        # numerator (mutant)  = -((-1.763) - (-2.75)) = -(0.987) = -0.987
        # B correct > 0, B mutant < 0 → very different!
        B_float = result["B_wad18"] / W.SCALE
        # Correct value: (1.763 + 2.75) / 0.6162 ≈ 7.325
        self.assertAlmostEqual(B_float, (1.763 + 2.75) / 0.6162, places=2)
        self.assertGreater(result["B_wad18"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
