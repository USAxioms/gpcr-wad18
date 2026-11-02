"""
Allosteric propagation and pathway bias index.

The allosteric response is computed in float64 (G_ij is a float64 matrix
element from the pseudoinverse).  Only the free-energy layer — where
ΔGbind, ΔGconf, ΔGternary, and B are computed — uses WAD-18.

The division between float64 (linear algebra) and WAD-18 (thermodynamics)
is enforced by the type-checking in wad18.assert_wad().
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wad18 as W


# ──────────────────────────────── allosteric response (float64 layer)

def allosteric_response(G: "np.ndarray", chi: float,
                        site_i: int, dG_bind_float: float) -> "np.ndarray":
    """
    Compute Δa_j = χ · G_ij · Δμ_i  for all j.

    Parameters
    ----------
    G            : (N, N) pseudoinverse (float64)
    chi          : global susceptibility tr(G)/N (float64)
    site_i       : binding-site residue index (int)
    dG_bind_float: ΔGbind in kcal/mol as Python float (for scaling only)

    Returns
    -------
    delta_a : (N,) float64 array of activity changes
    """
    import numpy as np
    G_col    = G[:, site_i]          # column i of the propagator
    delta_a  = chi * G_col * dG_bind_float
    return delta_a


def hdx_protection_changes(delta_a: "np.ndarray",
                           beta_hdx: float = 1.5) -> "np.ndarray":
    """
    Predicted HDX protection factor changes: ΔP_j = -β_HDX · Δa_j.

    β_HDX ∈ [1, 2] is the empirical HDX proportionality constant.
    """
    return -beta_hdx * delta_a


# ──────────────────────────────── free-energy decomposition (WAD-18 layer)

def effective_free_energy_wad18(dG_bind: int, dG_conf: int, dG_ternary: int,
                                 lambda_k: int, mu_k: int) -> int:
    """
    ΔGeff(path k) = ΔGbind + λk·ΔGconf + μk·ΔGternary   [WAD-18]

    All arguments are WAD-18 integers.
    Returns a WAD-18 integer.
    """
    W.assert_wad(dG_bind, dG_conf, dG_ternary, lambda_k, mu_k)
    return dG_bind + W.mul(lambda_k, dG_conf) + W.mul(mu_k, dG_ternary)


def bias_index_wad18(dGe_k1: int, dGe_k2: int, RT: int) -> int:
    """
    B_{k1→k2} = -[ΔGeff(k1) - ΔGeff(k2)] / RT   [WAD-18]

    The binding free energy cancels exactly because it is pathway-independent
    and appears with the same sign in both ΔGeff(k1) and ΔGeff(k2).

    All arguments are WAD-18 integers.
    Returns a WAD-18 integer.
    """
    W.assert_wad(dGe_k1, dGe_k2, RT)
    numerator = -(dGe_k1 - dGe_k2)    # exact cancellation of ΔGbind
    return W.div(numerator, RT)


def decomposed_bias_wad18(lambda_k1: int, lambda_k2: int,
                           mu_k1: int, mu_k2: int,
                           dG_conf: int, dG_ternary: int,
                           RT: int) -> dict:
    """
    B = -[(λk1 - λk2)·ΔGconf + (μk1 - μk2)·ΔGternary] / RT   [WAD-18]

    Demonstrates exact ΔGbind cancellation at the symbolic level.
    Returns dict with all intermediate WAD-18 values.
    """
    W.assert_wad(lambda_k1, lambda_k2, mu_k1, mu_k2,
                 dG_conf, dG_ternary, RT)

    delta_lambda = lambda_k1 - lambda_k2
    delta_mu     = mu_k1     - mu_k2

    conf_term    = W.mul(delta_lambda, dG_conf)
    tern_term    = W.mul(delta_mu,     dG_ternary)
    numerator    = -(conf_term + tern_term)
    B            = W.div(numerator, RT)

    return {
        "delta_lambda_wad18":  delta_lambda,
        "delta_mu_wad18":      delta_mu,
        "conf_term_wad18":     conf_term,
        "ternary_term_wad18":  tern_term,
        "numerator_wad18":     numerator,
        "B_wad18":             B,
        "B_str":               W.to_str(B, 9),
        "occupancy_ratio":     None,   # filled by caller after exp()
    }
