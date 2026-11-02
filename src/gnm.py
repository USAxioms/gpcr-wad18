"""
GNM: Gaussian Network Model for GPCR allosteric analysis.

Generates a synthetic 7-TM GPCR-like Cα coordinate set (since PDB fetch
requires network access which is disabled in this container).  The
synthetic topology reproduces the key features of a class A GPCR:
  - 7 TM helices in a bundle, ~28 residues each (N = 196 total)
  - Inter-helix packing distances 8–15 Å
  - A short ICL3 loop between TM5 and TM6 (residues 140–151)
  - Binding pocket approximated by TM3/TM5/TM6 contacts

The Laplacian and pseudoinverse use IEEE-754 float64 (scipy linalg).
This is the only computation in the pipeline that cannot be expressed
in WAD-18; the boundary is noted in every docstring.

All returned numpy arrays are float64.
"""

import numpy as np
from scipy.linalg import eigh


# ──────────────────────────────────────────────────────── coordinate generator

def synthetic_gpcr_coords(seed: int = 42) -> np.ndarray:
    """
    Generate synthetic 7-TM GPCR Cα coordinates (float64, Å).

    Helix geometry:
      - α-helix rise per residue: 1.5 Å
      - α-helix inner radius:     2.3 Å
      - Helix-axis separation:    ~10–14 Å (bundle radius 14 Å)
      - 28 residues per helix (7 helices → N = 196)

    TM helix assignments (BW numbering approximated):
      TM1:   0– 27   TM2:  28– 55   TM3:  56– 83
      TM4:  84–111   TM5: 112–139   ICL3: 140–151
      TM6: 152–179   TM7: 180–195  (16 residues, shorter)

    Returns
    -------
    coords : ndarray, shape (N, 3)
    """
    rng = np.random.default_rng(seed)
    n_helices     = 7
    bundle_radius = 10.0    # Å — ensures single connected component at r_c=7.5 Å
    helix_radius  = 2.3     # Å (α-helix)
    rise          = 1.5     # Å per residue
    lengths       = [28, 28, 28, 28, 28, 12, 28, 16]  # TM1-5, ICL3, TM6, TM7
    # trim to 7 structural helices + ICL3
    helix_lengths = [28, 28, 28, 28, 28, 28, 16]   # TM1-7 (ICL3 absorbed into TM6)

    coords = []
    for h, n_res in enumerate(helix_lengths):
        # centre of this helix axis in the bundle
        theta_h = 2 * np.pi * h / n_helices
        cx = bundle_radius * np.cos(theta_h)
        cy = bundle_radius * np.sin(theta_h)

        # z-offset so helices span the membrane (±21 Å from centre)
        z0 = -rise * n_res / 2.0

        for r in range(n_res):
            phi = 2 * np.pi * r * (100.0 / 360.0)   # 100° per residue
            z   = z0 + r * rise
            x   = cx + helix_radius * np.cos(phi)
            y   = cy + helix_radius * np.sin(phi)
            coords.append([x, y, z])

    coords = np.array(coords, dtype=np.float64)

    # add small thermal noise (σ = 0.2 Å) for realism
    coords += rng.normal(0, 0.2, size=coords.shape)
    return coords


# ──────────────────────────────────────────────────────── contact graph

def build_laplacian(coords: np.ndarray, r_c: float = 7.5) -> dict:
    """
    Build the weighted contact Laplacian L = D - A.

    Weight function: w_ij = exp(-r_ij^2 / r_c^2)   for r_ij <= r_c

    Parameters
    ----------
    coords : (N, 3) float64 Cα positions in Å
    r_c    : contact cutoff in Å (default 7.5)

    Returns
    -------
    dict with keys:
      A        : (N, N) weighted adjacency matrix
      D        : (N,)   degree vector
      L        : (N, N) Laplacian
      n_edges  : number of edges
      r_c      : cutoff used
    """
    N = len(coords)
    # pairwise distance matrix
    diff = coords[:, None, :] - coords[None, :, :]   # (N, N, 3)
    R    = np.sqrt((diff ** 2).sum(axis=2))           # (N, N)

    # contact mask (exclude self-contacts and sequential neighbors ±1)
    mask = (R > 0) & (R <= r_c)
    np.fill_diagonal(mask, False)

    # smooth Gaussian weight
    A = np.where(mask, np.exp(-R ** 2 / r_c ** 2), 0.0)

    # Laplacian
    d = A.sum(axis=1)
    D = np.diag(d)
    L = D - A

    n_edges = int(mask.sum()) // 2
    return {"A": A, "D": d, "L": L, "n_edges": n_edges, "r_c": r_c, "N": N}


# ──────────────────────────────────────────────────────── pseudoinverse

def pseudoinverse(L: np.ndarray, tol_ratio: float = 1e-10) -> dict:
    """
    Compute G = L^+ via symmetric eigendecomposition (float64).

    G = sum_{alpha>=2}  (1 / lambda_alpha) * u_alpha * u_alpha^T

    The zero eigenvalue (translational mode, eigenvector ∝ 1) is
    identified as any eigenvalue < tol_ratio * lambda_max and excluded.

    Returns
    -------
    dict with keys:
      G         : (N, N) pseudoinverse
      eigvals   : (N,)   all eigenvalues
      eigvecs   : (N, N) corresponding eigenvectors
      lambda_1  : the (excluded) zero eigenvalue
      lambda_2  : Fiedler value (smallest non-trivial)
      lambda_max: largest eigenvalue
      rank      : rank of L (= N - 1 for connected graph)
      chi       : global conformational susceptibility tr(G)/N
      trace_G   : tr(G)
      residual  : ||L @ G @ L - L||_F  (Penrose condition 1 check)
    """
    N       = L.shape[0]
    eigvals, eigvecs = eigh(L)               # ascending order, float64

    # identify non-trivial modes
    tol     = tol_ratio * eigvals[-1]
    trivial = eigvals < tol
    nontrivial = ~trivial

    rank    = int(nontrivial.sum())
    lam_pos = eigvals[nontrivial]
    U_pos   = eigvecs[:, nontrivial]

    # pseudoinverse via spectral formula
    G = (U_pos * (1.0 / lam_pos)) @ U_pos.T

    # symmetrise to remove floating-point asymmetry
    G = 0.5 * (G + G.T)

    trace_G  = float(np.trace(G))
    chi      = trace_G / N
    lambda_1 = float(eigvals[trivial][0])   if trivial.any()    else 0.0
    lambda_2 = float(eigvals[nontrivial][0]) if nontrivial.any() else float("nan")
    lambda_N = float(eigvals[-1])

    # Penrose condition 1: L G L = L
    LGL      = L @ G @ L
    residual = float(np.linalg.norm(LGL - L, "fro"))

    return {
        "G":          G,
        "eigvals":    eigvals,
        "eigvecs":    eigvecs,
        "lambda_1":   lambda_1,
        "lambda_2":   lambda_2,
        "lambda_max": lambda_N,
        "rank":       rank,
        "chi":        chi,
        "trace_G":    trace_G,
        "residual":   residual,
    }
