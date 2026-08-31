"""Validate the low-discrepancy initializers against an independent reference.

The Sobol and Halton generators in `edgesnn.initializers` are checked against
John Burkardt's public-domain reference implementations
(https://people.sc.fsu.edu/~jburkardt/py_src/{halton,sobol}), which are
downloaded on first run into a local cache directory.

Three things are checked:

1.  **Halton** — our vectorised radical inverse must reproduce Burkardt's
    `halton(i, m)` exactly, at low and high dimension.
2.  **Sobol** — `torch.quasirandom.SobolEngine` is compared against Burkardt's
    `i4_sobol`. These are NOT expected to be identical beyond dimension 2: the
    two ship different direction-number tables (Joe-Kuo 2008 versus Bratley-Fox
    ACM 659, 1988), which is a legitimate free choice in the construction. What
    must hold is that the one-dimensional uniformity is identical, since that is
    the property the sequence is chosen for. Burkardt's port caps at 40
    dimensions and so cannot generate the 256 needed here — it is an oracle,
    not a substitute.
3.  **Conditioning** — the effective rank of the recurrent matrix each method
    produces, dense and after the sparsity mask, against the Gaussian baseline.
    This is the check that failed for the original one-dimensional construction.

Run with:  uv run python scripts/check_lds_against_burkardt.py
"""
import math
import sys
import urllib.request
from pathlib import Path

import numpy as np
import torch

from edgesnn.initializers import (
    _first_primes,
    _next_prime_above,
    _radical_inverse,
    generate_sparse_mask,
    initialize_reservoir,
)

REFERENCE_URLS = {
    "halton.py": "https://people.sc.fsu.edu/~jburkardt/py_src/halton/halton.py",
    "sobol.py": "https://people.sc.fsu.edu/~jburkardt/py_src/sobol/sobol.py",
}
CACHE_DIR = Path(__file__).resolve().parent / "_burkardt_reference"

N_RES = 256
CONNECTIVITY = 0.1


def fetch_reference():
    """Download the reference sources once, then make them importable."""
    CACHE_DIR.mkdir(exist_ok=True)
    for filename, url in REFERENCE_URLS.items():
        target = CACHE_DIR / filename
        if not target.exists():
            print(f"  downloading {url}")
            urllib.request.urlretrieve(url, target)
    sys.path.insert(0, str(CACHE_DIR))


def effective_rank(matrix):
    """exp(entropy of the normalised singular-value spectrum).

    Preferred over a hard rank threshold because it degrades smoothly: a matrix
    with one dominant direction scores near 1 whatever the numerical tolerance.
    """
    singular = np.linalg.svd(np.asarray(matrix, dtype=np.float64), compute_uv=False)
    spectrum = singular / singular.sum()
    spectrum = spectrum[spectrum > 0]
    return float(np.exp(-(spectrum * np.log(spectrum)).sum()))


def check_halton(halton):
    print("\n1. Halton: our radical inverse vs Burkardt halton(i, m)")
    ok = True
    for dimension in (8, 64, 256):
        bases = _first_primes(dimension)
        indices = torch.arange(1, 33, dtype=torch.long)
        ours = torch.stack([_radical_inverse(indices, b) for b in bases], dim=1).numpy()
        reference = np.array([halton(i, dimension) for i in range(1, 33)])
        deviation = np.abs(ours - reference).max()
        identical = deviation < 1e-12
        ok &= identical
        print(f"   m={dimension:<4d} n=32   max|ours - reference| = {deviation:.3e}"
              f"   {'IDENTICAL' if identical else 'MISMATCH'}")
    return ok


def check_sobol(i4_sobol):
    print("\n2. Sobol: torch.SobolEngine vs Burkardt i4_sobol (reference caps at d=40)")
    for dimension in (1, 2, 3, 40):
        ours = torch.quasirandom.SobolEngine(dimension=dimension, scramble=False).draw(64).numpy()
        reference = np.zeros((64, dimension))
        seed = 0
        for row in range(64):
            reference[row, :], seed = i4_sobol(dimension, seed)
        deviation = np.abs(ours - reference).max()
        note = "identical" if deviation < 1e-12 else "differs (direction numbers)"
        print(f"   d={dimension:<3d} n=64   max deviation = {deviation:.3e}   {note}")

    print("   one-dimensional uniformity (largest gap per dimension, d=8, n=256):")
    for label, points in (("torch", None), ("reference", None)):
        if label == "torch":
            points = torch.quasirandom.SobolEngine(dimension=8, scramble=False).draw(256).numpy()
        else:
            points = np.zeros((256, 8))
            seed = 0
            for row in range(256):
                points[row, :], seed = i4_sobol(8, seed)
        gaps = [np.diff(np.concatenate([[0.0], np.sort(points[:, j]), [1.0]])).max()
                for j in range(8)]
        print(f"     {label:<10s} min {min(gaps):.5f}  max {max(gaps):.5f}"
              f"   (ideal 1/256 = {1 / 256:.5f})")


def check_start_of_sequence():
    print("\n   why the first two Sobol points are skipped:")
    raw = torch.quasirandom.SobolEngine(dimension=N_RES, scramble=False).draw(3)
    shifted = raw * 2.0 - 1.0
    for index in range(3):
        row = shifted[index]
        print(f"     point {index}: all values equal = {bool((row == row[0]).all())}"
              f"   value {row[0].item():+.1f}"
              f"{'   <- saturated row' if index == 0 else ''}"
              f"{'   <- ZERO row (neuron with no recurrent input)' if index == 1 else ''}")


def check_conditioning():
    print("\n3. Conditioning of the 256x256 recurrent matrix (p_c = 0.1, seed 0)")
    mask = generate_sparse_mask((N_RES, N_RES), CONNECTIVITY, seed=0).numpy()
    print(f"   {'method':<12s} {'dense':>8s} {'masked':>8s} {'spec.rad':>9s} {'std(nz)':>9s}")
    for method in ("gaussian", "lcg", "lfsr", "sobol", "halton"):
        dense = initialize_reservoir((N_RES, N_RES), method=method, fan_in=N_RES,
                                     seed=0).numpy()
        sparse = initialize_reservoir((N_RES, N_RES), method=method, fan_in=N_RES,
                                      connectivity=CONNECTIVITY, seed=0).numpy()
        radius = float(np.abs(np.linalg.eigvals(sparse)).max())
        print(f"   {method:<12s} {effective_rank(dense):8.1f} {effective_rank(sparse):8.1f}"
              f" {radius:9.3f} {sparse[sparse != 0].std():9.4f}")
    print("   (dense rank ~1 for sobol/halton would indicate the one-dimensional bug)")


def check_leap():
    print("\n   Halton leap (Kocis & Whiten 1997: a prime that is not one of the bases)")
    bases = _first_primes(N_RES)
    leap = _next_prime_above(bases[-1])
    print(f"     dimensions {N_RES}, largest base {bases[-1]}, chosen leap {leap}"
          f"   in bases: {leap in set(bases)}")


def main():
    print("Cross-checking edgesnn.initializers against Burkardt's reference code")
    fetch_reference()
    from halton import halton
    from sobol import i4_sobol

    halton_ok = check_halton(halton)
    check_sobol(i4_sobol)
    check_start_of_sequence()
    check_leap()
    check_conditioning()

    print("\nHalton exactness against the reference:",
          "PASS" if halton_ok else "FAIL")
    return 0 if halton_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
