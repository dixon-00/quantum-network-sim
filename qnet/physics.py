"""
Classical physics of a fibre entanglement link: loss, detectors, timing and
accidental coincidences.

Everything here is ordinary (classical) engineering maths. The quantum part
lives in quantum.py. The bridge between the two is the "Werner weight" w:
the fraction of detected coincidences that really come from an entangled pair.

All parameter values are illustrative, typical-of-the-literature numbers.
They are not the parameters of any specific CERN or Geneva Quantum Network link.
"""

import numpy as np
from scipy.special import erf

# ---------------------------------------------------------------------------
# Default parameters (change these in the notebooks to explore)
# ---------------------------------------------------------------------------
DEFAULTS = dict(
    alpha_db_per_km=0.2,       # standard telecom fibre loss at 1550 nm
    pair_rate=1e6,             # entangled pairs emitted per second by the source
    eta_det=0.85,              # detector efficiency (superconducting nanowire detectors)
    dark_rate=100.0,           # dark counts per second per detector
    det_jitter=20e-12,         # timing jitter of each detector (s, standard deviation)
    w_source=0.98,             # intrinsic quality of the source (Werner weight at 0 km)
    fibre_speed_km_s=2.0e5,    # light in fibre: ~200,000 km/s, i.e. ~5 us per km
)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------
def fibre_transmission(length_km, alpha_db_per_km=DEFAULTS["alpha_db_per_km"]):
    """Probability a photon survives `length_km` of fibre (0.2 dB/km -> 1% after 100 km)."""
    return 10 ** (-alpha_db_per_km * np.asarray(length_km) / 10)


def classical_delay(length_km, speed=DEFAULTS["fibre_speed_km_s"]):
    """One-way time (s) for a classical message to cross `length_km` of fibre."""
    return np.asarray(length_km) / speed


# ---------------------------------------------------------------------------
# Timing: how many true pairs fall inside the coincidence window
# ---------------------------------------------------------------------------
def total_jitter(sync_jitter, det_jitter=DEFAULTS["det_jitter"]):
    """Spread of the arrival-time difference between the two detectors.

    Two detector jitters plus the clock synchronisation error between the
    two nodes, added in quadrature (independent Gaussian errors).
    """
    return np.sqrt(2 * det_jitter**2 + np.asarray(sync_jitter) ** 2)


def window_capture(window, sigma, offset=0.0):
    """Fraction of true pairs whose time difference lands inside a window
    of full width `window`, given Gaussian spread `sigma` and an uncorrected
    clock offset `offset` (all in seconds)."""
    s = np.sqrt(2) * sigma
    return 0.5 * (erf((window / 2 - offset) / s) - erf((-window / 2 - offset) / s))


# ---------------------------------------------------------------------------
# Coincidences on a single link (source at node A, photon B travels L km)
# ---------------------------------------------------------------------------
def link_counts(length_km, window, sync_jitter=10e-12, offset=0.0,
                extra_noise_B=0.0, **kw):
    """Detected rates on one link.

    Returns a dict with
      true   : true coincidences per second (both photons of one pair detected, in window)
      acc    : accidental coincidences per second (unrelated clicks that happen to line up)
      w_acc  : true / (true + acc), the "signal fraction"
      singles_A, singles_B : single-detector click rates
    `extra_noise_B` adds background counts at B (e.g. Raman photons from a
    classical channel sharing the fibre).
    """
    p = {**DEFAULTS, **kw}
    eta_A = p["eta_det"]
    eta_B = p["eta_det"] * fibre_transmission(length_km, p["alpha_db_per_km"])
    sigma = total_jitter(sync_jitter, p["det_jitter"])

    singles_A = p["pair_rate"] * eta_A + p["dark_rate"]
    singles_B = p["pair_rate"] * eta_B + p["dark_rate"] + extra_noise_B

    true = p["pair_rate"] * eta_A * eta_B * window_capture(window, sigma, offset)
    acc = singles_A * singles_B * window          # standard accidental-rate formula
    w_acc = true / (true + acc)
    return dict(true=true, acc=acc, w_acc=w_acc,
                singles_A=singles_A, singles_B=singles_B, eta_B=eta_B)


def link_werner_weight(length_km, window, sync_jitter=10e-12, **kw):
    """Overall Werner weight of the detected pairs: source quality x signal fraction."""
    p = {**DEFAULTS, **kw}
    c = link_counts(length_km, window, sync_jitter, **kw)
    return p["w_source"] * c["w_acc"], c


# ---------------------------------------------------------------------------
# Monte Carlo timestamp simulation (used to check the analytic formulas above)
# ---------------------------------------------------------------------------
def simulate_timestamps(length_km, duration, window, sync_jitter=10e-12,
                        offset=0.0, extra_noise_B=0.0, seed=1, **kw):
    """Generate detector timestamps for `duration` seconds and count coincidences.

    Each emitted pair gets a random emission time. Each photon is detected with
    its efficiency, timestamps get Gaussian jitter, node B's clock is off by
    `offset` plus synchronisation jitter, and dark counts arrive uniformly.
    Then, like a real time tagger, every click at A is matched to the nearest
    click at B, and pairs closer than window/2 count as a coincidence.
    """
    p = {**DEFAULTS, **kw}
    rng = np.random.default_rng(seed)
    n_pairs = rng.poisson(p["pair_rate"] * duration)
    t_emit = np.sort(rng.uniform(0, duration, n_pairs))
    pair_id = np.arange(n_pairs)

    eta_B = p["eta_det"] * fibre_transmission(length_km, p["alpha_db_per_km"])
    detA = rng.random(n_pairs) < p["eta_det"]
    detB = rng.random(n_pairs) < eta_B

    tA = t_emit[detA] + rng.normal(0, p["det_jitter"], detA.sum())
    tB = (t_emit[detB] + rng.normal(0, p["det_jitter"], detB.sum())
          + rng.normal(offset, sync_jitter, detB.sum()))   # B's clock error
    idA, idB = pair_id[detA], pair_id[detB]

    # dark counts / background (id = -1 so they can never be a "true" pair)
    nA = rng.poisson(p["dark_rate"] * duration)
    nB = rng.poisson((p["dark_rate"] + extra_noise_B) * duration)
    tA = np.concatenate([tA, rng.uniform(0, duration, nA)])
    idA = np.concatenate([idA, -np.ones(nA, int)])
    tB = np.concatenate([tB, rng.uniform(0, duration, nB)])
    idB = np.concatenate([idB, -2 * np.ones(nB, int)])

    oA, oB = np.argsort(tA), np.argsort(tB)
    tA, idA, tB, idB = tA[oA], idA[oA], tB[oB], idB[oB]

    # nearest B click for each A click
    j = np.clip(np.searchsorted(tB, tA), 1, len(tB) - 1)
    left, right = tB[j - 1], tB[j]
    jbest = np.where(np.abs(tA - left) <= np.abs(tA - right), j - 1, j)
    dt = tB[jbest] - tA
    hit = np.abs(dt) <= window / 2
    is_true = hit & (idA == idB[jbest])
    return dict(true=is_true.sum() / duration,
                acc=(hit & ~is_true).sum() / duration,
                dt=dt[hit])
