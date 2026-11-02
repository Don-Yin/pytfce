"""Gaussian Random Field helpers for pTFCE.

Expected cluster size, cluster-size PDF, and Bayesian posterior P(h|c).
Ported from the R pTFCE package (github.com/spisakt/pTFCE) and FSL.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import special, stats
from scipy.integrate import quad

_GAMMA_2_5 = float(special.gamma(2.5))
_LOG_2PI = math.log(2.0 * math.pi)
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)
_INV_SQRT_2 = 1.0 / math.sqrt(2.0)
_UPPER_Z = 37.06503921751289  # = min(38.0, norm.isf(1e-300)), precomputed
_SF_UPPER = 1e-300  # norm.sf(_UPPER_Z) ≈ 1e-300
# R pTFCE does not floor E[S] at one voxel inside pvox.clust; set True to restore the earlier python behaviour.
_CLAMP_ES_IN_PVOX = False


def expected_cluster_size(h: np.ndarray, V: int, Rd: float) -> np.ndarray:
    """Expected cluster size at Z-threshold *h* (GRF, FSL port).

    Parameters
    ----------
    h : array-like
        Z-score thresholds.
    V : int
        Number of masked voxels.
    Rd : float
        RESEL count (dLh * V).
    """
    h = np.asarray(h, dtype=np.float64)
    log_num = np.log(V) + stats.norm.logsf(h)
    log_den = np.full_like(h, -np.inf)
    valid = h >= 1.1
    h2 = h[valid] ** 2
    log_den[valid] = np.log(Rd) + np.log(h2 - 1) - h2 / 2.0 - 2.0 * _LOG_2PI
    log_ec = log_num.copy()
    log_ec[valid] = log_num[valid] - log_den[valid]
    return np.exp(log_ec).clip(min=1.0)


def _expected_cluster_size_scalar(h: float, log_V: float, log_Rd: float,
                                  clamp: bool = True) -> float:
    """Scalar fast-path for expected_cluster_size (avoids array alloc).

    ``clamp=False`` reproduces R pTFCE, which does not floor E[S] at 1 voxel.
    """
    sf = 0.5 * math.erfc(h * _INV_SQRT_2)
    # exact log-survival in the far tail: the former "+ 1e-300" made E[S] grow (so lam fall) again
    # beyond z ~ 37, a non-monotone artefact that the exponent scaling in pvox_clust must not see
    log_sf = math.log(sf) if sf > 1e-290 else float(special.log_ndtr(-h))
    log_num = log_V + log_sf
    if h >= 1.1:
        h2 = h * h
        log_den = log_Rd + math.log(h2 - 1.0) - h2 * 0.5 - 2.0 * _LOG_2PI
        log_ec = log_num - log_den
    else:
        log_ec = log_num
    ec = math.exp(log_ec)
    return ec if (ec > 1.0 or not clamp) else 1.0


def cluster_pdf(c: float, lam: float) -> float:
    """PDF of cluster size *c* given rate parameter *lam* (D=3)."""
    if c <= 0:
        return 0.0
    return (2.0 * lam / (3.0 * c ** (1.0 / 3.0))) * np.exp(-lam * c ** (2.0 / 3.0))


def pvox_clust(V: int, Rd: float, c: float, h: float,
               z_est_thr: float = 1.3) -> float:
    """P(Z >= h | cluster_size = c) via Bayes' rule with GRF likelihood.

    Uses numerical integration (scipy.integrate.quad).
    """
    if c <= 0 or not math.isfinite(h):
        return 1.0

    log_V = math.log(V)
    log_Rd = math.log(Rd)
    c_23 = c ** (2.0 / 3.0)
    clamp = _CLAMP_ES_IN_PVOX

    # lam(x) grows with x, so exp(-lam(x) * c**(2/3)) peaks at the LEFT edge of each integral
    # and is exactly 0 at every quad node once lam(edge) * c**(2/3) exceeds ~745 (denominator:
    # c > ~1.1e6 voxels at ADNI-sized V/Rd; numerator: much sooner).  The old guard then
    # returned the marginal sf(h) ("no evidence") instead of ~0 ("maximal evidence").
    # Factor exp(-lam(edge) * c**(2/3)) out of each integral and restore it in log space.
    def _lam(xi: float) -> float:
        ec = _expected_cluster_size_scalar(xi, log_V, log_Rd, clamp)
        return (ec / _GAMMA_2_5) ** (-2.0 / 3.0)

    def _scaled_integral(lo: float) -> tuple[float, float]:
        # integral of  lam * exp(-(lam - lam(lo)) c^(2/3) - (x^2 - lo^2)/2)  over [lo, upper]; the common
        # factor  exp(-lam(lo) c^(2/3)) * phi(lo) * 2 / (3 c^(1/3))  is restored in log space below
        lam_lo = _lam(lo)

        def integrand(xi: float) -> float:
            lam = _lam(xi)
            expo = -(lam - lam_lo) * c_23 - 0.5 * (xi * xi - lo * lo)
            return lam * math.exp(min(expo, 700.0))

        value, _ = quad(integrand, lo, _UPPER_Z, limit=200)
        return value, lam_lo

    numerator, lam_h = _scaled_integral(h)
    denominator, lam_0 = _scaled_integral(z_est_thr)

    if numerator <= 0.0 or denominator <= 0.0:  # still unresolvable (c >> V): saturate, never the marginal
        return _SF_UPPER
    log_p = (math.log(numerator) - math.log(denominator)
             - (lam_h - lam_0) * c_23 - 0.5 * (h * h - z_est_thr * z_est_thr))
    return float(min(max(math.exp(min(log_p, 0.0)), _SF_UPPER), 1.0))


def aggregate_logpvals(s: float, delta: float) -> float:
    """Q-function: equidistant incremental log-probability aggregation."""
    if s <= 0:
        return 0.0
    return 0.5 * (math.sqrt(delta * (8.0 * s + delta)) - delta)


def aggregate_logpvals_vec(s: np.ndarray, delta: float) -> np.ndarray:
    """Vectorised Q-function over an array of accumulated -logp sums."""
    out = np.zeros_like(s)
    pos = s > 0
    out[pos] = 0.5 * (np.sqrt(delta * (8.0 * s[pos] + delta)) - delta)
    return out
