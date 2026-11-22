"""Array-level statistics shared by archived and user-defined domain analyses."""

import math

import numpy as np
from scipy import stats

from .bias_norms import EMOTIONS
from .experiment_data import signed_rank


def normalize(matrix, order):
    """Normalize full columns, then full rows, before selecting rated targets."""
    if not np.isfinite(matrix).all() or (matrix < 0).any():
        raise ValueError("Activation matrix contains negative or non-finite values")
    values = np.array(matrix, dtype=float, copy=True)
    for axis in (0, 1):
        norms = np.linalg.norm(values, ord=order, axis=axis, keepdims=True)
        values /= np.where(norms == 0, 1, norms)
    return values


def equal_slopes(x, y):
    """Gaussian GLM with intercept and the null hypothesis of equal slopes."""
    n, k = x.shape
    if n <= k + 1 or k < 2:
        return None
    full = np.column_stack((np.ones(n), x))
    coefficients, _, rank, _ = np.linalg.lstsq(full, y, rcond=None)
    if rank != k + 1:
        return None
    reduced = np.column_stack((np.ones(n), x.sum(axis=1)))
    common, _, reduced_rank, _ = np.linalg.lstsq(reduced, y, rcond=None)
    if reduced_rank != 2:
        return None
    full_sse = np.square(y - full @ coefficients).sum()
    reduced_sse = np.square(y - reduced @ common).sum()
    if full_sse <= 0 or reduced_sse <= 0:
        return None
    improvement = max(0.0, reduced_sse - full_sse)
    statistic = improvement / (full_sse / (n - rank))
    return {"n": n, "df": k - 1, "statistic": float(statistic),
            "p": float(stats.chi2.sf(statistic, k - 1)), "partial_r2": float(improvement / reduced_sse)}


def valence_statistics(raw, primes, aligned_ratings):
    matrix = normalize(raw, 2)
    rated = np.isfinite(aligned_ratings)
    x, y = matrix[rated], aligned_ratings[rated]
    n = len(y)
    rows = []
    for j, prime in enumerate(primes):
        row = dict(prime=prime, n=n, slope=None, p=None, ci_low=None, ci_high=None)
        if n >= 3 and np.ptp(x[:, j]) > 0 and np.ptp(y) > 0:
            fit = stats.linregress(x[:, j], y)
            margin = float(stats.t.ppf(.975, n - 2) * fit.stderr)
            row.update(slope=float(fit.slope), p=float(fit.pvalue),
                       ci_low=float(fit.slope - margin), ci_high=float(fit.slope + margin))
        rows.append(row)
    return {"rows": rows, "joint": equal_slopes(x, y)}


def emotion_statistics(pair, indices):
    activation = normalize(pair, 1)
    difference = activation[:, 0] - activation[:, 1]
    rows = []
    for emotion in EMOTIONS:
        d = difference[indices[emotion]]
        effect, _ = signed_rank(d.tolist())
        p = math.erfc(abs(effect) * math.sqrt(len(d) / 2)) if effect is not None else None
        rows.append({"emotion": emotion, "n": len(d), "nonzero": int(np.count_nonzero(d)),
                     "effect": effect, "p": p})
    return rows
