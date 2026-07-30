"""Brain-age gap (BAG) definitions: raw, bias-corrected, and INT.

All three follow the research code exactly:

raw
    ``raw_bag = predicted_age - chronological_age``

bias-corrected (de Lange & Cole style, division form)
    Fit ``predicted_age = a * chronological_age + b`` by ordinary least squares
    on a *reference* sample — in the paper, the cognitively-normal (CN)
    out-of-fold predictions — then

    ``bias_corrected_bag = (predicted_age - b) / a - chronological_age``

    The coefficients (a, b) are fit once on the reference sample and applied
    unchanged to every other row (other visits, other cohorts, patient groups).

INT (rank-based inverse-normal transformation)
    Applied to the bias-corrected BAG:

    ``ranks = rankdata(x, method='average')`` over the finite values, then
    ``norm.ppf((ranks - 0.5) / n)``; non-finite entries stay NaN.

Order is always raw -> bias correction -> INT.
"""

import numpy as np
from scipy.stats import norm, rankdata


def raw_bag(chronological_age, predicted_age):
    """``predicted_age - chronological_age``."""
    return np.asarray(predicted_age, dtype=float) - np.asarray(chronological_age, dtype=float)


def fit_bias_correction(chronological_age, predicted_age):
    """Fit ``predicted_age = a * age + b`` on the reference sample.

    Returns ``(a, b)``. Use out-of-fold predictions of healthy controls, as in
    the paper; fitting on the same rows the model was trained on would leak.
    """
    age = np.asarray(chronological_age, dtype=float)
    pred = np.asarray(predicted_age, dtype=float)
    finite = np.isfinite(age) & np.isfinite(pred)
    if finite.sum() < 2:
        raise ValueError('Need at least 2 finite (age, prediction) pairs to fit bias correction')
    a, b = np.polyfit(age[finite], pred[finite], 1)
    if a == 0:
        raise ValueError('Degenerate bias-correction fit (slope = 0)')
    return float(a), float(b)


def apply_bias_correction(chronological_age, predicted_age, a, b):
    """``(predicted_age - b) / a - chronological_age``."""
    age = np.asarray(chronological_age, dtype=float)
    pred = np.asarray(predicted_age, dtype=float)
    return (pred - b) / a - age


def inverse_normal_transformation(values):
    """Rank-based INT: ``norm.ppf((rank - 0.5) / n)`` with average ties."""
    values = np.asarray(values, dtype=float)
    out = np.full(values.shape, np.nan, dtype=float)

    valid_mask = np.isfinite(values)
    valid_values = values[valid_mask]

    if len(valid_values) == 0:
        return out

    ranks = rankdata(valid_values, method='average')
    out[valid_mask] = norm.ppf((ranks - 0.5) / len(valid_values))
    return out
