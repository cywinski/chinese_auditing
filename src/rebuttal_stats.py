# ABOUTME: Reusable offline statistics for the rebuttal: paired cluster bootstrap and recall at a fixed FPR.
# ABOUTME: Pure numpy; resampling is expressed as per-unit multiplicity counts so paired statistics share indices.

import numpy as np


def bootstrap_counts(n_units: int, n_resamples: int, seed: int) -> np.ndarray:
    """Draw bootstrap resamples of `n_units` units as multiplicity counts.

    Returns an int array of shape (n_resamples, n_units); row b says how many times
    each unit appears in resample b (each row sums to n_units).
    """
    rng = np.random.default_rng(seed)
    return rng.multinomial(n_units, np.full(n_units, 1.0 / n_units), size=n_resamples)


def column_ratio_means(sums: np.ndarray, n_valid: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Mean over columns of the per-column mean over (resampled) units.

    Args:
        sums: (n_units, n_cols) sum of the valid values in each cell (0 if none).
        n_valid: (n_units, n_cols) number of valid values in each cell (a 0/1 mask
            when each cell holds at most one value).
        counts: (n_resamples, n_units) unit multiplicities (use ones((1, n_units))
            for the point estimate).

    Returns:
        (n_resamples,) statistic: per column, sum of values / number of values over the
        resampled units; then the mean over columns. A column with no valid value in a
        resample is left out of that resample's mean (the same rule as
        metrics.aggregate_position_wise); every resample must keep >= 1 column.
    """
    assert sums.shape == n_valid.shape and sums.ndim == 2
    assert counts.shape[1] == sums.shape[0]
    numerator = counts @ sums
    denominator = counts @ n_valid.astype(float)
    has_values = denominator > 0
    assert has_values.any(axis=1).all(), "a resample has no valid values at all"
    column_means = np.divide(
        numerator, denominator, out=np.zeros_like(numerator, dtype=float), where=has_values
    )
    return column_means.sum(axis=1) / has_values.sum(axis=1)


def paired_bootstrap(
    sums_a: np.ndarray,
    n_valid_a: np.ndarray,
    sums_b: np.ndarray,
    n_valid_b: np.ndarray,
    counts: np.ndarray,
    ci_level: float,
) -> dict:
    """Paired bootstrap of statistic(a) - statistic(b) using the same resampled units.

    The statistic is `column_ratio_means`. Returns point estimates, the observed
    difference, the percentile CI of the difference, the bootstrap mean difference
    and the fraction of resamples where a > b.
    """
    ones = np.ones((1, sums_a.shape[0]))
    point_a = float(column_ratio_means(sums_a, n_valid_a, ones)[0])
    point_b = float(column_ratio_means(sums_b, n_valid_b, ones)[0])
    diffs = column_ratio_means(sums_a, n_valid_a, counts) - column_ratio_means(
        sums_b, n_valid_b, counts
    )
    alpha = 1.0 - ci_level
    ci_low, ci_high = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "point_a": point_a,
        "point_b": point_b,
        "diff": point_a - point_b,
        "boot_mean_diff": float(diffs.mean()),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "frac_a_gt_b": float((diffs > 0).mean()),
    }


def threshold_at_fpr(negative_scores: np.ndarray, target_fpr: float) -> float:
    """Lowest threshold t such that the fraction of negatives with score > t is <= target_fpr."""
    negatives = np.sort(np.asarray(negative_scores, dtype=float))[::-1]
    assert len(negatives) > 0
    n_allowed = int(np.floor(target_fpr * len(negatives)))
    return float(negatives[n_allowed])


def recall_at_fpr(
    positive_scores: np.ndarray, negative_scores: np.ndarray, target_fpr: float
) -> float:
    """Fraction of positives scoring above the threshold that gives FPR <= target_fpr on negatives."""
    positives = np.asarray(positive_scores, dtype=float)
    assert len(positives) > 0
    return float((positives > threshold_at_fpr(negative_scores, target_fpr)).mean())
