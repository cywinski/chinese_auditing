# ABOUTME: Unit tests for src/rebuttal_stats.py on inputs with known answers.
# ABOUTME: Covers bootstrap counts, masked means, paired bootstrap and recall at fixed FPR.

import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from rebuttal_stats import (
    bootstrap_counts,
    column_ratio_means,
    paired_bootstrap,
    recall_at_fpr,
    threshold_at_fpr,
)


def test_bootstrap_counts_shape_sum_and_determinism():
    counts = bootstrap_counts(n_units=90, n_resamples=50, seed=1)
    assert counts.shape == (50, 90)
    assert (counts.sum(axis=1) == 90).all()
    assert (counts == bootstrap_counts(90, 50, seed=1)).all()
    assert not (counts == bootstrap_counts(90, 50, seed=2)).all()


def test_column_ratio_means_ignores_empty_cells_and_averages_columns():
    sums = np.array([[1.0, 10.0], [3.0, 0.0]])
    n_valid = np.array([[1, 1], [1, 0]])
    # column means: (1+3)/2 = 2 and 10/1 = 10 -> mean 6
    assert column_ratio_means(sums, n_valid, np.ones((1, 2)))[0] == 6.0
    # unit 0 drawn twice, unit 1 never: columns 1 and 10 -> 5.5
    assert column_ratio_means(sums, n_valid, np.array([[2, 0]]))[0] == 5.5
    # only unit 1 drawn: column 1 has no valid value and is left out -> 3
    assert column_ratio_means(sums, n_valid, np.array([[0, 2]]))[0] == 3.0


def test_column_ratio_means_pools_several_values_per_cell():
    # unit 0 holds two values (2 and 4) in its single cell, unit 1 holds one (9)
    sums = np.array([[6.0], [9.0]])
    n_valid = np.array([[2], [1]])
    assert column_ratio_means(sums, n_valid, np.ones((1, 2)))[0] == 5.0
    assert column_ratio_means(sums, n_valid, np.array([[2, 0]]))[0] == 3.0


def test_paired_bootstrap_identical_inputs_give_zero_diff():
    rng = np.random.default_rng(0)
    mask = rng.random((90, 10)) > 0.1
    values = np.where(mask, rng.normal(size=(90, 10)), 0.0)
    counts = bootstrap_counts(90, 200, seed=0)
    result = paired_bootstrap(values, mask, values, mask, counts, ci_level=0.95)
    assert result["diff"] == 0.0
    assert result["ci_low"] == 0.0 and result["ci_high"] == 0.0
    assert result["frac_a_gt_b"] == 0.0


def test_paired_bootstrap_constant_shift_is_recovered_exactly():
    rng = np.random.default_rng(0)
    values = rng.normal(size=(90, 1))
    mask = np.ones((90, 1), dtype=bool)
    counts = bootstrap_counts(90, 200, seed=0)
    result = paired_bootstrap(values + 2.0, mask, values, mask, counts, ci_level=0.95)
    assert np.isclose(result["diff"], 2.0)
    assert np.isclose(result["ci_low"], 2.0) and np.isclose(result["ci_high"], 2.0)
    assert result["frac_a_gt_b"] == 1.0


def test_paired_bootstrap_ci_covers_known_effect_and_is_narrower_than_unpaired_noise():
    rng = np.random.default_rng(0)
    question_effect = rng.normal(scale=20.0, size=(200, 1))
    base = question_effect + rng.normal(scale=1.0, size=(200, 1))
    method = question_effect + 1.0 + rng.normal(scale=1.0, size=(200, 1))
    mask = np.ones((200, 1), dtype=bool)
    counts = bootstrap_counts(200, 2000, seed=0)
    result = paired_bootstrap(method, mask, base, mask, counts, ci_level=0.95)
    assert result["ci_low"] < 1.0 < result["ci_high"]
    # pairing removes the large shared question effect
    assert result["ci_high"] - result["ci_low"] < 1.0
    assert result["frac_a_gt_b"] == 1.0


def test_perfectly_separating_score_gives_auroc_one_and_recall_one():
    positives = np.array([5.0, 6.0, 7.0])
    negatives = np.array([1.0, 2.0, 3.0, 4.0])
    labels = np.r_[np.ones(3), np.zeros(4)]
    assert roc_auc_score(labels, np.r_[positives, negatives]) == 1.0
    assert recall_at_fpr(positives, negatives, target_fpr=0.01) == 1.0


def test_recall_at_fpr_known_values():
    negatives = np.arange(100.0)  # 0..99
    # 1% FPR allows exactly one false positive -> threshold is 98 (only 99 exceeds it)
    assert threshold_at_fpr(negatives, 0.01) == 98.0
    assert threshold_at_fpr(negatives, 0.0) == 99.0
    positives = np.array([50.0, 98.0, 98.5, 200.0])
    assert recall_at_fpr(positives, negatives, 0.01) == 0.5
    # fewer than 100 negatives: no false positive is allowed, threshold is the max negative
    assert threshold_at_fpr(np.array([1.0, 3.0, 2.0]), 0.01) == 3.0
    # fully overlapping (positives below all negatives) gives recall 0
    assert recall_at_fpr(np.array([-1.0, -2.0]), negatives, 0.01) == 0.0
