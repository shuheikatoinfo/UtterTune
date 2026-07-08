#!/usr/bin/env python3
"""Paired 2-system statistical comparison (UtterTune paper, Table 1).

Reproduces the paper's reported significance/effect-size statistics for a
pairwise comparison (e.g. baseline CosyVoice 2 vs. CosyVoice 2 + UtterTune)
on a per-utterance metric (CER, UTMOSv2, speaker similarity, ...).

Per the paper's Table 1 caption, the significance test is a one-sided paired
Wilcoxon signed-rank test (direction chosen by which system has the higher
mean; see `compare_scores_3systems.py` for the 3-system case, which also
applies a multiple-comparison correction across the three pairwise tests).
This script additionally reports paired effect sizes (Cohen's d_p and paired
Cliff's delta) with bootstrap confidence intervals, and an optional bootstrap
CI directly on the mean paired difference.

Expected input CSV schema: one row per utterance, with one column per
system containing that utterance's score under the two systems (paired,
i.e. same utterance/prompt, same row). Column names are given via
`--systems`.

Example (Test Set 1, UTMOSv2 naturalness, baseline vs. UtterTune):

    python -m scripts.eval.compare_scores_2systems \\
        --input_csv utmos_easy_wide.csv \\
        --systems cv2_base cv2_uttertune \\
        --bootstrap --n_iter 1000 --ci 95

`utmos_easy_wide.csv` needs columns `cv2_base` and `cv2_uttertune`, each
holding the per-utterance UTMOSv2 score for that system, aligned row-for-row
by utterance.

Statistical computation is unchanged from the original research script;
only comments/docstrings/CLI help text were cleaned up for this release.
"""
import argparse
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

def bootstrap_diff(a, b, n_iter=1000, ci=95, seed=42):
    """Compute bootstrap CI for paired differences b - a."""
    rng = np.random.default_rng(seed)
    diffs = []
    n = len(a)
    for _ in range(n_iter):
        idx = rng.integers(0, n, n)
        diffs.append((b[idx] - a[idx]).mean())
    lower = np.percentile(diffs, (100 - ci) / 2)
    upper = np.percentile(diffs, 100 - (100 - ci) / 2)
    return lower, upper

def cohen_dp(diff):
    """Paired Cohen's d = mean(diff) / sd(diff)."""
    return diff.mean() / diff.std(ddof=1)

def cliffs_delta_paired(x, y):
    """Paired Cliff's delta (only counting nonzero differences)."""
    d = y - x
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    n_pos = np.sum(d > 0)
    n_neg = np.sum(d < 0)
    return (n_pos - n_neg) / len(d)

def bootstrap_ci(effect_func, x, y, bootstrap=1000, alpha=0.05, seed=42):
    """Bootstrap CI for a paired effect-size statistic.

    x, y : paired 1-D arrays (same length, same row order)
    effect_func : cohen_dp or cliffs_delta_paired
    """
    rng = np.random.default_rng(seed)
    diff = y - x
    stats = []
    for _ in range(bootstrap):
        sample = rng.choice(diff, size=len(diff), replace=True)
        if effect_func is cohen_dp:
            stat = cohen_dp(sample)
        else:  # Cliff's delta needs x, y separately; map the resampled
            # diff onto x=0, y=sample, which is equivalent for this metric.
            stat = cliffs_delta_paired(np.zeros_like(sample), sample)
        stats.append(stat)
    low = np.percentile(stats, 100 * alpha/2)
    high = np.percentile(stats, 100 * (1 - alpha/2))
    point = (effect_func((y - x)) if effect_func is cohen_dp
             else effect_func(x, y))
    return point, (low, high)

def main():
    parser = argparse.ArgumentParser(
        description="Compare a per-utterance metric (CER, UTMOSv2, speaker similarity, ...) across 2 systems with a paired Wilcoxon test + effect sizes + optional bootstrap CI."
    )
    parser.add_argument(
        "--input_csv", type=str, required=True,
        help="CSV file containing paired per-utterance score columns, one column per system."
    )
    parser.add_argument(
        "--systems", nargs=2, default=['A','B'],
        help="Names of the two system columns in the CSV."
    )
    parser.add_argument(
        "--bootstrap", action='store_true',
        help="Also compute bootstrap CI for the paired differences."
    )
    parser.add_argument(
        "--n_iter", type=int, default=1000,
        help="Number of bootstrap iterations."
    )
    parser.add_argument(
        "--ci", type=float, default=95.0,
        help="Confidence level for bootstrap CI."
    )
    parser.add_argument(
        "--upper_threshold", type=float, default=100,
        help="Upper bound to filter out outlier scores."
    )
    parser.add_argument(
        "--lower_threshold", type=float, default=-100,
        help="Lower bound to filter out outlier scores."
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.input_csv)
    num_samples_initial = len(df)
    sys1, sys2 = args.systems

    # Filter rows outside the accepted score range for both systems.
    mask = (
        (args.lower_threshold <= df[sys1]) & (df[sys1] <= args.upper_threshold) &
        (args.lower_threshold <= df[sys2]) & (df[sys2] <= args.upper_threshold)
    )

    df = df[mask].reset_index(drop=True)
    num_samples_new = len(df)
    a = df[sys1].values
    b = df[sys2].values

    print(f"Dropped: {num_samples_initial - num_samples_new} ({100 * (num_samples_initial - num_samples_new) / num_samples_initial}%)")

    # Mean scores
    print(f"Mean of {sys1}: {a.mean():.4f}")
    print(f"Mean of {sys2}: {b.mean():.4f}")

    # One-sided paired Wilcoxon signed-rank test; direction follows the
    # observed sign of the mean difference (matches the paper's "one-sided
    # paired Wilcoxon test" methodology description).
    diff = b.mean() - a.mean()

    if diff > 0:
        stat_w, p_w = wilcoxon(a, b, alternative="less")
    else:
        stat_w, p_w = wilcoxon(a, b, alternative="greater")
    print(f"\nWilcoxon {sys1} vs {sys2}: stat = {stat_w:.3f}, p = {p_w:.4f}, Δscore = {diff:.4f}")

    d_point, d_ci  = bootstrap_ci(cohen_dp, a, b, bootstrap=args.n_iter, alpha=(100 - args.ci)/100, seed=args.seed)
    delta_point, delta_ci = bootstrap_ci(cliffs_delta_paired, a, b, bootstrap=args.n_iter, alpha=(100 - args.ci)/100, seed=args.seed)

    print(f"Cohen's dₚ = {d_point:+.3f}  ({args.ci}% CI {d_ci[0]:+.3f} … {d_ci[1]:+.3f})")
    print(f"Cliff's Δ  = {delta_point:+.3f}  ({args.ci}% CI {delta_ci[0]:+.3f} … {delta_ci[1]:+.3f})")

    # Bootstrap CI on the mean paired difference (optional)
    if args.bootstrap:
        lower, upper = bootstrap_diff(a, b, n_iter=args.n_iter, ci=args.ci, seed=args.seed)
        print(f"\nBootstrap {args.ci}% CI for Δscore (n_iter={args.n_iter}):")
        print(f"{sys2} - {sys1}: mean Δ = {diff:.4f}, CI = [{lower:.4f}, {upper:.4f}]")

if __name__ == "__main__":
    main()
