#!/usr/bin/env python3
"""3-system statistical comparison with multiple-comparison correction
(UtterTune paper, Table 1, Test Set 2's 3-way comparison).

Reproduces the paper's reported significance/effect-size statistics for a
3-way comparison (e.g. CosyVoice 2 baseline vs. kana-input baseline vs.
CosyVoice 2 + UtterTune) on a per-utterance metric (typically CER on Test
Set 2). Runs an omnibus Friedman test across the three paired systems, then
pairwise paired Wilcoxon signed-rank tests for all 3 pairs, with a
Holm-Bonferroni correction across those pairwise p-values (the paper's
Table 1 caption describes "Bonferroni correction if needed" for multiple
comparisons; Holm-Bonferroni is the step-down variant used here, which is
uniformly at least as powerful as the single-step Bonferroni correction
while controlling the same family-wise error rate). Also reports paired
effect sizes (Cohen's d_p and paired Cliff's delta) with bootstrap
confidence intervals, and an optional bootstrap CI on each pairwise mean
difference.

Expected input CSV schema: one row per utterance, with one column per
system containing that utterance's score under each of the three systems
(paired, i.e. same utterance/prompt, same row). Column names are given via
`--systems`.

Example (Test Set 2, CER, 3-way baseline / kana-input / UtterTune):

    python -m scripts.eval.compare_scores_3systems \\
        --input_csv cer_difficult_wide.csv \\
        --systems cv2_base cv2_base_kana cv2_uttertune \\
        --bootstrap --n_iter 1000 --ci 95

`cer_difficult_wide.csv` needs columns `cv2_base`, `cv2_base_kana`, and
`cv2_uttertune`, each holding the per-utterance CER for that system, aligned
row-for-row by utterance.

Statistical computation is unchanged from the original research script;
only comments/docstrings/CLI help text were cleaned up for this release.
"""
import argparse
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon
from statsmodels.stats.multitest import multipletests

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
        description="Compare a per-utterance metric (typically CER) across 3 systems with Friedman + pairwise Wilcoxon + Holm-Bonferroni correction."
    )
    parser.add_argument(
        "--input_csv", type=str, required=True,
        help="CSV file containing paired per-utterance score columns, one column per system."
    )
    parser.add_argument(
        "--systems", nargs=3, default=['A','B','C'],
        help="Names of the three system columns in the CSV."
    )
    parser.add_argument(
        "--bootstrap", action='store_true',
        help="Also compute bootstrap CI for pairwise differences."
    )
    parser.add_argument(
        "--n_iter", type=int, default=1000,
        help="Number of bootstrap iterations."
    )
    parser.add_argument(
        "--ci", type=float, default=95.0,
        help="Confidence level for bootstrap CI."
    )
    parser.add_argument("--upper_threshold", type=float, default=100)
    parser.add_argument("--lower_threshold", type=float, default=-100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.input_csv)
    num_samples_initial = len(df)
    sys_names = args.systems
    mask = (args.lower_threshold <= df[sys_names[0]]) \
        & (df[sys_names[0]] <= args.upper_threshold) \
        & (args.lower_threshold <= df[sys_names[1]]) \
        & (df[sys_names[1]] <= args.upper_threshold) \
        & (args.lower_threshold <= df[sys_names[2]]) \
        & (df[sys_names[2]] <= args.upper_threshold)
    df = df[mask].reset_index(drop=True)
    num_samples_new = len(df)
    print(f"Dropped: {num_samples_initial - num_samples_new} ({100 * (num_samples_initial - num_samples_new) / num_samples_initial}%)")

    cer_mat = df[sys_names].values
    print(f"Mean of {sys_names[0]}: {df[sys_names[0]].mean()}")
    print(f"Mean of {sys_names[1]}: {df[sys_names[1]].mean()}")
    print(f"Mean of {sys_names[2]}: {df[sys_names[2]].mean()}")

    # 1. Friedman omnibus test across the three paired systems
    stat, p_friedman = friedmanchisquare(cer_mat[:,0], cer_mat[:,1], cer_mat[:,2])
    print(f"Friedman test: chi2 = {stat:.3f}, p = {p_friedman:.4f}")

    # 2. Pairwise paired Wilcoxon signed-rank tests (one-sided, uncorrected).
    # One-sided, in the direction of the observed mean difference, matching
    # compare_scores_2systems.py and the paper's stated methodology ("one-sided
    # paired Wilcoxon test", Table 1 caption).
    pairs = [(0,1), (0,2), (1,2)]
    p_vals = []
    diffs = []

    for i, j in pairs:
        diff = cer_mat[:, j].mean() - cer_mat[:, i].mean()
        if diff > 0:
            stat_w, p_w = wilcoxon(cer_mat[:, i], cer_mat[:, j], alternative="less")
        else:
            stat_w, p_w = wilcoxon(cer_mat[:, i], cer_mat[:, j], alternative="greater")
        p_vals.append(p_w)
        diffs.append(diff)
        print(f"Wilcoxon {sys_names[i]} vs {sys_names[j]}: stat = {stat_w:.3f}, p = {p_w:.4f}, Δscore = {diff:.4f}")

    # 3. Holm-Bonferroni correction across the three pairwise p-values
    reject, p_corr, _, _ = multipletests(p_vals, alpha=0.05, method='holm')

    for (i, j), p_raw, p_holm, rej in zip(pairs, p_vals, p_corr, reject):
        print(f"Holm-corrected {sys_names[i]} vs {sys_names[j]}: raw p = {p_raw:.4f}, corrected p = {p_holm:.4f}, reject H0: {rej}")

    # 4. Bootstrap CI + effect sizes per pair (optional)
    if args.bootstrap:
        print(f"\nBootstrap {args.ci}% CI for Δscore (n_iter={args.n_iter}):")
        for (i, j), diff in zip(pairs, diffs):
            a = cer_mat[:, i]
            b = cer_mat[:, j]
            lower, upper = bootstrap_diff(a, b, n_iter=args.n_iter, ci=args.ci, seed=args.seed)
            print(f"{sys_names[j]} - {sys_names[i]}: mean Δ = {diff:.4f}, CI[{args.ci}%] = [{lower:.4f}, {upper:.4f}]")

            d_point, d_ci  = bootstrap_ci(cohen_dp, a, b, bootstrap=args.n_iter, alpha=(100 - args.ci)/100, seed=args.seed)
            delta_point, delta_ci = bootstrap_ci(cliffs_delta_paired, a, b, bootstrap=args.n_iter, alpha=(100 - args.ci)/100, seed=args.seed)

            print(f"Cohen's dₚ = {d_point:+.3f}  ({args.ci}% CI {d_ci[0]:+.3f} … {d_ci[1]:+.3f})")
            print(f"Cliff's Δ  = {delta_point:+.3f}  ({args.ci}% CI {delta_ci[0]:+.3f} … {delta_ci[1]:+.3f})")

if __name__ == "__main__":
    main()
