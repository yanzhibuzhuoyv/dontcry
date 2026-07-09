"""
Statistical analysis of retrieval experiment results (real embedder).
Bootstrap confidence intervals and significance testing for MRR.
Pure numpy implementation.
"""
import json
import numpy as np

DATA_PATH = r"C:\Users\32202\WorkBuddy\2026-07-09-19-13-45\exchange\experiment\results_real.json"
REPORT_PATH = r"C:\Users\32202\WorkBuddy\2026-07-09-19-13-45\exchange\experiment\statistical_report_real.md"


def compute_mrr(ranks):
    """Compute Mean Reciprocal Rank. ranks: array-like, -1 means not recalled."""
    rrs = np.where(ranks >= 0, 1.0 / (ranks + 1), 0.0)
    return np.mean(rrs)


def bootstrap_mrr_ci(ranks, n_resamples=10000, alpha=0.05, seed=42):
    """Bootstrap CI for MRR."""
    rng = np.random.default_rng(seed)
    n = len(ranks)
    boot_means = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        boot_means[i] = compute_mrr(ranks[idx])
    lower = np.percentile(boot_means, 100 * alpha / 2)
    upper = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return lower, upper, boot_means


def bootstrap_delta_ci(baseline_ranks, hybrid_ranks, n_resamples=10000, alpha=0.05, seed=42):
    """Bootstrap CI for delta = MRR_hybrid - MRR_baseline (paired)."""
    rng = np.random.default_rng(seed)
    n = len(baseline_ranks)
    boot_deltas = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        mrr_b = compute_mrr(baseline_ranks[idx])
        mrr_h = compute_mrr(hybrid_ranks[idx])
        boot_deltas[i] = mrr_h - mrr_b
    lower = np.percentile(boot_deltas, 100 * alpha / 2)
    upper = np.percentile(boot_deltas, 100 * (1 - alpha / 2))
    return lower, upper, boot_deltas


def format_mrr(mrr_val):
    return f"{mrr_val:.6f}"


def analyze_subset(name, baseline_ranks, hybrid_ranks, n=10000, alpha=0.05):
    """Run full analysis on a subset and return result dict."""
    mrr_b = compute_mrr(baseline_ranks)
    mrr_h = compute_mrr(hybrid_ranks)
    delta = mrr_h - mrr_b

    ci_b_low, ci_b_high, boot_b = bootstrap_mrr_ci(baseline_ranks, n, alpha)
    ci_h_low, ci_h_high, boot_h = bootstrap_mrr_ci(hybrid_ranks, n, alpha)
    ci_d_low, ci_d_high, boot_d = bootstrap_delta_ci(baseline_ranks, hybrid_ranks, n, alpha)

    significant = ci_d_low > 0 or ci_d_high < 0
    nb_miss = int(np.sum(baseline_ranks < 0))
    nh_miss = int(np.sum(hybrid_ranks < 0))

    return {
        "name": name,
        "count": len(baseline_ranks),
        "mrr_baseline": mrr_b,
        "mrr_hybrid": mrr_h,
        "delta": delta,
        "ci_baseline": (ci_b_low, ci_b_high),
        "ci_hybrid": (ci_h_low, ci_h_high),
        "ci_delta": (ci_d_low, ci_d_high),
        "significant": significant,
        "baseline_miss": nb_miss,
        "hybrid_miss": nh_miss,
        "boot_baseline": boot_b,
        "boot_hybrid": boot_h,
        "boot_delta": boot_d,
    }


def format_report(results):
    lines = []
    lines.append("# Statistical Report — Real Embedder Retrieval Experiment\n")
    lines.append(f"**Date:** 2026-07-09\n")
    lines.append("## Experiment Overview\n")
    lines.append(f"- Total queries: {results[0]['count']}")
    lines.append(f"- Embedding model: BAAI/bge-small-zh-v1.5")
    lines.append(f"- Bootstrap resamples: 10000")
    lines.append(f"- Confidence level: 95% (α = 0.05)\n")

    lines.append("## Overall Results (All 1000 Queries)\n")
    r = results[0]
    lines.append(f"| Metric | Baseline | Hybrid |")
    lines.append(f"|--------|----------|--------|")
    lines.append(f"| MRR | {format_mrr(r['mrr_baseline'])} | {format_mrr(r['mrr_hybrid'])} |")
    lines.append(f"| 95% CI | [{format_mrr(r['ci_baseline'][0])}, {format_mrr(r['ci_baseline'][1])}] | "
                 f"[{format_mrr(r['ci_hybrid'][0])}, {format_mrr(r['ci_hybrid'][1])}] |")
    lines.append(f"| Not Recalled (rank=-1) | {r['baseline_miss']} / {r['count']} | {r['hybrid_miss']} / {r['count']} |\n")
    lines.append(f"**Delta (Hybrid − Baseline) = {format_mrr(r['delta'])}**")
    lines.append(f"- 95% CI: [{format_mrr(r['ci_delta'][0])}, {format_mrr(r['ci_delta'][1])}]")
    lines.append(f"- Significant at α=0.05: **{'YES' if r['significant'] else 'NO'}** (CI {'does not' if r['significant'] else 'does'} cross zero)\n")

    lines.append("## Stratified Analysis\n")

    for r in results[1:]:
        lines.append(f"### {r['name']} ({r['count']} Queries)\n")
        lines.append(f"| Metric | Baseline | Hybrid |")
        lines.append(f"|--------|----------|--------|")
        lines.append(f"| MRR | {format_mrr(r['mrr_baseline'])} | {format_mrr(r['mrr_hybrid'])} |")
        lines.append(f"| 95% CI | [{format_mrr(r['ci_baseline'][0])}, {format_mrr(r['ci_baseline'][1])}] | "
                     f"[{format_mrr(r['ci_hybrid'][0])}, {format_mrr(r['ci_hybrid'][1])}] |")
        lines.append(f"| Not Recalled | {r['baseline_miss']} / {r['count']} | {r['hybrid_miss']} / {r['count']} |\n")
        lines.append(f"**Delta (Hybrid − Baseline) = {format_mrr(r['delta'])}**")
        lines.append(f"- 95% CI: [{format_mrr(r['ci_delta'][0])}, {format_mrr(r['ci_delta'][1])}]")
        lines.append(f"- Significant at α=0.05: **{'YES' if r['significant'] else 'NO'}** (CI {'does not' if r['significant'] else 'does'} cross zero)\n")

    lines.append("## Summary\n")

    # Summary table
    lines.append(f"| Subset | N | Baseline MRR | Hybrid MRR | Delta | 95% CI (Delta) | Significant? |")
    lines.append(f"|--------|---|---|---|------|-----------------|-------------|")
    for r in results:
        sig = "✓" if r['significant'] else "✗"
        lines.append(f"| {r['name']} | {r['count']} | {format_mrr(r['mrr_baseline'])} | {format_mrr(r['mrr_hybrid'])} | "
                     f"{format_mrr(r['delta'])} | [{format_mrr(r['ci_delta'][0])}, {format_mrr(r['ci_delta'][1])}] | {sig} |")

    lines.append("\n---\n")
    lines.append("_Generated automatically by analyze_real.py_")
    return "\n".join(lines)


def main():
    with open(DATA_PATH) as f:
        data = json.load(f)

    records = data["per_query"]
    all_baseline = np.array([r["baseline_rank"] for r in records], dtype=int)
    all_hybrid = np.array([r["hybrid_rank"] for r in records], dtype=int)

    # Split: even index = exact (qid 0,2,4,...), odd index = fuzzy (qid 1,3,5,...)
    exact_baseline = all_baseline[0::2]
    exact_hybrid = all_hybrid[0::2]
    fuzzy_baseline = all_baseline[1::2]
    fuzzy_hybrid = all_hybrid[1::2]

    N = 10000
    ALPHA = 0.05

    results = []
    results.append(analyze_subset("Overall (All 1000)", all_baseline, all_hybrid, N, ALPHA))
    results.append(analyze_subset("Exact（精确查询）", exact_baseline, exact_hybrid, N, ALPHA))
    results.append(analyze_subset("Fuzzy（模糊查询）", fuzzy_baseline, fuzzy_hybrid, N, ALPHA))

    report = format_report(results)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)

    print(report)


if __name__ == "__main__":
    main()
