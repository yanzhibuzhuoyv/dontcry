"""
Retrieval Experiment Statistical Analysis
- Bootstrap CI for MRR
- Bootstrap CI for MRR difference (delta)
- Cohen's d effect size
- Descriptive statistics
"""
import json
import numpy as np

# --- Configuration ---
DATA_FILE = r"C:\Users\32202\WorkBuddy\2026-07-09-19-13-45\exchange\experiment\results.json"
REPORT_FILE = r"C:\Users\32202\WorkBuddy\2026-07-09-19-13-45\exchange\experiment\statistical_report.md"
N_BOOTSTRAP = 10000
ALPHA = 0.05
RANDOM_SEED = 42

# --- Load Data ---
with open(DATA_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

queries = data["per_query"]
n_queries = len(queries)

# Extract ranks
baseline_ranks = np.array([q["baseline_rank"] for q in queries], dtype=np.float64)
hybrid_ranks = np.array([q["hybrid_rank"] for q in queries], dtype=np.float64)

# --- Reciprocal Rank (RR) ---
# rank >= 0 -> 1/(rank+1), rank == -1 -> 0
baseline_rr = np.where(baseline_ranks >= 0, 1.0 / (np.maximum(baseline_ranks, 0) + 1.0), 0.0)
hybrid_rr = np.where(hybrid_ranks >= 0, 1.0 / (np.maximum(hybrid_ranks, 0) + 1.0), 0.0)

# Observed MRR
mrr_baseline_obs = np.mean(baseline_rr)
mrr_hybrid_obs = np.mean(hybrid_rr)
delta_obs = mrr_hybrid_obs - mrr_baseline_obs

# --- Bootstrap CI ---
rng = np.random.default_rng(RANDOM_SEED)

def bootstrap_ci(rr_values, n_bootstrap, alpha):
    """Bootstrap 95% CI for mean of reciprocal ranks."""
    n = len(rr_values)
    boot_means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(rr_values, size=n, replace=True)
        boot_means[i] = np.mean(sample)
    ci_lower = np.percentile(boot_means, 100 * alpha / 2)
    ci_upper = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return boot_means, ci_lower, ci_upper

# Baseline CI
boot_baseline, ci_baseline_low, ci_baseline_high = bootstrap_ci(
    baseline_rr, N_BOOTSTRAP, ALPHA
)

# Hybrid CI
boot_hybrid, ci_hybrid_low, ci_hybrid_high = bootstrap_ci(
    hybrid_rr, N_BOOTSTRAP, ALPHA
)

# Delta CI (paired bootstrap: resample indices, compute delta each time)
n = n_queries
boot_delta = np.empty(N_BOOTSTRAP)
for i in range(N_BOOTSTRAP):
    idx = rng.integers(0, n, size=n)
    mrr_b = np.mean(baseline_rr[idx])
    mrr_h = np.mean(hybrid_rr[idx])
    boot_delta[i] = mrr_h - mrr_b

ci_delta_low = np.percentile(boot_delta, 100 * ALPHA / 2)
ci_delta_high = np.percentile(boot_delta, 100 * (1 - ALPHA / 2))

# Significance based on CI
significant = (ci_delta_low > 0) or (ci_delta_high < 0)

# --- Cohen's d (effect size) ---
mean_diff = delta_obs
# Pooled std: sqrt((var1 + var2) / 2)
var_baseline = np.var(baseline_rr, ddof=1)
var_hybrid = np.var(hybrid_rr, ddof=1)
pooled_std = np.sqrt((var_baseline + var_hybrid) / 2.0)
cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0.0

# Interpret effect size
if abs(cohens_d) < 0.2:
    effect_size_label = "微小 (negligible)"
elif abs(cohens_d) < 0.5:
    effect_size_label = "小 (small)"
elif abs(cohens_d) < 0.8:
    effect_size_label = "中 (medium)"
else:
    effect_size_label = "大 (large)"

# --- Descriptive Statistics ---

# Recall rate (rank >= 0)
baseline_recalled = np.sum(baseline_ranks >= 0)
hybrid_recalled = np.sum(hybrid_ranks >= 0)
baseline_recall_rate = baseline_recalled / n_queries
hybrid_recall_rate = hybrid_recalled / n_queries

# Rank distribution among recalled queries
baseline_recalled_ranks = baseline_ranks[baseline_ranks >= 0]
hybrid_recalled_ranks = hybrid_ranks[hybrid_ranks >= 0]

def rank_stats(ranks, label):
    if len(ranks) == 0:
        return f"- {label}: 无召回结果\n"
    mean_r = np.mean(ranks)
    median_r = np.median(ranks)
    min_r = np.min(ranks)
    max_r = np.max(ranks)
    top1 = np.sum(ranks == 0)
    top5 = np.sum(ranks < 5)  # ranks 0-4
    top10 = np.sum(ranks < 10)  # ranks 0-9
    return (
        f"- {label}: n={len(ranks)}, "
        f"mean_rank={mean_r:.2f}, median_rank={median_r:.1f}, "
        f"range=[{min_r}, {max_r}], "
        f"top1={top1}, top5={top5}, top10={top10}\n"
    )

# Per-query delta (hybrid RR - baseline RR)
delta_rr = hybrid_rr - baseline_rr
delta_pos = np.sum(delta_rr > 0)
delta_neg = np.sum(delta_rr < 0)
delta_zero = np.sum(delta_rr == 0)

# --- Build Report ---
lines = []

lines.append("# Retrieval Experiment Statistical Report")
lines.append("")
lines.append("## 1. 实验配置")
lines.append("")
lines.append(f"- Query 数量: {n_queries}")
lines.append(f"- Baseline 策略: {data['config']['strategies'][0]}")
lines.append(f"- Hybrid 策略: {data['config']['strategies'][1]}")
lines.append(f"- top_k: {data['config']['top_k']}")
lines.append(f"- hybrid_alpha: {data['config']['hybrid_alpha']}")
lines.append(f"- hybrid_candidates: {data['config']['hybrid_candidates']}")
lines.append(f"- Embedding dim: {data['config']['embedding_dim']}")
lines.append(f"- Bootstrap 重抽样次数: {N_BOOTSTRAP}")
lines.append(f"- 置信水平: {(1-ALPHA)*100:.0f}%")
lines.append("")

lines.append("## 2. 描述性统计")
lines.append("")
lines.append("### 2.1 召回率")
lines.append("")
lines.append(f"- Baseline 召回: {baseline_recalled}/{n_queries} = {baseline_recall_rate:.4f} ({baseline_recall_rate*100:.2f}%)")
lines.append(f"- Hybrid 召回: {hybrid_recalled}/{n_queries} = {hybrid_recall_rate:.4f} ({hybrid_recall_rate*100:.2f}%)")
lines.append(f"- 召回提升: {hybrid_recalled - baseline_recalled} queries ({(hybrid_recall_rate - baseline_recall_rate)*100:.2f}pp)")
lines.append("")

lines.append("### 2.2 召回 Query 的排名分布")
lines.append("")
lines.append(rank_stats(baseline_recalled_ranks, "Baseline"))
lines.append(rank_stats(hybrid_recalled_ranks, "Hybrid"))
lines.append("")

lines.append("### 2.3 Per-Query RR 差异分布")
lines.append("")
lines.append(f"- Hybrid 优于 Baseline (delta > 0): {delta_pos} queries ({delta_pos/n_queries*100:.1f}%)")
lines.append(f"- Hybrid 劣于 Baseline (delta < 0): {delta_neg} queries ({delta_neg/n_queries*100:.1f}%)")
lines.append(f"- 无差异 (delta == 0): {delta_zero} queries ({delta_zero/n_queries*100:.1f}%)")
lines.append(f"  (其中双方均未召回: {np.sum((baseline_rr == 0) & (hybrid_rr == 0))} queries)")
lines.append("")

lines.append("### 2.4 RR 基本统计")
lines.append("")
lines.append(f"| 指标 | Baseline | Hybrid |")
lines.append(f"|------|----------|--------|")
lines.append(f"| Mean (MRR) | {mrr_baseline_obs:.6f} | {mrr_hybrid_obs:.6f} |")
lines.append(f"| Std | {np.std(baseline_rr, ddof=1):.6f} | {np.std(hybrid_rr, ddof=1):.6f} |")
lines.append(f"| Median | {np.median(baseline_rr):.6f} | {np.median(hybrid_rr):.6f} |")
lines.append(f"| Max | {np.max(baseline_rr):.6f} | {np.max(hybrid_rr):.6f} |")
lines.append(f"| Non-zero RR count | {np.sum(baseline_rr > 0)} | {np.sum(hybrid_rr > 0)} |")
lines.append("")

lines.append("## 3. Bootstrap 置信区间 (95% CI)")
lines.append("")
lines.append(f"- **Baseline MRR**: {mrr_baseline_obs:.6f}, 95% CI = [{ci_baseline_low:.6f}, {ci_baseline_high:.6f}]")
lines.append(f"- **Hybrid MRR**:  {mrr_hybrid_obs:.6f}, 95% CI = [{ci_hybrid_low:.6f}, {ci_hybrid_high:.6f}]")
lines.append("")

lines.append("## 4. MRR 差异显著性检验")
lines.append("")
lines.append(f"- **Delta (Hybrid - Baseline)**: {delta_obs:.6f}")
lines.append(f"- **Delta 95% CI**: [{ci_delta_low:.6f}, {ci_delta_high:.6f}]")
lines.append("")

if significant:
    lines.append(f"- **结论**: CI 不跨零 → **Hybrid 显著优于 Baseline (p < 0.05)**")
    if ci_delta_low > 0:
        lines.append(f"  Delta 的 95% CI 下限 {ci_delta_low:.6f} > 0，表明 hybrid 策略在 95% 置信水平下显著优于 baseline。")
    else:
        lines.append(f"  Delta 的 95% CI 上限 {ci_delta_high:.6f} < 0，表明 hybrid 策略在 95% 置信水平下显著劣于 baseline。")
else:
    lines.append(f"- **结论**: CI 跨零 → **Hybrid 与 Baseline 差异不显著 (p >= 0.05)**")
lines.append("")

lines.append("## 5. 效应量 (Effect Size)")
lines.append("")
lines.append(f"- **Cohen's d**: {cohens_d:.4f}")
lines.append(f"- **效应量大小**: {effect_size_label}")
lines.append("")

# Interpret
if abs(cohens_d) >= 0.8:
    lines.append('  Cohen\'s d >= 0.8，表示效应量为「大」，hybrid 策略的改进具有实际意义。')
elif abs(cohens_d) >= 0.5:
    lines.append('  Cohen\'s d 在 0.5~0.8 之间，表示效应量为「中」，hybrid 策略的改进较为明显。')
elif abs(cohens_d) >= 0.2:
    lines.append('  Cohen\'s d 在 0.2~0.5 之间，表示效应量为「小」，hybrid 策略的改进幅度有限。')
else:
    lines.append('  Cohen\'s d < 0.2，表示效应量为「微小」，hybrid 策略的改进在实际中可忽略。')
lines.append("")

lines.append("## 6. 综合结论")
lines.append("")

# Overall conclusion
improv_pct = (mrr_hybrid_obs / mrr_baseline_obs - 1) * 100 if mrr_baseline_obs > 0 else float('inf')
recall_improv = hybrid_recalled - baseline_recalled

if significant:
    lines.append(f"Hybrid 策略在 MRR 上显著优于 Baseline（delta={delta_obs:.6f}，95% CI=[{ci_delta_low:.6f}, {ci_delta_high:.6f}]，Cohen's d={cohens_d:.3f}，{effect_size_label}）。")
    if mrr_baseline_obs > 0:
        lines.append(f"MRR 相对提升: {improv_pct:.1f}%")
    lines.append(f"召回量提升: {recall_improv} queries（{baseline_recalled} → {hybrid_recalled}）")
else:
    lines.append(f"Hybrid 策略与 Baseline 在 MRR 上无显著差异（delta={delta_obs:.6f}，95% CI=[{ci_delta_low:.6f}, {ci_delta_high:.6f}]，Cohen's d={cohens_d:.3f}，{effect_size_label}）。")

lines.append("")

report = "\n".join(lines)

# --- Save Report ---
with open(REPORT_FILE, "w", encoding="utf-8") as f:
    f.write(report)

# --- Print summary to stdout ---
print("========== 统计分析结果 ==========")
print(f"\nBootstrap 重抽样次数: {N_BOOTSTRAP}")
print(f"置信水平: {(1-ALPHA)*100:.0f}%")
print()
print(f"Baseline MRR: {mrr_baseline_obs:.6f}")
print(f"  95% CI: [{ci_baseline_low:.6f}, {ci_baseline_high:.6f}]")
print()
print(f"Hybrid MRR:  {mrr_hybrid_obs:.6f}")
print(f"  95% CI: [{ci_hybrid_low:.6f}, {ci_hybrid_high:.6f}]")
print()
print(f"Delta (Hybrid - Baseline): {delta_obs:.6f}")
print(f"  Delta 95% CI: [{ci_delta_low:.6f}, {ci_delta_high:.6f}]")
print(f"  显著性: {'显著 (CI 不跨零)' if significant else '不显著 (CI 跨零)'}")
print()
print(f"Cohen's d: {cohens_d:.4f}")
print(f"效应量: {effect_size_label}")
print()
print(f"召回率 Baseline: {baseline_recalled}/{n_queries} ({baseline_recall_rate*100:.2f}%)")
print(f"召回率 Hybrid:  {hybrid_recalled}/{n_queries} ({hybrid_recall_rate*100:.2f}%)")
print()
print(f"报告已保存至: {REPORT_FILE}")
print("==================================")
