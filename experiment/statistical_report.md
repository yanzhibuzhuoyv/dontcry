# Retrieval Experiment Statistical Report

## 1. 实验配置

- Query 数量: 1000
- Baseline 策略: baseline
- Hybrid 策略: hybrid
- top_k: 10
- hybrid_alpha: 0.3
- hybrid_candidates: 40
- Embedding dim: 64
- Bootstrap 重抽样次数: 10000
- 置信水平: 95%

## 2. 描述性统计

### 2.1 召回率

- Baseline 召回: 21/1000 = 0.0210 (2.10%)
- Hybrid 召回: 81/1000 = 0.0810 (8.10%)
- 召回提升: 60 queries (6.00pp)

### 2.2 召回 Query 的排名分布

- Baseline: n=21, mean_rank=4.24, median_rank=4.0, range=[0.0, 9.0], top1=3, top5=12, top10=21

- Hybrid: n=81, mean_rank=0.74, median_rank=0.0, range=[0.0, 6.0], top1=61, top5=75, top10=81


### 2.3 Per-Query RR 差异分布

- Hybrid 优于 Baseline (delta > 0): 78 queries (7.8%)
- Hybrid 劣于 Baseline (delta < 0): 0 queries (0.0%)
- 无差异 (delta == 0): 922 queries (92.2%)
  (其中双方均未召回: 919 queries)

### 2.4 RR 基本统计

| 指标 | Baseline | Hybrid |
|------|----------|--------|
| Mean (MRR) | 0.006826 | 0.067298 |
| Std | 0.063446 | 0.242699 |
| Median | 0.000000 | 0.000000 |
| Max | 1.000000 | 1.000000 |
| Non-zero RR count | 21 | 81 |

## 3. Bootstrap 置信区间 (95% CI)

- **Baseline MRR**: 0.006826, 95% CI = [0.003328, 0.011126]
- **Hybrid MRR**:  0.067298, 95% CI = [0.052631, 0.082845]

## 4. MRR 差异显著性检验

- **Delta (Hybrid - Baseline)**: 0.060472
- **Delta 95% CI**: [0.047387, 0.074652]

- **结论**: CI 不跨零 → **Hybrid 显著优于 Baseline (p < 0.05)**
  Delta 的 95% CI 下限 0.047387 > 0，表明 hybrid 策略在 95% 置信水平下显著优于 baseline。

## 5. 效应量 (Effect Size)

- **Cohen's d**: 0.3409
- **效应量大小**: 小 (small)

  Cohen's d 在 0.2~0.5 之间，表示效应量为「小」，hybrid 策略的改进幅度有限。

## 6. 综合结论

Hybrid 策略在 MRR 上显著优于 Baseline（delta=0.060472，95% CI=[0.047387, 0.074652]，Cohen's d=0.341，小 (small)）。
MRR 相对提升: 885.9%
召回量提升: 60 queries（21 → 81）
