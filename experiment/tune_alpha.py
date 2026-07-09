#!/usr/bin/env python3
"""Sweep hybrid_alpha to find the optimal BM25/vector fusion weight.

Reuses the exact data generation (seed=42) from run_real.py so results are
directly comparable. For each query we compute the vector candidates and BM25
scores ONCE, then evaluate every alpha cheaply.

Output: tune_results.json + a console table.
"""

import json
import os
import sys
import time
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

EXCHANGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXCHANGE))
# allow importing run_real helpers
sys.path.insert(0, str(EXCHANGE / "experiment"))

from run_real import make_documents, make_queries, _try_create_embedder  # noqa: E402
from rag_system.bm25 import BM25Index  # noqa: E402
from rag_system.documents import Chunk  # noqa: E402
from rag_system.vector_store import VectorStore  # noqa: E402

NUM_DOCS = 500
TOP_K = 10
CANDIDATES = TOP_K * 4  # 40
# alpha = BM25 weight; 1-alpha = vector weight. 0.0 = pure vector, 1.0 = pure BM25.
ALPHAS = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0]
OUT_PATH = Path(__file__).resolve().parent / "tune_results.json"


def _rank_of(cand_sources: list[str], rel: str) -> int:
    for i, s in enumerate(cand_sources):
        if s == rel:
            return i
    return -1


def _mrr(ranks: list[int]) -> float:
    if not ranks:
        return 0.0
    return sum(1.0 / (r + 1) for r in ranks if r >= 0) / len(ranks)


def _recall(ranks: list[int]) -> float:
    if not ranks:
        return 0.0
    return sum(1 for r in ranks if r >= 0) / len(ranks)


def main() -> None:
    t0 = time.time()
    print("=" * 64)
    print("  hybrid_alpha 调参实验 (真实 embedder, 500 文档 / 1000 query)")
    print("=" * 64)

    docs = make_documents(NUM_DOCS)
    embedder, used_fallback = _try_create_embedder()
    dim = embedder.dimension

    store = VectorStore(dim)
    texts = [d["content"] for d in docs]
    embeddings = embedder.embed_documents(texts)
    chunks = [Chunk(text=d["content"], source=d["id"], chunk_index=0, metadata={}) for d in docs]
    store.add_documents(chunks, embeddings)
    print(f"  索引构建完成: {store.count} 向量 (dim={dim})")

    queries = make_queries(docs)
    print(f"  查询: {len(queries)} (exact={sum(1 for q in queries if q['type']=='exact')}, "
          f"fuzzy={sum(1 for q in queries if q['type']=='fuzzy')})")
    print()

    # ranks[alpha] = {"exact": [...], "fuzzy": [...], "all": [...]}
    ranks = {a: {"exact": [], "fuzzy": [], "all": []} for a in ALPHAS}

    for idx, q in enumerate(queries):
        qvec = embedder.embed_query(q["query"])
        cand = store.search(qvec, k=CANDIDATES)
        rel = q["relevant_source"]
        qtype = q["type"]

        if not cand:
            for a in ALPHAS:
                ranks[a][qtype].append(-1)
                ranks[a]["all"].append(-1)
            continue

        cand_texts = [c.text for c in cand]
        cand_scores = [c.score for c in cand]
        cand_sources = [c.source for c in cand]
        bm25 = BM25Index(cand_texts)
        raw_bm25 = bm25.scores(q["query"])
        max_b = max(raw_bm25) if raw_bm25 else 0.0

        for a in ALPHAS:
            if max_b <= 0.0:
                # No BM25 signal — pure vector order (same as baseline).
                order = list(range(len(cand)))
            else:
                fused = [
                    (a * (bs / max_b) + (1 - a) * vs, src)
                    for bs, vs, src in zip(raw_bm25, cand_scores, cand_sources)
                ]
                fused.sort(key=lambda x: x[0], reverse=True)
                order = [src for _, src in fused]

            rank = _rank_of(order, rel)
            # Only top_k counts as recalled.
            if rank >= TOP_K:
                rank = -1
            ranks[a][qtype].append(rank)
            ranks[a]["all"].append(rank)

        if (idx + 1) % 200 == 0:
            print(f"  已处理 {idx + 1}/{len(queries)} 查询 ...")

    # Report
    print()
    print("=" * 64)
    print("  调参结果 (MRR / Recall@10)")
    print("=" * 64)
    header = f"{'alpha':>6} | {'all MRR':>8} | {'all R@10':>8} | {'exact MRR':>9} | {'fuzzy MRR':>9}"
    print(header)
    print("-" * len(header))

    summary = {}
    best_fuzzy_alpha = None
    best_fuzzy_mrr = -1.0
    best_all_alpha = None
    best_all_mrr = -1.0

    for a in ALPHAS:
        am = _mrr(ranks[a]["all"])
        ar = _recall(ranks[a]["all"])
        em = _mrr(ranks[a]["exact"])
        fm = _mrr(ranks[a]["fuzzy"])
        print(f"{a:>6.1f} | {am:>8.4f} | {ar:>8.4f} | {em:>9.4f} | {fm:>9.4f}")
        summary[str(a)] = {
            "all_mrr": round(am, 6),
            "all_recall": round(ar, 6),
            "exact_mrr": round(em, 6),
            "fuzzy_mrr": round(fm, 6),
        }
        if fm > best_fuzzy_mrr:
            best_fuzzy_mrr = fm
            best_fuzzy_alpha = a
        if am > best_all_mrr:
            best_all_mrr = am
            best_all_alpha = a

    print()
    print(f"  最优 alpha (全体 MRR):  {best_all_alpha}  (MRR={best_all_mrr:.4f})")
    print(f"  最优 alpha (模糊 MRR):  {best_fuzzy_alpha}  (MRR={best_fuzzy_mrr:.4f})")
    print(f"  耗时: {time.time() - t0:.1f}s")

    out = {
        "config": {
            "num_docs": NUM_DOCS,
            "num_queries": len(queries),
            "top_k": TOP_K,
            "candidates": CANDIDATES,
            "alphas": ALPHAS,
            "embedding_dim": dim,
            "used_fallback": used_fallback,
        },
        "results": summary,
        "best_all_alpha": best_all_alpha,
        "best_fuzzy_alpha": best_fuzzy_alpha,
    }
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"  结果已保存: {OUT_PATH}")


if __name__ == "__main__":
    main()
