#!/usr/bin/env python3
"""Retrieval quality evaluation with a REAL embedder (bge-small-zh-v1.5).

Generates 500 synthetic Chinese tech docs, creates 1000 queries,
evaluates baseline / hybrid / rerank strategies, reports MRR.
"""

import hashlib
import json
import os
import random
import signal
import sys
import time
from pathlib import Path

import numpy as np

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# Add exchange dir to path for rag_system imports
EXCHANGE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXCHANGE_DIR))

from rag_system.bm25 import BM25Index
from rag_system.config import EmbeddingConfig
from rag_system.documents import Chunk
from rag_system.vector_store import SearchResult, VectorStore

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
NUM_DOCS = 500
TOP_K = 10
HYBRID_CANDIDATES = TOP_K * 4  # 40
HYBRID_ALPHA = 0.3
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
RERANKER_MODEL = "BAAI/bge-reranker-base"
RERANKER_TIMEOUT = 180  # seconds
RESULTS_PATH = Path(__file__).resolve().parent / "results_real.json"

# ---------------------------------------------------------------------------
# FakeEmbedder fallback
# ---------------------------------------------------------------------------


class FakeEmbedder:
    """Deterministic fake embedder — used only when real embedding fails."""

    def __init__(self, dim: int = 384):
        self.dim = dim
        self._rng = np.random.RandomState(42)

    @property
    def dimension(self) -> int:
        return self.dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        n = len(texts)
        vecs = self._rng.randn(n, self.dim).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return (vecs / norms).tolist()

    def embed_query(self, text: str) -> list[float]:
        seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.RandomState(seed)
        v = rng.randn(self.dim).astype(np.float32)
        v /= np.linalg.norm(v)
        return v.tolist()


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------
TECH_TOPICS = [
    "人工智能", "机器学习", "深度学习", "自然语言处理", "计算机视觉",
    "数据库", "分布式系统", "云计算", "微服务架构", "容器技术",
    "网络安全", "区块链", "物联网", "大数据", "数据挖掘",
    "推荐系统", "搜索引擎", "知识图谱", "图数据库", "消息队列",
    "缓存系统", "负载均衡", "API网关", "服务网格", "边缘计算",
    "神经网络", "强化学习", "迁移学习", "联邦学习", "对抗生成网络",
    "情感分析", "文本分类", "命名实体识别", "关系抽取", "机器翻译",
    "图像分割", "目标检测", "人脸识别", "姿态估计", "三维重建",
    "时间序列", "异常检测", "聚类分析", "回归分析", "降维算法",
    "排序算法", "数据结构", "设计模式", "性能优化", "自动化测试",
]

TITLE_TEMPLATES = [
    "{t1}在{t2}中的应用与挑战",
    "基于{t1}的{t2}系统设计",
    "{t1}技术原理与{t2}实践",
    "深入理解{t1}：从{t2}到实践",
    "{t1}最新进展与{t2}发展趋势",
    "大规模{t1}系统架构设计与优化",
    "基于{t1}的实时{t2}解决方案",
    "{t1}算法优化：{t2}视角",
    "面向{t1}的{t2}平台构建",
    "下一代{t1}：融合{t2}的创新实践",
]

BODY_TEMPLATES = [
    "本文详细介绍了{t1}的核心原理和关键技术。通过分析{t2}的最新研究成果，提出了一种创新的解决方案。实验结果表明，该方法在{t3}指标上取得了显著提升。",
    "随着{t1}技术的快速发展，{t2}领域也迎来了新的机遇和挑战。本文从系统架构角度出发，设计了一套基于{t1}的{t2}框架，有效解决了传统方法存在的问题。",
    "针对当前{t1}领域面临的计算效率和存储瓶颈问题，本文提出了一种结合{t2}技术的优化方案。经过大量实验验证，该方案在{t3}场景下表现优异。",
    "在实际{t1}系统部署过程中，我们遇到了{t2}相关的性能瓶颈。通过引入{t3}策略进行针对性优化，系统吞吐量提升了四倍以上，延迟降低了显著。",
    "本文综述了{t1}领域近年来的重要研究进展，重点分析了{t2}、{t3}等关键技术。在此基础上，预测了未来{t4}的发展方向和潜在应用场景。",
]

random.seed(42)


def make_documents(num_docs: int) -> list[dict]:
    docs: list[dict] = []
    for i in range(num_docs):
        t1 = random.choice(TECH_TOPICS)
        t2 = random.choice(TECH_TOPICS)
        t3 = random.choice(TECH_TOPICS)
        t4 = random.choice(TECH_TOPICS)

        title_tpl = random.choice(TITLE_TEMPLATES)
        title = title_tpl.format(t1=t1, t2=t2)
        if len(title) > 20:
            title = title[:20]
        if len(title) < 10:
            title = t1[:4] + "技术" + title

        body_tpl = random.choice(BODY_TEMPLATES)
        body = body_tpl.format(t1=t1, t2=t2, t3=t3, t4=t4)
        if len(body) < 50:
            body += "该方法在多个数据集上进行了测试，所有指标均达到了预期水平。"
        if len(body) > 100:
            body = body[:100]

        full = title + "。" + body
        docs.append({"id": f"doc_{i}", "title": title, "body": body, "content": full})
    return docs


# ---------------------------------------------------------------------------
# Query generation
# ---------------------------------------------------------------------------
def make_queries(documents: list[dict]) -> list[dict]:
    queries: list[dict] = []
    for doc in documents:
        title = doc["title"]
        q1_len = random.randint(8, min(15, len(title)))
        q1 = title[:q1_len]

        body = doc["body"]
        max_start = max(0, len(body) - 20)
        start = random.randint(0, max_start)
        q2_len = random.randint(10, min(20, len(body) - start))
        q2 = body[start : start + q2_len]

        queries.append({"query": q1, "relevant_source": doc["id"], "type": "exact"})
        queries.append({"query": q2, "relevant_source": doc["id"], "type": "fuzzy"})
    return queries


# ---------------------------------------------------------------------------
# Embedder creation with fallback
# ---------------------------------------------------------------------------


def _try_create_embedder() -> tuple:
    """Try to create a real embedder; return (embedder, used_fallback_bool)."""
    from rag_system.embeddings import create_embedder

    config = EmbeddingConfig(provider="local", model=EMBEDDING_MODEL, device="cpu")
    try:
        embedder = create_embedder(config)
        # Trigger lazy load
        dim = embedder.dimension
        print(f"  真实 embedder 加载成功 (dim={dim})")
        return embedder, False
    except Exception as exc:
        print(f"  真实 embedder 加载失败: {exc}")
        print(f"  回退到 FakeEmbedder (dim=384)")
        fallback = FakeEmbedder(dim=384)
        return fallback, True


# ---------------------------------------------------------------------------
# Reranker creation with timeout
# ---------------------------------------------------------------------------


def _try_create_reranker() -> tuple:
    """Try to create a real reranker; return (reranker, success_bool)."""
    from rag_system.reranker import Reranker

    class TimeoutError(Exception):
        pass

    def _signal_handler(signum, frame):
        raise TimeoutError(f"reranker download took >{RERANKER_TIMEOUT}s")

    # Set alarm (Unix only; Windows fallback: manual time check)
    old_handler = None
    try:
        if hasattr(signal, "SIGALRM"):
            old_handler = signal.signal(signal.SIGALRM, _signal_handler)
            signal.alarm(RERANKER_TIMEOUT)

        t0 = time.time()
        r = Reranker(RERANKER_MODEL, device="cpu")
        test_q = "测试"
        test_c = [SearchResult(text="测试文档内容", score=0.5, source="test", metadata={})]
        _ = r.rerank(test_q, test_c, 1)
        elapsed = time.time() - t0
        print(f"  重排序模型加载成功 ({elapsed:.1f}s)")
        return r, True

    except Exception as exc:
        elapsed = time.time() - t0 if "t0" in dir() else 0
        print(f"  重排序模型加载失败 ({elapsed:.1f}s): {exc}")
        return None, False

    finally:
        if hasattr(signal, "SIGALRM") and old_handler is not None:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)


# ---------------------------------------------------------------------------
# Retrieval helpers
# ---------------------------------------------------------------------------
def _rank_of(results: list[SearchResult], source: str) -> int:
    for i, r in enumerate(results):
        if r.source == source:
            return i
    return -1


def baseline(embedder, store: VectorStore, query: str, top_k: int) -> list[SearchResult]:
    qvec = embedder.embed_query(query)
    return store.search(qvec, k=top_k)


def hybrid_fuse(
    embedder,
    store: VectorStore,
    query: str,
    candidates_count: int,
    alpha: float,
) -> list[SearchResult]:
    qvec = embedder.embed_query(query)
    cand = store.search(qvec, k=candidates_count)
    if not cand:
        return []

    bm25 = BM25Index([c.text for c in cand])
    raw_bm25 = bm25.scores(query)
    max_b = max(raw_bm25) if raw_bm25 else 0.0

    if max_b <= 0.0:
        return cand

    fused: list[SearchResult] = []
    for r, bs in zip(cand, raw_bm25):
        score = alpha * (bs / max_b) + (1 - alpha) * r.score
        fused.append(SearchResult(text=r.text, score=float(score), source=r.source, metadata=r.metadata))
    fused.sort(key=lambda x: x.score, reverse=True)
    return fused


def hybrid(
    embedder,
    store: VectorStore,
    query: str,
    top_k: int,
    candidates_count: int,
    alpha: float,
) -> list[SearchResult]:
    return hybrid_fuse(embedder, store, query, candidates_count, alpha)[:top_k]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    t_start = time.time()

    print("=" * 60)
    print("  检索质量统计测试实验 (真实 Embedder)")
    print("=" * 60)
    print(f"  文档数: {NUM_DOCS}, 查询数: {NUM_DOCS * 2}, top_k: {TOP_K}")
    print(f"  hybrid_alpha: {HYBRID_ALPHA}, 模型: {EMBEDDING_MODEL}")
    print(f"  HF_ENDPOINT: {os.environ.get('HF_ENDPOINT', '(default)')}")
    print()

    # 1. Generate synthetic documents
    print("[1/5] 生成合成文档 ...")
    docs = make_documents(NUM_DOCS)
    print(f"  -> {len(docs)} 篇文档")
    print(f"  -> 示例标题: {docs[0]['title']}")
    print()

    # 2. Create real embedder
    print("[2/5] 初始化 Embedder (下载可能需要 1-2 分钟) ...")
    embedder, used_fallback = _try_create_embedder()
    if used_fallback:
        print("  *** 警告: 使用了 FakeEmbedder 回退，结果不具有实际意义 ***")
    EMBEDDING_DIM = embedder.dimension
    print()

    # 3. Build FAISS vector store
    print("[3/5] 构建向量索引 ...")
    store = VectorStore(dimension=EMBEDDING_DIM)
    texts = [d["content"] for d in docs]
    t0 = time.time()
    embeddings = embedder.embed_documents(texts)
    print(f"  文档嵌入完成 ({len(embeddings)} vectors, {time.time() - t0:.1f}s)")

    chunks = [Chunk(text=d["content"], source=d["id"], chunk_index=0, metadata={}) for d in docs]
    store.add_documents(chunks, embeddings)
    print(f"  向量库包含 {store.count} 个向量 (dim={EMBEDDING_DIM})")
    print()

    # 4. Generate queries
    print("[4/5] 生成查询 ...")
    queries = make_queries(docs)
    n_exact = sum(1 for q in queries if q["type"] == "exact")
    n_fuzzy = sum(1 for q in queries if q["type"] == "fuzzy")
    print(f"  -> {len(queries)} 个查询 (exact={n_exact}, fuzzy={n_fuzzy})")
    print(f"  -> 精确查询: {queries[0]['query']!r}")
    print(f"  -> 模糊查询: {queries[1]['query']!r}")
    print()

    # 5. Try to load cross-encoder reranker
    print("[5/5] 尝试加载重排序模型 ...")
    print(f"    模型: {RERANKER_MODEL} (~1.1GB, 超时 {RERANKER_TIMEOUT}s)")
    strategies = ["baseline", "hybrid"]
    reranker = None
    reranker_ok = False
    if not used_fallback:
        reranker, reranker_ok = _try_create_reranker()
        if reranker_ok:
            strategies.append("rerank")
        else:
            print("  跳过 rerank 策略，只比较 baseline 和 hybrid")
    else:
        print("  使用 FakeEmbedder 回退，跳过 reranker（结果不可靠）")
    print()

    # 6. Run all strategies
    print("=" * 60)
    print("  执行检索策略 ...")
    print("=" * 60)
    per_query: list[dict] = []

    for idx, q in enumerate(queries):
        entry: dict = {
            "qid": idx,
            "query": q["query"],
            "query_type": q["type"],
            "relevant_source": q["relevant_source"],
        }

        # baseline
        bl = baseline(embedder, store, q["query"], TOP_K)
        entry["baseline_rank"] = _rank_of(bl, q["relevant_source"])

        # hybrid
        hy = hybrid(embedder, store, q["query"], TOP_K, HYBRID_CANDIDATES, HYBRID_ALPHA)
        entry["hybrid_rank"] = _rank_of(hy, q["relevant_source"])

        # rerank (if loaded)
        if reranker is not None:
            fused_cand = hybrid_fuse(embedder, store, q["query"], HYBRID_CANDIDATES, HYBRID_ALPHA)
            reranked = reranker.rerank(q["query"], fused_cand, TOP_K)
            entry["rerank_rank"] = _rank_of(reranked, q["relevant_source"])

        per_query.append(entry)

        if (idx + 1) % 200 == 0:
            print(f"    已处理 {idx + 1}/{len(queries)} 个查询 ...")

    print(f"    全部 {len(queries)} 个查询处理完成")
    print()

    # 7. Compute MRR
    print("=" * 60)
    print("  MRR (Mean Reciprocal Rank) 结果")
    print("=" * 60)
    mrr_results: dict[str, float] = {}
    for s in strategies:
        ranks = [e[f"{s}_rank"] for e in per_query]
        rr_sum = sum(1.0 / (r + 1) for r in ranks if r >= 0)
        mrr = rr_sum / len(ranks)
        mrr_results[s] = round(mrr, 6)

        recall = sum(1 for r in ranks if r >= 0) / len(ranks)

        exact_rr = sum(
            1.0 / (e[f"{s}_rank"] + 1)
            for e in per_query
            if e["query_type"] == "exact" and e[f"{s}_rank"] >= 0
        )
        exact_mrr = exact_rr / n_exact
        fuzzy_rr = sum(
            1.0 / (e[f"{s}_rank"] + 1)
            for e in per_query
            if e["query_type"] == "fuzzy" and e[f"{s}_rank"] >= 0
        )
        fuzzy_mrr = fuzzy_rr / n_fuzzy

        print(f"  [{s}]")
        print(f"    MRR (全体)  : {mrr:.6f}")
        print(f"    MRR (精确)  : {exact_mrr:.6f}")
        print(f"    MRR (模糊)  : {fuzzy_mrr:.6f}")
        print(f"    Recall@10   : {recall:.6f}")
        print()

    # 8. Save results_real.json
    output = {
        "config": {
            "num_docs": NUM_DOCS,
            "num_queries": len(queries),
            "top_k": TOP_K,
            "hybrid_alpha": HYBRID_ALPHA,
            "hybrid_candidates": HYBRID_CANDIDATES,
            "embedding_model": EMBEDDING_MODEL,
            "embedding_dim": EMBEDDING_DIM,
            "used_fallback": used_fallback,
            "strategies": strategies,
            "reranker_model": RERANKER_MODEL if reranker_ok else None,
        },
        "per_query": [
            {
                "qid": e["qid"],
                "query": e["query"],
                "query_type": e["query_type"],
                "relevant_source": e["relevant_source"],
                **{f"{s}_rank": e[f"{s}_rank"] for s in strategies},
            }
            for e in per_query
        ],
        "mrr": mrr_results,
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - t_start
    print(f"  结果已保存至: {RESULTS_PATH}")
    print(f"  总耗时: {elapsed:.2f} 秒")
    print()

    # 9. Verify output
    print("-" * 60)
    print("  格式验证")
    print("-" * 60)
    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert len(data["per_query"]) == len(queries), (
        f"per_query 数量不匹配: 期望 {len(queries)}, 实际 {len(data['per_query'])}"
    )
    for s in strategies:
        assert f"{s}_rank" in data["per_query"][0], f"缺少策略字段: {s}_rank"
        assert s in data["mrr"], f"mrr 中缺少策略: {s}"
    assert set(data["config"].keys()) >= {
        "num_docs", "num_queries", "top_k", "hybrid_alpha",
        "embedding_model", "embedding_dim", "strategies",
    }
    print(f"  格式验证通过: {len(data['per_query'])} 条 per_query 记录")
    print(f"  所有策略字段齐全: {strategies}")
    print(f"  所有 MRR 指标存在: {list(data['mrr'].keys())}")


if __name__ == "__main__":
    main()
