"""Quality check: real documents + incremental update + anchor memory."""

import os, sys, tempfile, shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

with open(str(Path(__file__).resolve().parents[1] / ".env"), encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ[k.strip()] = v.strip()
os.environ["RAG_EMBEDDING_PROVIDER"] = "local"
os.environ["RAG_EMBEDDING_MODEL"] = "BAAI/bge-small-zh-v1.5"

PASS = FAIL = 0


def check(cond, label):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {label}")
    else:
        FAIL += 1
        print(f"  [FAIL] {label}")


# ============================================================
# Test A: Real document ingestion
# ============================================================
print("\n=== Test A: Real document ingestion ===")

from rag_system.documents import DocumentLoader

loader = DocumentLoader()
docs = []
for path in [
    "c:/Users/32202/Desktop/99999/手链.docx",
    "c:/Users/32202/Desktop/99999/绛珠子证恩仇.docx",
]:
    doc = loader.load(path)
    if doc:
        docs.append(doc)
        print(f"  [INFO] {Path(path).name}: {len(doc.content)} chars")

check(len(docs) == 2, f"loaded {len(docs)}/2 real docx files")
check(all(len(d.content) > 100 for d in docs), "documents have meaningful content")

# ============================================================
# Test B: Chinese text splitting
# ============================================================
print("\n=== Test B: Chinese text splitting ===")

from rag_system.splitter import TextSplitter

splitter = TextSplitter(chunk_size=256, chunk_overlap=64)
chunks = splitter.split_documents(docs)
check(len(chunks) >= 4, f"split into {len(chunks)} chunks")
check(all(10 <= len(c.text) <= 300 for c in chunks), "chunk sizes within bounds")

# ============================================================
# Test C: RAG ingest + query
# ============================================================
print("\n=== Test C: RAG ingest + query ===")

td = tempfile.mkdtemp()
os.environ["RAG_VECTOR_STORE_DIR"] = td + "/rag_index"

from rag_system.rag import RAGSystem
from rag_system.config import load_rag_config

rag = RAGSystem(load_rag_config())

r = rag.ingest("c:/Users/32202/Desktop/99999/手链.docx")
check(r["files"] >= 1, f"ingest: {r['files']} files, {r['chunks']} chunks")
check(r["chunks"] >= 2, f"chunks >= 2 ({r['chunks']})")

answer = rag.query("手链讲的是什么故事？", top_k=3)
check(len(answer) > 30, f"LLM answer ({len(answer)} chars)")

# ============================================================
# Test D: Incremental update
# ============================================================
print("\n=== Test D: Incremental update ===")

r2 = rag.ingest("c:/Users/32202/Desktop/99999/手链.docx")
check(r2["skipped"] >= 1, f"second ingest skips unchanged ({r2['skipped']})")

r3 = rag.ingest("c:/Users/32202/Desktop/99999/手链.docx", force=True)
check(r3.get("updated", 0) == 0, "force same hash: no update")

# ============================================================
# Test E: Multi-file index
# ============================================================
print("\n=== Test E: Multi-file index ===")

r4 = rag.ingest("c:/Users/32202/Desktop/99999/绛珠子证恩仇.docx")
check(r4["files"] >= 1, f"ingest second file: {r4['files']} files")

from rag_system.vector_store import VectorStore
store = VectorStore.load(td + "/rag_index")
check(store.count >= 4, f"total chunks: {store.count}")

# ============================================================
# Test F: Anchor memory
# ============================================================
print("\n=== Test F: Anchor memory ===")

from rag_system.session_memory import SessionMemory
memory = SessionMemory(base_dir=Path(td))

s1 = memory.end_session(
    "用户：RAG检索精度如何提升？\n助手：调整chunk_size到256-512，加入reranker，使用更好的嵌入模型。",
    slug="rag-optimization",
    session_date="2026-07-01",
)
check(len(s1["prompts"]) > 0, f"session 1: {len(s1['prompts'])} prompts")

s2 = memory.end_session(
    "用户：锚点记忆合并去重有bug吗？\n助手：已修复replace_source的向量重建，改用faiss.reconstruct。",
    slug="memory-fix",
    session_date="2026-07-02",
)
check(len(s2["prompts"]) > 0, f"session 2: {len(s2['prompts'])} prompts")
check(s2["merged_count"] >= len(s2["prompts"]), "merged with previous")

check(memory.recall("RAG检索精度")["found"], "recall 'RAG检索精度'")
check(memory.recall("合并去重")["found"], "recall '合并去重'")
check(memory.recall_deep("replace_source")["found"], "deep recall 'replace_source'")

# ============================================================
# Report
# ============================================================
print(f"\n{'='*50}")
print(f"  Results: {PASS + FAIL} tests, {PASS} passed, {FAIL} failed")
print(f"{'='*50}")

shutil.rmtree(td, ignore_errors=True)
if FAIL > 0:
    sys.exit(1)
