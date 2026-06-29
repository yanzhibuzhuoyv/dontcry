"""Self-test for the anchor-point session memory system.

Simulates 3 consecutive sessions and verifies the full pipeline:
  generate prompts → merge-dedup → save → recall → rollback.
"""

import os
import sys
from pathlib import Path

os.environ["RAG_LLM_API_KEY"] = "test-key-for-validation-only"
os.environ["RAG_EMBEDDING_PROVIDER"] = "local"
os.environ["RAG_EMBEDDING_MODEL"] = "BAAI/bge-small-zh-v1.5"

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_system.session_memory import (
    SessionMemory,
    _merge_dedup,
    _make_slug,
    _parse_session_filename,
)

BASE = Path(__file__).resolve().parents[1]
PASS = 0
FAIL = 0


def check(condition, label):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {label}")
    else:
        FAIL += 1
        print(f"  [FAIL] {label}")


# ============================================================
# Test 1: merge_dedup
# ============================================================
print("\n=== Test 1: merge_dedup ===")

new = ["锚点记忆法", "RAG检索", "提示词合并"]
old = ["RAG检索", "FAISS存储", "会话管理"]
merged = _merge_dedup(new, old)
check(merged == ["锚点记忆法", "RAG检索", "提示词合并", "FAISS存储", "会话管理"],
      "basic merge: new first, old deduped")
check("RAG检索" in merged and merged.index("RAG检索") == 1,
      "duplicate kept at new position")

# Whitespace handling
new2 = ["  锚点 记忆 法  ", "rag 检索"]
old2 = ["锚点记忆法", "会话管理"]
merged2 = _merge_dedup(new2, old2)
check(len(merged2) == 3, "whitespace-normalized dedup")


# ============================================================
# Test 2: slug  generation
# ============================================================
print("\n=== Test 2: slug generation ===")

slug = _make_slug("用户询问锚点记忆法的实现方案")
check(len(slug) <= 30, f"slug length <= 30")
check(not any(c in slug for c in " \t\n"), "no whitespace in slug")


# ============================================================
# Test 3: filename parsing
# ============================================================
print("\n=== Test 3: filename parsing ===")

d, s = _parse_session_filename("2026-06-28-anchor-memory-230416")
check(d == "2026-06-28" and "anchor-memory" in s, "standard parse with timestamp")


# ============================================================
# Test 4: 3-session lifecycle
# ============================================================
print("\n=== Test 4: Session lifecycle (3 sessions) ===")

memory = SessionMemory(base_dir=BASE)

# Session 1
c1 = """
用户: 我需要一个锚点记忆系统，能在每次会话结束后自动总结提示词。
助手: 可以设计 SessionMemory 模块，用 LLM 提取 5-10 个锚点提示词。
用户: 提示词怎么存储？怎么检索？
助手: 提示词写入 prompts/ 目录，通过 RAG 索引到 FAISS 向量库。
检索时分两级：先搜提示词索引，找不到再全文检索 sessions/*.md。
用户: 多次对话之间提示词怎么处理？
助手: 每次新对话生成新提示词，与上一次的提示词合并去重。
"""
r1 = memory.end_session(c1, slug="anchor-memory-design", session_date="2026-06-28")
check(len(r1["prompts"]) > 0, f"session 1: {len(r1['prompts'])} prompts")
check(Path(r1["session_file"]).exists(), "session 1 file exists")

# Session 2
c2 = """
用户: 上次说的锚点记忆法开始写代码吧。
助手: 核心模块 session_memory.py 包含 SessionMemory 类。
用户: merge_dedup 函数怎么设计？
助手: 用 set 去重，normalize 后比较。新提示词在前。
用户: 目录结构怎么规划？
助手: prompts/ 存提示词，sessions/ 存会话，prompt_index/ 存索引。
"""
r2 = memory.end_session(c2, slug="implementation-details", session_date="2026-06-29")
check(len(r2["prompts"]) > 0, f"session 2: {len(r2['prompts'])} prompts")
check(Path(r2["session_file"]).exists(), "session 2 file exists")

# Session 3
c3 = """
用户: 锚点记忆系统写完了，需要自测。
助手: 应该覆盖 merge_dedup、多会话合并、检索召回、回退等场景。
用户: 用真实对话文本模拟多个连续会话。
"""
r3 = memory.end_session(c3, slug="self-test-plan", session_date="2026-06-30")
check(len(r3["prompts"]) > 0, f"session 3: {len(r3['prompts'])} prompts")
check(Path(r3["session_file"]).exists(), "session 3 file exists")

merged3 = memory._load_previous_merged()
print(f"  [INFO] merged prompts ({len(merged3)}):")
for p in merged3:
    print(f"         - {p}")


# ============================================================
# Test 5: Recall — prompt index search
# ============================================================
print("\n=== Test 5: Recall — prompt index ===")

r = memory.recall("锚点记忆")
check(r["found"], f"recall '锚点记忆': found via {r['method']}")
if r["found"]:
    for item in r["results"]:
        print(f"  [INFO] {item['score']:.2f} {item['text'][:60]}")

r2 = memory.recall("量子计算")
check(not r2["found"], "recall '量子计算': not found")


# ============================================================
# Test 6: Deep recall — full-text search
# ============================================================
print("\n=== Test 6: Deep recall — full-text ===")

deep = memory.recall_deep("SessionMemory")
check(deep["found"], f"deep recall: {len(deep['results'])} results")


# ============================================================
# Test 7: Versions + rollback
# ============================================================
print("\n=== Test 7: Versions + rollback ===")

versions = memory.list_prompt_versions()
check(len(versions) >= 3, f"versions >= 3 ({len(versions)})")

s1 = next((v for v in versions if "anchor-memory-design" in v["file"]), None)
if s1:
    ok = memory.rollback_to(s1["file"])
    check(ok, f"rollback to {s1['file']}")
    check((memory.prompt_index_dir / "index.faiss").exists(),
          "index rebuilt after rollback")


# ============================================================
# Test 8: Session listing
# ============================================================
print("\n=== Test 8: Session listing ===")

sessions = memory.list_sessions()
check(len(sessions) >= 3, f"sessions >= 3 ({len(sessions)})")


# ============================================================
# Report
# ============================================================
print(f"\n{'='*50}")
print(f"  Total: {PASS + FAIL}   Passed: {PASS}   Failed: {FAIL}")
print(f"{'='*50}")

if FAIL > 0:
    sys.exit(1)
