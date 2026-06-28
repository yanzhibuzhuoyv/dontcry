# Anchor-Point Memory Rule

When working within the rag-system project, follow this memory workflow.

## 1. Session Start — Ask

At the start of every conversation, ask:

> [Anchor Memory] Enable session memory? (y/n)

If y/yes → enable. If n/no/skip → continue normally.

## 2. During Session — Note

If enabled, track: key topics, decisions, technical points, TODOs, user preferences.

## 3. Session End — Summarize

When the conversation concludes, run:

```python
from rag_system.session_memory import SessionMemory
from pathlib import Path

memory = SessionMemory(base_dir=Path("c:/Users/32202/Desktop/99999/rag-system"))
result = memory.end_session(
    content=conversation_text,
    slug="short-slug",
)
print(f"Prompts: {len(result['prompts'])}, merged: {result['merged_count']}")
```

This generates:
- `sessions/{date}-{slug}.md` — full session record
- `prompts/{date}-{slug}.prompts.txt` — session anchor prompts
- `prompts/{date}-{slug}.merged.txt` — merged + deduped with previous session
- Re-indexes `prompt_index/`

## 4. Next Session — Merge-Dedup

On next session end:
- Generate new prompts
- Merge with only the PREVIOUS session's merged.txt
- Older sessions are NOT re-merged
- New prompts first, deduped old ones appended

## 5. Recall — Search

When user asks to recall past topics:

**Step 1: Search prompt index**
```python
result = memory.recall("query keywords")
if result["found"]:
    # Show matching prompts + session files
```

**Step 2: Not found → tell user**
```
[Anchor Memory] No memory found for "xxx".
```

**Step 3: User insists → full-text search**
```python
deep = memory.recall_deep("query keywords")
# Show matching session snippets
```

**Step 4: If still not enough → RAG-index sessions/ and query**

## 6. Rollback

```python
versions = memory.list_prompt_versions()
memory.rollback_to("2026-06-28-xxx.merged.txt")
```

## Directory Layout

```
rag-system/
├── prompts/          # Anchor prompt files (*.prompts.txt, *.merged.txt)
├── sessions/         # Full session markdown (*.md)
├── prompt_index/     # Prompt RAG index (index.faiss + metadata.json)
└── .claude/rules/
    └── memory-anchor.md  # This rule
```

## Search Priority

1. RAG search prompt_index (prompt level)
2. If miss → tell user
3. User insists → grep sessions/*.md (full-text)
4. Optional: index sessions/ into main RAG for deeper search
