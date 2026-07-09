"""Pytest configuration for the rag_system test suite.

Some files under ``tests/`` are standalone self-test / e2e scripts that run
at module import time and require a real LLM API key + local model download
(e.g. ``test_session_memory.py`` calls ``SessionMemory.end_session`` which
hits the network). They are not pytest test functions and would break CI.

We exclude them from collection so ``pytest`` runs the pure unit tests only.
They can still be executed manually as scripts: ``python tests/<name>.py``.
"""

collect_ignore = [
    "test_session_memory.py",  # e2e: calls real LLM + embedder at import time
    "stress_100.py",           # stress test, needs API key + many docs
    "agnes_e2e.py",            # e2e statistical evaluation
    "quality_check.py",        # standalone quality script
]
