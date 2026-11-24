"""Two live example questions through the real chat pipeline for each local model, one run each, 2026-11-23. Routing,
retrieval and spelling are model-independent, so every model gets the same chunks; the script checks that it does."""
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import requests  # noqa: E402

from config import settings  # noqa: E402
from src.core import citations  # noqa: E402
from src.core.rag import RagPipeline  # noqa: E402

MODELS = ["llama3.2:3b", "qwen3:1.7b", "qwen3.5:2b"]
QUESTIONS = ["what is OLAP?", "how do I load dat into BigQuery?"]

pipeline = RagPipeline()
out, chunk_sets = [], {}
for model in MODELS:
    for q in QUESTIONS:
        start = time.perf_counter()
        result = pipeline.answer(q, model=model)
        answer = "".join(result.answer_stream)
        ids = [c.chunk_id for c in result.chunks]
        chunk_sets.setdefault(q, ids)
        assert chunk_sets[q] == ids, f"{model} {q}: different chunks"
        check = citations.check(answer, ids)
        out.append({"model": model, "question": q, "route": result.routing.path, "search_query": result.search_query,
                    "fallback": result.used_fallback, "chunks": [[c.chunk_id, round(c.score, 3)] for c in result.chunks],
                    "seconds": round(time.perf_counter() - start, 1), "citations": check.to_dict(), "answer": answer})
        print(f"{model:12} {q:36} {out[-1]['seconds']:6.1f}s valid cites {check.counts['valid']}", flush=True)
    requests.post(f"{settings.OLLAMA_BASE_URL}/api/generate", json={"model": model, "keep_alive": 0}, timeout=60)
(HERE / "live_examples.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
print("saved", len(out))
