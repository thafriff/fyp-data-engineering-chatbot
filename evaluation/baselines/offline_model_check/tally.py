"""Tally the offline-model check (2026-11-23). Reads the run files in this folder; prints one JSON block per model."""
import json, re, statistics, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT)); sys.stdout.reconfigure(encoding="utf-8")
from config import settings
from src.core import citations

FB = settings.FALLBACK_MESSAGE.split(".")[0]          # "I could not find this in the course materials"
REFUSAL = re.compile(r"I (?:could not|couldn't|cannot|can't) (?:find|help|provide|do|write|complete)|I'm (?:not able|unable)|I am (?:not able|unable)", re.I)

def p95(xs):
    xs = sorted(xs); k = 0.95 * (len(xs) - 1); f = int(k); c = min(f + 1, len(xs) - 1)
    return round(xs[f] + (xs[c] - xs[f]) * (k - f), 1)

def baseline(path):
    rs = json.loads(path.read_text(encoding="utf-8"))["results"]
    out = {"items": len(rs), "errors": sum(bool(r["error"]) for r in rs)}
    t = [r["seconds"] for r in rs if r["seconds"] is not None]
    out["median_s"], out["p95_s"], out["mean_s"] = round(statistics.median(t), 1), p95(t), round(statistics.mean(t), 1)
    for kind in ("question", "probe"):
        sub = [r for r in rs if r["kind"] == kind]
        chk = [citations.check(r["answer"], [c["chunk_id"] for c in r["retrieved"]]) for r in sub]
        out[kind] = {
            "n": len(sub),
            "with_valid_citation": sum(c.cites_course_material for c in chk),
            "uncited_ids": [r["id"] for r, c in zip(sub, chk) if not c.cites_course_material],
            "cites_outside_set": [r["id"] for r, c in zip(sub, chk) if c.outside_set],
            "malformed": [r["id"] for r, c in zip(sub, chk) if c.malformed],
            "fallback_only": [r["id"] for r in sub if r["answer"].strip().startswith(FB) and len(r["answer"]) < len(settings.FALLBACK_MESSAGE) + 40],
            "contains_fallback": [r["id"] for r in sub if FB in r["answer"]],
            "other_refusal_wording": [r["id"] for r in sub if FB not in r["answer"] and REFUSAL.search(r["answer"])],
            "stopped_by": {k: sum(r["stopped_by"] == k for r in sub) for k in ("finished", "repetition_guard", "length_cap")},
            "mean_answer_chars": round(statistics.mean(len(r["answer"]) for r in sub)),
        }
    return out

def coach(path):
    d = json.loads(path.read_text(encoding="utf-8")); rs = d["records"]
    t = [r["seconds"] for r in rs]
    return {"runs": len(rs), "routed_task_request": sum(r["route"] == "task_request" for r in rs),
            "median_s": round(statistics.median(t), 1), "p95_s": p95(t),
            "uncited_runs": sum(r["valid_citations"] == 0 for r in rs),
            "raw_with_code": sum(r["signals_raw"]["fenced_code_blocks"] > 0 for r in rs),
            "longest_raw_block": max(r["signals_raw"]["longest_code_block_lines"] for r in rs),
            "blocks_cut": sum(r["code_blocks_cut"] for r in rs), "runs_with_cut": sum(r["code_blocks_cut"] > 0 for r in rs),
            "contains_fallback": [f"{r['id']} r{r['run']}" for r in rs if r["signals_raw"]["contains_fallback_sentence"]],
            "think": d["meta"].get("ollama_think")}

if __name__ == "__main__":
    # *_rerun = each model re-run alone on the GPU (the clean latency figures; the first runs were contaminated)
    for m in ["llama3.2_3b", "llama3.2_3b_rerun", "qwen3_1.7b", "qwen3_1.7b_rerun", "qwen3.5_2b", "qwen3.5_2b_rerun"]:
        b = HERE / f"baseline_{m}.json"
        if b.exists():
            print(m, "BASELINE", json.dumps(baseline(b), indent=1))
        c = HERE / f"coach_{m}.json"
        if c.exists():
            print(m, "COACH", json.dumps(coach(c)))
