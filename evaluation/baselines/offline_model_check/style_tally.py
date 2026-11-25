"""Answer-style gate tally (2026-11-24): word counts, cap hits, citations, fallback sentences and phrase checks for
baseline run files of the 110 frozen items, before and after the answer-style prompt change."""
import json, re, statistics, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT)); sys.stdout.reconfigure(encoding="utf-8")
from config import settings
from src.core import citations

FB = settings.FALLBACK_MESSAGE.split(".")[0]
# Phrases the author asked the answers to avoid (rule 1c), checked case-insensitively.
BANNED = {"based on the provided": r"based on the (provided|given)", "the excerpts": r"\bthe (provided )?excerpts?\b",
          "the text": r"\bthe text\b", "echoes the style rule": r"simple academic english|short steps",
          "the context": r"\bthe (provided )?context\b"}

def words(text):
    for note in (settings.ANSWER_CUT_NOTE, settings.ANSWER_REPEAT_NOTE):
        text = text.replace(note, "")
    return len(re.findall(r"\S+", text))

def tally(path):
    rs = json.loads(Path(path).read_text(encoding="utf-8"))["results"]
    qs = [r for r in rs if r["kind"] == "question"]
    out = {}
    for name, sub in (("questions", qs), ("probes", [r for r in rs if r["kind"] == "probe"])):
        chk = [citations.check(r["answer"], [c["chunk_id"] for c in r["retrieved"]]) for r in sub]
        w = [words(r["answer"]) for r in sub]
        out[name] = {"n": len(sub), "cited": sum(c.cites_course_material for c in chk),
                     "outside_set": [r["id"] for r, c in zip(sub, chk) if c.outside_set],
                     "contains_fallback": [r["id"] for r in sub if FB in r["answer"]],
                     "median_words": statistics.median(w), "p90_words": sorted(w)[int(0.9 * (len(w) - 1))], "max_words": max(w),
                     "in_150_250": sum(150 <= x <= 250 for x in w),
                     "hit_cap": [r["id"] for r in sub if r["stopped_by"] == "length_cap"],
                     "repetition": [r["id"] for r in sub if r["stopped_by"] == "repetition_guard"],
                     "banned": {k: [r["id"] for r in sub if re.search(p, r["answer"], re.I)] for k, p in BANNED.items()}}
    t = [r["seconds"] for r in rs if r["seconds"] is not None]
    out["median_s"] = round(statistics.median(t), 1)
    return out

if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(Path(p).name, json.dumps(tally(p), indent=1))
