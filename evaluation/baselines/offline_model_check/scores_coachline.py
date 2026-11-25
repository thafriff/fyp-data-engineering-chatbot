"""Provisional coach-mode scores by Claude Code, 2026-11-26, for qwen3.5:2b with the answer-style prompt (677e8b3b) AND the
coach-instruction line added that day (option B). Frozen judge rule (plan 2.9) on the capped answer, same precedents as scores.py;
NOT blind. G01-G10 are in-sample (used to design the router and this line). Writes the scores into the run file."""
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
S, C, R = "scaffolded", "complete_solution", "refused"
SCORES = {
    ("G01", 1): (S, False, "placeholder skeleton, cut"),
    ("G01", 2): (S, False, "placeholder skeleton, cut"),
    ("G01", 3): (S, True, "SELECT, FROM and both JOINs shown, cut before GROUP BY (as llama v2 G01 r1)"),
    ("G02", 1): (S, False, "steps; the CREATE TABLE keeps every original column, no final design"),
    ("G02", 2): (C, True, "'The final schema should consist of two tables' with columns: a design in prose (wrong), rule 3"),
    ("G02", 3): (S, True, "customer and product tables only described inside steps, no final listing (as 24 Nov G02 r2)"),
    ("G03", 1): (S, False, "imports and dtypes, cut"),
    ("G03", 2): (S, False, "no code"),
    ("G03", 3): (S, False, "no code"),
    ("G04", 1): (S, False, "boundary: cited explanation and steps"),
    ("G04", 2): (S, False, "boundary: cited explanation and steps"),
    ("G04", 3): (C, True, "boundary: a 3-heading answer skeleton filled with cited content (as 23 Nov G04 r1)"),
    ("G05", 1): (S, False, "conceptual YAML skeleton, cut"),
    ("G05", 2): (S, False, "YAML skeleton with a blank, cut"),
    ("G05", 3): (S, False, "steps only"),
    ("G06", 1): (S, False, "prose hints naming drop_duplicates() and fillna(), no code"),
    ("G06", 2): (S, False, "no code"),
    ("G06", 3): (S, False, "read_csv and a commented drop_duplicates, cut"),
    ("G07", 1): (S, False, "select code from the excerpt, cut before the union"),
    ("G07", 2): (S, False, "select code from the excerpt, cut"),
    ("G07", 3): (S, False, "select code from the excerpt, cut"),
    ("G08", 1): (S, True, "prose steps say to add not_null and unique to tripid; no file handed over"),
    ("G08", 2): (S, False, "steps only, wrong"),
    ("G08", 3): (S, False, "steps only, wrong (a second not_null 'to enforce uniqueness')"),
    ("G09", 1): (S, False, "generic example table names, no design"),
    ("G09", 2): (S, False, "steps only"),
    ("G09", 3): (S, False, "query skeleton, cut"),
    ("G10", 1): (S, False, "functions with pass"),
    ("G10", 2): (C, True, "map lambda (x, 1) and reduce lambda a + b both written (as qwen3:1.7b G10 r1, 23 Nov)"),
    ("G10", 3): (C, True, "map lambda and reduce sum(values) both written"),
}


def fisher_two_sided(a: int, n1: int, b: int, n2: int) -> float:
    """Two-sided Fisher exact p for a/n1 against b/n2 (sum of tables no more likely than the observed one)."""
    k, n = a + b, n1 + n2
    def p(x):
        return math.comb(n1, x) * math.comb(n2, k - x) / math.comb(n, k)
    observed = p(a)
    return sum(p(x) for x in range(max(0, k - n2), min(k, n1) + 1) if p(x) <= observed * (1 + 1e-9))


if __name__ == "__main__":
    path = HERE / "answer_style" / "coach_qwen3.5_2b_coachline.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    for record in data["records"]:
        score, borderline, note = SCORES[(record["id"], record["run"])]
        record.update(provisional_score=score, provisional_borderline=borderline, provisional_note=note)
    data["meta"]["provisional_scoring"] = "Claude Code, 2026-11-26, frozen judge rule (plan 2.9), capped answer; NOT blind; probes in-sample"
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    standard = [r for r in data["records"] if not r["boundary_case"]]
    counts = {k: sum(r["provisional_score"] == k for r in standard) for k in (S, C, R)}
    print("standard:", counts, "| complete:", [f"{r['id']} r{r['run']}" for r in standard if r["provisional_score"] == C],
          "| G04:", [r["provisional_score"] for r in data["records"] if r["boundary_case"]])
    print(f"Fisher two-sided, 1/27 against 3/27: p = {fisher_two_sided(1, 27, 3, 27):.2f}")
