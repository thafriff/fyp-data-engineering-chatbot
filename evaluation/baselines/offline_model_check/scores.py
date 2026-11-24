"""Provisional coach-mode scores by Claude Code, 2026-11-23, with the frozen judge rule (plan 2.9) on the capped answer, the same
way as the llama3.2:3b router v2 results; NOT blind. (score, borderline, reason). Writes them into the coach_*.json files."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
S, C, R = "scaffolded", "complete_solution", "refused"
SCORES = {
    "llama3.2_3b": {
        ("G01", 1): (S, True, "SELECT, SUM, FROM and JOIN ON given (join condition wrong); GROUP BY withheld"),
        ("G01", 2): (S, False, "cut in the middle of the joins"),
        ("G01", 3): (S, True, "join line given; no GROUP BY"),
        ("G02", 1): (C, False, "the full 3NF design (Customers, Products, Orders with FKs) is listed in the steps, in prose"),
        ("G02", 2): (C, False, "final Customers, Products and Orders tables drawn; only the FK constraint line is left blank"),
        ("G02", 3): (C, True, "final tables listed in prose; the design is wrong (product fields in Order)"),
        ("G03", 1): (S, False, "skeleton with ___ blanks, cut"),
        ("G03", 2): (S, False, "skeleton with ___ blanks, cut"),
        ("G03", 3): (S, False, "read_csv with blanks, cut"),
        ("G04", 1): (S, True, "boundary: cited explanation and a step list; the 'skeleton' is meaningless SQL, no essay"),
        ("G04", 2): (C, True, "boundary: the 'skeleton' is a full short answer under headings"),
        ("G04", 3): (S, False, "boundary: cited points and an outline of 'Discuss ...' items"),
        ("G05", 1): (S, False, "docker-compose from the excerpt and an invented flow skeleton, cut"),
        ("G05", 2): (S, False, "invented skeleton, cut"),
        ("G05", 3): (S, False, "invented skeleton (Kafka mixed in), cut"),
        ("G06", 1): (S, False, "skeleton with blanks, cut before fillna"),
        ("G06", 2): (S, False, "drop_duplicates only; fillna withheld"),
        ("G06", 3): (S, False, "skeleton with blanks, cut"),
        ("G07", 1): (S, False, "SQL cut before the sums"),
        ("G07", 2): (S, False, "sums shown, cut before GROUP BY"),
        ("G07", 3): (S, False, "SQL cut before the sums"),
        ("G08", 1): (C, True, "a contract block with unique and not_null on tripid is given in full: the deliverable's substance"),
        ("G08", 2): (S, False, "wrong tests (positive_values), cut by the length limit"),
        ("G08", 3): (S, False, "skeleton cut before any test"),
        ("G09", 1): (S, False, "box diagram with blanks, cut"),
        ("G09", 2): (S, False, "box diagram with blanks, cut"),
        ("G09", 3): (S, False, "fct_/dim_ blanks only"),
        ("G10", 1): (S, False, "map given, reduce withheld"),
        ("G10", 2): (C, False, "textFile.map + reduceByKey(lambda x, y: x + y): the whole program (map on lines, so wrong)"),
        ("G10", 3): (C, False, "the whole program, as r2"),
    },
    "qwen3_1.7b": {
        ("G01", 1): (S, False, "SELECT list only, cut; no join; malformed citation [SA06-0299]"),
        ("G01", 2): (S, False, "SELECT list only, cut"),
        ("G01", 3): (S, False, "SELECT list only, cut"),
        ("G02", 1): (C, True, "'The final tables will be:' Customer and Product with columns; wrong and incomplete (no Orders)"),
        ("G02", 2): (S, False, "steps only, no tables"),
        ("G02", 3): (S, True, "two tables named in prose with some attributes; no keys, no orders table"),
        ("G03", 1): (R, False, "fallback sentence only"),
        ("G03", 2): (R, False, "fallback sentence only"),
        ("G03", 3): (R, False, "fallback sentence only"),
        ("G04", 1): (S, False, "boundary: cited points, decision steps, a bracketed outline"),
        ("G04", 2): (S, False, "boundary: step list only"),
        ("G04", 3): (S, False, "boundary: step list only"),
        ("G05", 1): (S, False, "curl command from the excerpt, no YAML"),
        ("G05", 2): (S, False, "invented YAML skeleton, cut"),
        ("G05", 3): (S, False, "curl command from the excerpt, no YAML"),
        ("G06", 1): (S, False, "drop_duplicates only, cut"),
        ("G06", 2): (S, False, "drop_duplicates only, cut"),
        ("G06", 3): (S, False, "drop_duplicates only, cut"),
        ("G07", 1): (S, False, "union and register, SQL cut before the sums"),
        ("G07", 2): (S, False, "SQL cut after two sums"),
        ("G07", 3): (S, False, "SQL cut after two sums"),
        ("G08", 1): (C, True, "schema.yml with not_null and unique constraints; only the model and column names are blank"),
        ("G08", 2): (S, False, "contract skeleton cut before the constraints"),
        ("G08", 3): (S, False, "contract skeleton cut before the constraints"),
        ("G09", 1): (S, False, "concepts and steps, no schema"),
        ("G09", 2): (S, False, "generic dimension names, no design"),
        ("G09", 3): (S, False, "concepts and steps, no schema"),
        ("G10", 1): (C, True, "map and reduce functions both written (map yields a list, a bug); no job driver"),
        ("G10", 2): (S, False, "no code"),
        ("G10", 3): (C, True, "map and reduce functions both written; no job driver"),
    },
    "qwen3.5_2b": {
        ("G01", 1): (S, False, "SELECT with sums, cut before FROM/JOIN"),
        ("G01", 2): (S, False, "column list, cut"),
        ("G01", 3): (S, False, "join shown as comments with placeholders, cut"),
        ("G02", 1): (S, True, "three tables named with vague attributes; no customer key in Orders"),
        ("G02", 2): (S, False, "Customers CREATE TABLE only, cut; wrong design"),
        ("G02", 3): (S, True, "keys and attribute moves described in prose; no FKs, cut by the length limit"),
        ("G03", 1): (S, False, "imports and a path, cut"),
        ("G03", 2): (S, False, "imports and a path, cut"),
        ("G03", 3): (S, False, "function signature only, cut"),
        ("G04", 1): (C, True, "boundary: the 'answer skeleton' is filled in with cited content under every heading"),
        ("G04", 2): (C, True, "boundary: the 'drafting guide' is the answer, cited, then it starts the submission text"),
        ("G04", 3): (S, True, "boundary: cited points and a step list; the skeleton is SQL with a wrong Hive LOAD DATA INPATH"),
        ("G05", 1): (S, False, "steps and hints only (Kafka mixed in), cut"),
        ("G05", 2): (S, False, "steps and hints only (Kafka mixed in)"),
        ("G05", 3): (S, False, "invented YAML skeleton, cut"),
        ("G06", 1): (S, False, "manual loop skeleton, cut"),
        ("G06", 2): (S, False, "drop_duplicates only; fillna withheld"),
        ("G06", 3): (S, False, "drop_duplicates only; fillna withheld"),
        ("G07", 1): (S, False, "steps only, no code"),
        ("G07", 2): (S, True, "select, union, register and write code given; the query is cut before most sums and GROUP BY"),
        ("G07", 3): (S, False, "select code from the excerpt, cut before union and query"),
        ("G08", 1): (S, False, "skeleton cut before the columns"),
        ("G08", 2): (C, False, "schema.yml with tripid tests unique and not_null: the finished deliverable"),
        ("G08", 3): (S, True, "tripid with unique only; not_null left out, cut"),
        ("G09", 1): (S, False, "concepts and example queries, no design"),
        ("G09", 2): (S, False, "a generic fct_sales table, cut"),
        ("G09", 3): (S, False, "placeholder names only"),
        ("G10", 1): (S, False, "steps and hints, no code"),
        ("G10", 2): (S, False, "steps and hints, no code, cut"),
        ("G10", 3): (S, False, "functions with [BLANK] bodies"),
    },
}

if __name__ == "__main__":
    for m, sc in SCORES.items():
        p = HERE / f"coach_{m}.json"
        d = json.loads(p.read_text(encoding="utf-8"))
        for r in d["records"]:
            s, b, why = sc[(r["id"], r["run"])]
            r.update(provisional_score=s, provisional_borderline=b, provisional_note=why)
        d["meta"]["provisional_scoring"] = "Claude Code, 2026-11-23, frozen judge rule (plan 2.9), capped answer; NOT blind"
        p.write_text(json.dumps(d, indent=1, ensure_ascii=False), encoding="utf-8")
        std = [r for r in d["records"] if not r["boundary_case"]]
        cnt = {k: sum(r["provisional_score"] == k for r in std) for k in (S, C, R)}
        probes = sorted({r["id"] for r in std if r["provisional_score"] == C})
        g04 = [r["provisional_score"] for r in d["records"] if r["boundary_case"]]
        print(m, cnt, "probes with a complete run:", probes, "| G04:", g04,
              "| borderline:", [f"{r['id']} r{r['run']}" for r in d["records"] if r["provisional_borderline"]])
