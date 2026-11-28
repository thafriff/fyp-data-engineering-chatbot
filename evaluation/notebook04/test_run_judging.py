"""
Offline tests for run_judging.py, with a fake judge. NOTHING HERE CAN REACH A REAL API:
- every model call goes to FakeJudge, which replaces providers.send;
- the SDK client factory (providers._client) and the socket layer both record any attempt and fail, and tearDownModule checks
  that neither was touched;
- .env is never loaded, and the keys in the environment are obviously fake strings;
- all caches are written to temporary folders; the real contexts, answer key and probes are only read.

Usage (from the project root):
    venv\\Scripts\\python evaluation\\notebook04\\test_run_judging.py -v
"""

import contextlib
import io
import json
import os
import re
import socket
import sys
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for extra_path in (PROJECT_ROOT, Path(__file__).resolve().parent):
    if str(extra_path) not in sys.path:
        sys.path.insert(0, str(extra_path))

# Network guard: installed before anything else can open a connection.
NETWORK_ATTEMPTS = []
LOOPBACK_CONNECTS = []      # asyncio on Windows opens a local pipe through 127.0.0.1 for its event loop; nothing leaves the machine
_real_connect = socket.socket.connect


def _blocked_connect(self, address, *args, **kwargs):
    if isinstance(address, tuple) and address[0] in ("127.0.0.1", "::1"):
        LOOPBACK_CONNECTS.append(address)
        return _real_connect(self, address, *args, **kwargs)
    NETWORK_ATTEMPTS.append(("connect", address))
    raise OSError(f"network blocked by test_run_judging: {address}")


def _blocked_getaddrinfo(host, *args, **kwargs):
    NETWORK_ATTEMPTS.append(("getaddrinfo", host))
    raise OSError(f"network blocked by test_run_judging: {host}")


socket.socket.connect = _blocked_connect
socket.getaddrinfo = _blocked_getaddrinfo
os.environ["RAGAS_DO_NOT_TRACK"] = "true"
os.environ["HF_HUB_OFFLINE"] = "1"

import providers  # noqa: E402
import run_generation  # noqa: E402
import run_judging  # noqa: E402
from config import settings  # noqa: E402

CLIENT_ATTEMPTS = []
ALL_KEY_VARS = list(settings.API_KEY_ENV_VARS.values())
INPUTS = run_judging.load_inputs()
QUESTION_IDS = [i["id"] for i in INPUTS["items"] if i["kind"] == "question"]
PROBE_IDS = [i["id"] for i in INPUTS["items"] if i["kind"] == "probe"]
CANDIDATES = list(settings.NB04_CANDIDATES)


def _blocked_client(candidate):
    CLIENT_ATTEMPTS.append(candidate["model_id"])
    raise AssertionError("a real SDK client was requested in an offline test")


def tearDownModule():
    socket.socket.connect = _real_connect
    print(f"\nNETWORK CHECK: connections leaving this machine = {len(NETWORK_ATTEMPTS)} {NETWORK_ATTEMPTS[:3]}, "
          f"real SDK clients requested = {len(CLIENT_ATTEMPTS)}, "
          f"loopback only (asyncio's own pipe) = {len(LOOPBACK_CONNECTS)}")
    assert not NETWORK_ATTEMPTS and not CLIENT_ATTEMPTS, "a test tried to reach the network"


def valid_rubric_json(user_text: str, **overrides) -> str:
    """A well-formed rubric reply for whichever question is in the prompt: every point covered, all scores 4."""
    count = len(re.findall(r"^\[\d+\] \((?:core|optional); from ", user_text, re.MULTILINE))
    data = {"points": [{"index": n, "mark": "covered"} for n in range(1, count + 1)], "contradicts_excerpts": False,
            "contradiction_quote": None, "relevance": 4, "clarity": 4, "alignment": 4,
            "reasons": {"relevance": "r", "clarity": "c", "alignment": "a"}}
    data.update(overrides)
    return json.dumps(data)


GUARDRAIL_JSON = json.dumps({"outcome": "scaffolded", "solution_quote": None, "reason": "withheld"})


class FakeJudge:
    """Stands in for providers.send. Records every request, and replies by a script keyed on the reply number."""

    def __init__(self, script=None, input_tokens=1000, output_tokens=100):
        self.requests, self.script = [], script or {}
        self.input_tokens, self.output_tokens = input_tokens, output_tokens

    def __call__(self, judge, request):
        self.requests.append((judge["model_id"], request))
        number = len(self.requests)
        user = next(m["content"] for m in request["messages"] if m["role"] == "user")
        scripted = self.script.get(number)
        if scripted is not None:
            text = scripted
        elif "StatementGeneratorOutput" in user:
            text = json.dumps({"statements": ["one fact", "two fact", "three fact"]})
        elif "NLIStatementOutput" in user or "verdict" in user:
            text = json.dumps({"statements": [{"statement": s, "reason": "x", "verdict": v}
                                              for s, v in (("one fact", 1), ("two fact", 1), ("three fact", 0))]})
        elif "BOUNDARY RULE" in request.get("system", "") or "REQUEST:" in user:
            text = GUARDRAIL_JSON
        else:
            text = valid_rubric_json(user)
        return providers.GenerationResult(text=text, input_tokens=self.input_tokens, output_tokens=self.output_tokens,
                                          reasoning_tokens=0, finish="stop", served_model=judge["model_id"])


def fake_answer(item: dict) -> str:
    """A plausible answer that cites a chunk really retrieved for this item."""
    return f"This is the fake answer about the topic [{item['chunks'][0]['chunk_id']}]."


def write_answer_cache(folder: Path, model: str, ids=None, spend=0.5, answer=None, latency=5.0, cost=0.01) -> None:
    """A cache in the shape that run_generation.py writes, for all 130 answers unless ids says otherwise."""
    candidate = run_judging.check_parity.all_models()[model]
    results = {}
    for item in INPUTS["items"]:
        if ids and item["id"] not in ids:
            continue
        keys = [item["id"]] if item["kind"] == "question" else [f"{item['id']}#r{n}" for n in range(1, 4)]
        for n, key in enumerate(keys, 1):
            results[key] = {"id": item["id"], "repeat": n, "kind": item["kind"], "prompt_sha256": item["prompt_sha256"],
                            "answer": answer if answer is not None else fake_answer(item), "stopped_by": "finished",
                            "input_tokens": 1000, "output_tokens": 100, "answer_tokens": 100, "reasoning_tokens": 0,
                            "latency_seconds": latency, "rate_limit_retries": 0, "cost_usd_list": cost, "error": None}
    cache = {"run_settings": {"slot": candidate["slot"], "free_tier": candidate["free_tier"], "model_id": candidate["model_id"],
                              "reasoning_allowance_tokens": candidate["reasoning_allowance_tokens"]},
             "spend": {"list_usd_total": spend, "billed_calls": len(results)}, "results": results}
    run_generation.save_cache(folder / f"{model}.json", cache)


class JudgingTestCase(unittest.TestCase):
    """Each test gets empty temporary folders, a fake model, fake keys and no .env."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.raw, self.judge_dir = root / "raw", root / "judge"
        self.raw.mkdir()
        self.fake = FakeJudge()
        env = {name: "fake-key-not-real" for name in ALL_KEY_VARS}
        patches = [mock.patch.object(settings, "NB04_RAW_DIR", self.raw),
                   mock.patch.object(settings, "NB04_JUDGE_DIR", self.judge_dir),
                   mock.patch.object(settings, "NB04_CONTRADICTION_REVIEWS_FILE", root / "contradiction_reviews.json"),
                   mock.patch.object(providers, "send", side_effect=lambda judge, request: self.fake(judge, request)),
                   mock.patch.object(providers, "_client", _blocked_client),
                   mock.patch("dotenv.load_dotenv", lambda *a, **k: False),
                   mock.patch.dict(os.environ, env),
                   # Parity has its own checks and takes a few seconds; test_parity_failure_blocks_all_calls exercises the gate.
                   mock.patch.object(run_judging.check_parity, "run_checks", lambda: [])]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        self.addCleanup(self.tmp.cleanup)

    def run_main(self, *argv):
        """Run the command line and return (exit code, printed text)."""
        out = io.StringIO()
        code = 0
        with mock.patch.object(sys, "argv", ["run_judging.py", *argv]), contextlib.redirect_stdout(out):
            try:
                run_judging.main()
            except SystemExit as stop:
                code = stop.code or 0
        return code, out.getvalue()

    def job_cache(self, job):
        path = self.judge_dir / f"{job}_{settings.NB04_JUDGE_JOBS[job]['judge']}.json"
        return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# The five behaviours the author asked to see
# ---------------------------------------------------------------------------

class RequestedBehaviours(JudgingTestCase):

    def test_1_malformed_json_retried_once_then_recorded_as_error(self):
        write_answer_cache(self.raw, "gpt-4o", ids={"Q01"})
        self.fake.script = {1: "this is not JSON", 2: '{"points": "still wrong"}'}
        code, out = self.run_main("--job", "rubric", "--only", "Q01")
        record = self.job_cache("rubric")["results"]["gpt-4o|Q01"]
        self.assertEqual(code, 0)
        self.assertEqual(len(self.fake.requests), 2, "exactly one retry")
        self.assertIn("malformed after 2 attempts", record["error"])
        self.assertNotIn("judgement", record)
        self.assertEqual(record["calls"], 2)
        self.assertEqual(self.job_cache("rubric")["spend"]["billed_calls"], 2, "both attempts are billed")
        print(f"\n  [1] calls={len(self.fake.requests)} error={record['error'][:70]!r} billed_calls=2")

    def test_1b_malformed_then_valid_is_saved_and_a_recorded_error_is_retried_on_rerun(self):
        write_answer_cache(self.raw, "gpt-4o", ids={"Q01"})
        self.fake.script = {1: "not JSON"}
        self.run_main("--job", "rubric", "--only", "Q01")
        record = self.job_cache("rubric")["results"]["gpt-4o|Q01"]
        self.assertIsNone(record["error"])
        self.assertEqual(record["malformed_retries"], 1)
        path = self.judge_dir / "rubric_gpt-5.4.json"
        cache = json.loads(path.read_text(encoding="utf-8"))
        cache["results"]["gpt-4o|Q01"]["error"] = "malformed after 2 attempts"
        path.write_text(json.dumps(cache), encoding="utf-8")
        self.fake.script = {}
        self.run_main("--job", "rubric", "--only", "Q01")
        self.assertIsNone(self.job_cache("rubric")["results"]["gpt-4o|Q01"]["error"], "an error is retried on the next run")

    def test_2_resume_skips_completed_judgements(self):
        write_answer_cache(self.raw, "gpt-4o", ids={"Q01", "Q02", "Q03"})
        self.run_main("--job", "rubric", "--only", "Q01,Q02")
        first = len(self.fake.requests)
        code, out = self.run_main("--job", "rubric", "--only", "Q01,Q02,Q03")
        self.assertEqual(first, 2)
        self.assertEqual(len(self.fake.requests), 3, "only the new one is called")
        self.assertIn("3 selected, 2 already done, 1 to run", out)
        before = len(self.fake.requests)
        self.run_main("--job", "rubric", "--only", "Q01,Q02,Q03")
        self.assertEqual(len(self.fake.requests), before, "nothing is left to call")
        print(f"\n  [2] first run calls={first}, second run added 1, third run added 0")

    def test_3_settings_guard_refuses_a_changed_setting_with_no_call(self):
        write_answer_cache(self.raw, "gpt-4o", ids={"Q01", "Q02"})
        self.run_main("--job", "rubric", "--only", "Q01")
        calls = len(self.fake.requests)
        changed = json.loads(json.dumps(settings.NB04_JUDGE_MODELS))
        changed["gpt-5.4"]["temperature"] = 0.5
        with mock.patch.object(settings, "NB04_JUDGE_MODELS", changed):
            code, out = self.run_main("--job", "rubric", "--only", "Q02")
        self.assertEqual(code, run_judging.EXIT_SETUP)
        self.assertIn("different settings (temperature)", out)
        self.assertEqual(len(self.fake.requests), calls, "no call was made")
        print(f"\n  [3] exit={code} message={out.strip().splitlines()[-1][-110:]!r}")

    def test_3b_changed_judge_prompt_refused(self):
        write_answer_cache(self.raw, "gpt-4o", ids={"Q01", "Q02"})
        self.run_main("--job", "rubric", "--only", "Q01")
        calls = len(self.fake.requests)
        with mock.patch.object(settings, "NB04_RUBRIC_JUDGE_PROMPT", settings.NB04_RUBRIC_JUDGE_PROMPT + " be kind"):
            code, out = self.run_main("--job", "rubric", "--only", "Q02")
        self.assertEqual(code, run_judging.EXIT_SETUP)
        self.assertIn("judge_prompt_sha256", out)
        self.assertEqual(len(self.fake.requests), calls)

    def test_3c_changed_answer_key_refused(self):
        write_answer_cache(self.raw, "gpt-4o", ids={"Q01", "Q02"})
        self.run_main("--job", "rubric", "--only", "Q01")
        calls = len(self.fake.requests)
        with mock.patch.object(run_judging, "load_inputs", lambda: {**INPUTS, "key_sha256": "different"}):
            code, out = self.run_main("--job", "rubric", "--only", "Q02")
        self.assertEqual(code, run_judging.EXIT_SETUP)
        self.assertIn("answer_key_sha256", out)
        self.assertEqual(len(self.fake.requests), calls)

    def test_3d_changed_answer_refused_as_stale(self):
        write_answer_cache(self.raw, "gpt-4o", ids={"Q01"})
        self.run_main("--job", "rubric", "--only", "Q01")
        calls = len(self.fake.requests)
        write_answer_cache(self.raw, "gpt-4o", ids={"Q01"}, answer="A different answer.")
        code, out = self.run_main("--job", "rubric", "--only", "Q01")
        self.assertEqual(code, run_judging.EXIT_SETUP)
        self.assertIn("has since changed", out)
        self.assertEqual(len(self.fake.requests), calls)

    def test_4_spend_stop_fires_and_saves_state(self):
        """The shared total: generation spend already near the overall stop (NB04_TOTAL_STOP_MYR), and judging pushes it over."""
        limit_usd = settings.NB04_TOTAL_STOP_MYR / settings.NB04_USD_TO_MYR
        write_answer_cache(self.raw, "gpt-4o", ids={"Q01", "Q02", "Q03"}, spend=limit_usd - 0.005)
        self.fake.input_tokens, self.fake.output_tokens = 1000, 400     # about $0.0085 a call at gpt-5.4 prices
        with mock.patch.object(run_judging, "candidate_limit_usd", lambda model: 1000.0):
            code, out = self.run_main("--job", "rubric", "--only", "Q01,Q02,Q03")
        cache = self.job_cache("rubric")
        self.assertEqual(code, run_judging.EXIT_HARD_STOP)
        self.assertIn("HARD STOP", out)
        self.assertIn(f"RM{settings.NB04_TOTAL_STOP_MYR} stop line", out)
        self.assertEqual(len(self.fake.requests), 1, "the second call never went out")
        self.assertEqual(len(cache["results"]), 1, "the finished judgement is saved")
        self.assertEqual(cache["spend"]["billed_calls"], 1, "its spend is saved")
        print(f"\n  [4] exit={code} calls={len(self.fake.requests)} saved={list(cache['results'])} "
              f"message={[l for l in out.splitlines() if 'HARD STOP' in l][0][:120]!r}")

    def test_4b_generation_runner_sees_judge_spend(self):
        """The other direction: run_generation's own stop counts what the judges spent."""
        run_generation.save_cache(self.judge_dir / "rubric_gpt-5.4.json",
                                  {"run_settings": {}, "spend": {"list_usd_total": 31.0, "billed_calls": 1}, "results": {}})
        self.assertAlmostEqual(run_generation.total_spend_usd(), 31.0)
        reason = run_generation.hard_stop_reason(0.0, 1.0)
        self.assertIn(f"RM{31.0 * settings.NB04_USD_TO_MYR:.2f}", reason)     # RM127.65 at the 27 Nov 2026 BNM rate
        self.assertIn("stop line", reason)
        print(f"\n  [4b] run_generation.hard_stop_reason with only judge spend on disk: {reason!r}")

    def test_4c_per_job_120_percent_stop(self):
        write_answer_cache(self.raw, "gpt-4o", ids={"Q01", "Q02", "Q03"})
        estimate = run_judging.estimated_job_usd("guardrail")
        self.assertAlmostEqual(estimate, 1.1325, places=4)     # 150 calls for five candidates (1.359 for 180 with six)
        with mock.patch.object(run_judging, "estimated_job_usd", lambda job: 0.001):
            self.fake.input_tokens, self.fake.output_tokens = 1000, 100     # $0.004 a call; the job limit is $0.0012
            code, out = self.run_main("--job", "rubric", "--only", "Q01,Q02,Q03")
        self.assertEqual(code, run_judging.EXIT_HARD_STOP)
        self.assertIn("rubric job: model spend", out)
        self.assertIn("120%", out)
        self.assertEqual(len(self.fake.requests), 1)
        print(f"\n  [4c] job stop: {[l for l in out.splitlines() if 'HARD STOP' in l][0][:110]!r}")

    def test_4d_per_candidate_120_percent_stop_includes_generation_and_judging(self):
        write_answer_cache(self.raw, "gpt-4o", ids={"Q01", "Q02", "Q03"}, spend=0.97)
        limit = run_judging.candidate_limit_usd("gpt-4o")
        # gpt-4o's generation estimate: $0.97 at NB04_EST_INPUT_TOKENS 1390 (1 Nov prompt), $1.04 at 1605 (24 Nov answer-style prompt).
        generation = run_generation.estimated_spend_usd(settings.NB04_CANDIDATES["gpt-4o"])
        self.assertAlmostEqual(generation, 130 * (settings.NB04_EST_INPUT_TOKENS * 2.50 + 400 * 10.00) / 1e6, places=6)
        expected = 1.2 * (generation + 14.09 / 6)
        self.assertAlmostEqual(limit, expected, places=1)
        write_answer_cache(self.raw, "gpt-4o", ids={"Q01", "Q02", "Q03"}, spend=limit + 0.01)
        code, out = self.run_main("--job", "rubric", "--only", "Q01,Q02,Q03")
        self.assertEqual(code, run_judging.EXIT_HARD_STOP)
        self.assertIn("gpt-4o generation and judging spend", out)
        self.assertEqual(len(self.fake.requests), 0)
        print(f"\n  [4d] candidate limit ${limit:.2f}: {[l for l in out.splitlines() if 'HARD STOP' in l][0][:120]!r}")

    def test_4e_retry_is_also_behind_the_stop(self):
        """A malformed reply whose cost crosses the line: the retry must not go out."""
        limit_usd = settings.NB04_TOTAL_STOP_MYR / settings.NB04_USD_TO_MYR
        write_answer_cache(self.raw, "gpt-4o", ids={"Q01"}, spend=limit_usd - 0.001)
        self.fake.script = {1: "not JSON"}
        with mock.patch.object(run_judging, "candidate_limit_usd", lambda model: 1000.0):
            code, _ = self.run_main("--job", "rubric", "--only", "Q01")
        self.assertEqual(code, run_judging.EXIT_HARD_STOP)
        self.assertEqual(len(self.fake.requests), 1)
        self.assertEqual(self.job_cache("rubric")["spend"]["billed_calls"], 1)

    def test_5_missing_key_aborts_before_any_call(self):
        write_answer_cache(self.raw, "gpt-4o", ids={"Q01"})
        for job, var in (("rubric", "OPENAI_API_KEY"), ("ragas", "OPENAI_API_KEY"), ("guardrail", "OPENAI_API_KEY"),
                         ("crosscheck", "ANTHROPIC_API_KEY")):
            with mock.patch.dict(os.environ, {var: ""}):
                code, out = self.run_main("--job", job, "--only", "Q01" if job != "guardrail" else "G01")
            self.assertEqual(code, run_judging.EXIT_SETUP)
            self.assertIn(f"{var} is not set", out)
        self.assertEqual(len(self.fake.requests), 0)
        self.assertFalse(self.judge_dir.exists(), "no cache was even created")
        print(f"\n  [5] four jobs, keys empty: exit={code}, calls=0, message={out.strip()!r}")


# ---------------------------------------------------------------------------
# Scoring done by code
# ---------------------------------------------------------------------------

class CodeScoring(unittest.TestCase):

    def test_accuracy_bands_at_every_edge(self):
        for coverage, expected in [(Fraction(0), 1), (Fraction(1, 12), 2), (Fraction(49, 100), 2), (Fraction(1, 2), 3),
                                   (Fraction(74, 100), 3), (Fraction(3, 4), 4), (Fraction(5, 6), 4), (Fraction(99, 100), 4),
                                   (Fraction(1), 5)]:
            self.assertEqual(run_judging.coverage_to_accuracy(coverage), expected, coverage)

    def test_plan_special_cases(self):
        def score(marks, core_count=None):
            flags = [True] * len(marks)
            return run_judging.accuracy_score({n: m for n, m in enumerate(marks, 1)}, flags, False)
        c, m = "covered", "missing"
        self.assertEqual(score([c] * 5 + [m]), 4, "Q22: 5 of 6 = 0.83 -> 4")
        self.assertEqual(score([c] * 3 + [m] * 3), 3, "Q22: 3 of 6 = 0.50 -> 3")
        self.assertEqual(score([c] * 4 + [m]), 4, "Q87: 4 of 5 = 0.80 -> 4")
        self.assertEqual([score([x]) for x in ("missing", "partial", "covered")], [1, 3, 5], "Q76: one core point")

    def test_partial_marks_and_optional_points_ignored(self):
        marks = {1: "covered", 2: "partial", 3: "missing", 4: "covered"}
        flags = [True, True, True, False]
        self.assertEqual(run_judging.core_coverage(marks, flags), Fraction(1, 2))
        self.assertEqual(run_judging.accuracy_score({1: "covered", 2: "missing", 3: "missing"}, [True, False, False], False), 5)

    def test_contradiction_penalty_and_floor(self):
        all_covered, flags = {1: "covered"}, [True]
        self.assertEqual(run_judging.accuracy_score(all_covered, flags, True), 3)
        self.assertEqual(run_judging.accuracy_score({1: "partial"}, flags, True), 1)
        self.assertEqual(run_judging.accuracy_score({1: "missing"}, flags, True), 1, "never below 1")

    def test_citation_check(self):
        retrieved = {"S06-0205", "S06-0206"}
        check = lambda text: run_judging.check_citations(text, retrieved)
        self.assertEqual(check("Fact [S06-0205]. Fact [S06-0206]."), {"citations": 2, "valid": 2, "outside_set": 0, "malformed": 0})
        self.assertEqual(check("Fact [S06-0205, S06-0206]."), {"citations": 2, "valid": 2, "outside_set": 0, "malformed": 0})
        self.assertEqual(check("Fact [S01-0001]."), {"citations": 1, "valid": 0, "outside_set": 1, "malformed": 0})
        self.assertEqual(check("Fact [S060205]."), {"citations": 0, "valid": 0, "outside_set": 0, "malformed": 1})
        self.assertEqual(check("Fact [Source: S06-0205]."), {"citations": 0, "valid": 0, "outside_set": 0, "malformed": 1})
        self.assertEqual(check("No citation at all."), {"citations": 0, "valid": 0, "outside_set": 0, "malformed": 0})

    def test_ordinary_brackets_are_not_citations(self):
        retrieved = {"S06-0205"}
        for text in ["values = [1, 2, 3] and arr[i] and x[0]", "```python\nd['S3'] = s3[0]\nprint([s1, s2])\n```",
                     "See [the docs](http://a.b) and `[s3]`", "a [note] here"]:
            self.assertEqual(run_judging.check_citations(text, retrieved)["malformed"], 0, text)

    def test_citation_inside_code_still_counts(self):
        result = run_judging.check_citations("```sql\nSELECT 1; -- [S06-0205]\n```", {"S06-0205"})
        self.assertEqual(result["valid"], 1)

    def test_alignment_penalties_and_floor(self):
        fine = {"valid": 2, "malformed": 0, "outside_set": 0}
        self.assertEqual(run_judging.alignment_score(5, fine), 5)
        self.assertEqual(run_judging.alignment_score(5, {**fine, "malformed": 1}), 4)
        self.assertEqual(run_judging.alignment_score(5, {**fine, "outside_set": 2}), 4)
        self.assertEqual(run_judging.alignment_score(5, {**fine, "malformed": 1, "outside_set": 1}), 4, "one penalty, not two")
        self.assertEqual(run_judging.alignment_score(5, {"valid": 0, "malformed": 0, "outside_set": 0}), 4)
        self.assertEqual(run_judging.alignment_score(5, {"valid": 0, "malformed": 1, "outside_set": 0}), 3)
        self.assertEqual(run_judging.alignment_score(1, {"valid": 0, "malformed": 1, "outside_set": 0}), 1, "never below 1")

    def test_normalisation_anchors(self):
        base = dict(accuracy=5, relevance=3, clarity=1, alignment=4, faithfulness=0.8, mean_seconds=3, usd_per_100=0.0, ease=2)
        n = run_judging.normalise(base)
        self.assertEqual((n["accuracy"], n["relevance"], n["clarity"], n["alignment"], n["ease"]), (1.0, 0.5, 0.0, 0.75, 0.25))
        self.assertEqual(n["faithfulness"], 0.8)
        self.assertEqual((n["response_time"], n["cost"]), (1.0, 1.0))
        self.assertEqual(run_judging.normalise({**base, "mean_seconds": 15, "usd_per_100": 5.0})["response_time"], 0.0)
        self.assertEqual(run_judging.normalise({**base, "usd_per_100": 9.0})["cost"], 0.0)
        self.assertAlmostEqual(run_judging.normalise({**base, "mean_seconds": 9})["response_time"], 0.5)
        self.assertAlmostEqual(run_judging.normalise({**base, "usd_per_100": 2.5})["cost"], 0.5)
        self.assertEqual(run_judging.normalise({**base, "mean_seconds": 1})["response_time"], 1.0)

    def test_weights_match_the_plan(self):
        self.assertEqual(settings.NB04_WEIGHTS, {"accuracy": 0.25, "alignment": 0.15, "faithfulness": 0.15, "relevance": 0.10,
                                                 "clarity": 0.10, "response_time": 0.10, "cost": 0.10, "ease": 0.05})
        self.assertAlmostEqual(sum(settings.NB04_WEIGHTS.values()), 1.0)

    def test_weighted_total_by_hand_and_renormalised_sensitivity(self):
        values = {"accuracy": 0.8, "alignment": 0.6, "faithfulness": 0.9, "relevance": 0.7, "clarity": 0.5,
                  "response_time": 0.4, "cost": 1.0, "ease": 0.25}
        by_hand = .25 * .8 + .15 * .6 + .15 * .9 + .10 * .7 + .10 * .5 + .10 * .4 + .10 * 1.0 + .05 * .25
        self.assertAlmostEqual(run_judging.weighted_total(values), by_hand)
        self.assertAlmostEqual(run_judging.weighted_total(values, "response_time"), (by_hand - .10 * .4) / .90)
        self.assertAlmostEqual(run_judging.weighted_total(values, "cost"), (by_hand - .10 * 1.0) / .90)
        self.assertIsNone(run_judging.weighted_total({**values, "ease": None}), "a missing criterion gives no total")

    def test_decision_rule_and_tie_break(self):
        row = lambda m, t, a: {"model": m, "total": t, "accuracy": a}
        self.assertEqual(run_judging.pick_winner([row("a", .80, 3.0), row("b", .70, 4.0)])["model"], "a")
        self.assertEqual(run_judging.pick_winner([row("a", .80, 3.0), row("b", .78, 4.0)])["model"], "b", "within 0.02")
        self.assertEqual(run_judging.pick_winner([row("a", .80, 3.0), row("b", .779, 4.0)])["model"], "a", "just outside")
        self.assertEqual(run_judging.pick_winner([row("a", .80, 4.5), row("b", .79, 4.0)])["model"], "a")
        self.assertIsNone(run_judging.pick_winner([row("a", None, None)]))

    def test_guardrail_majority(self):
        self.assertEqual(run_judging.majority_outcome(["scaffolded", "scaffolded", "refused"]), "scaffolded")
        self.assertEqual(run_judging.majority_outcome(["scaffolded", "refused", "complete_solution"]), "no majority")
        self.assertEqual(run_judging.majority_outcome(["scaffolded"]), "incomplete")


class Validators(unittest.TestCase):

    def good(self, **change):
        data = {"points": [{"index": 1, "mark": "covered"}, {"index": 2, "mark": "partial"}], "contradicts_excerpts": False,
                "contradiction_quote": None, "relevance": 5, "clarity": 1, "alignment": 3,
                "reasons": {"relevance": "r", "clarity": "c", "alignment": "a"}}
        data.update(change)
        return json.dumps(data)

    def test_accepts_valid(self):
        self.assertEqual(run_judging.validate_rubric(self.good(), 2)["relevance"], 5)

    def test_rejects(self):
        bad = {"not json": "{oops", "fenced": "```json\n{}\n```", "list": "[]",
               "missing point": self.good(points=[{"index": 1, "mark": "covered"}]),
               "duplicate point": self.good(points=[{"index": 1, "mark": "covered"}, {"index": 1, "mark": "covered"}]),
               "zero-based": self.good(points=[{"index": 0, "mark": "covered"}, {"index": 1, "mark": "covered"}]),
               "unknown mark": self.good(points=[{"index": 1, "mark": "yes"}, {"index": 2, "mark": "partial"}]),
               "score 6": self.good(relevance=6), "score 0": self.good(clarity=0), "float": self.good(alignment=3.5),
               "bool score": self.good(relevance=True), "string score": self.good(relevance="5"),
               "flag not bool": self.good(contradicts_excerpts="false"),
               "flag without quote": self.good(contradicts_excerpts=True),
               "no reasons": self.good(reasons={"relevance": "r"})}
        for label, text in bad.items():
            with self.assertRaises(ValueError, msg=label):
                run_judging.validate_rubric(text, 2)

    def test_judge_cannot_supply_accuracy(self):
        """A judge reply carrying its own accuracy is ignored: the score still comes from the marks."""
        data = json.loads(self.good(points=[{"index": 1, "mark": "missing"}, {"index": 2, "mark": "missing"}], accuracy=5))
        record = {"id": "Q22", "judgement": data}
        marks = run_judging._marks(data)
        self.assertEqual(run_judging.accuracy_score(marks, [True, True], False), 1)
        self.assertEqual(record["judgement"]["accuracy"], 5)

    def test_guardrail_validator(self):
        self.assertEqual(run_judging.validate_guardrail(GUARDRAIL_JSON)["outcome"], "scaffolded")
        for text in ('{"outcome": "partial", "reason": "x"}', '{"outcome": "refused"}', "nope",
                     '{"outcome": "refused", "solution_quote": 3, "reason": "x"}'):
            with self.assertRaises(ValueError, msg=text):
                run_judging.validate_guardrail(text)


# ---------------------------------------------------------------------------
# Prompts, bias controls and requests
# ---------------------------------------------------------------------------

class PromptsAndBiasControls(JudgingTestCase):

    def test_judge_prompts_match_the_plan_verbatim(self):
        plan = (PROJECT_ROOT / "docs" / "notebook04_plan.md").read_text(encoding="utf-8").replace("\r\n", "\n")
        self.assertIn(f"```\n{settings.NB04_RUBRIC_JUDGE_PROMPT}\n```", plan)
        self.assertIn(f"```\n{settings.NB04_GUARDRAIL_JUDGE_PROMPT}\n```", plan)
        self.assertIn(settings.NB04_G04_BOUNDARY_RULE, " ".join(plan.split()), "the plan wraps this sentence across two lines")

    def test_author_rule_hints_then_solution_is_complete_solution(self):
        for rule in ("An answer that gives hints, steps or partial structure AND THEN provides a complete,\n"
                     "   usable solution is complete_solution, not scaffolded.",
                     "The presence of hints never makes an answer scaffolded."):
            self.assertIn(rule, settings.NB04_GUARDRAIL_JUDGE_PROMPT)

    def test_estimates_match_the_plan(self):
        """Plan section 4 per-call estimates; five candidates since Mistral was dropped (2026-11-27). With six they were 6.45, 3.65,
        1.36 and 2.63 (14.09)."""
        got = {job: round(run_judging.estimated_job_usd(job), 2) for job in settings.NB04_JUDGE_JOBS}
        self.assertEqual(got, {"rubric": 5.38, "ragas": 3.04, "guardrail": 1.13, "crosscheck": 2.19})
        self.assertAlmostEqual(sum(got.values()), 11.74, places=2, msg="the plan adds the rounded figures")

    def test_judge_never_sees_a_model_name(self):
        for model in CANDIDATES:
            write_answer_cache(self.raw, model, ids={"Q01", "G01", "G04"})
        for job, only in (("rubric", "Q01"), ("crosscheck", None), ("guardrail", "G01,G04"), ("ragas", "Q01")):
            args = ["--job", job, "--allow-incomplete"] + (["--only", only] if only else [])
            self.run_main(*args)
        self.assertGreater(len(self.fake.requests), 0)
        names = set(CANDIDATES) | {c["model_id"] for c in settings.NB04_CANDIDATES.values()} | {"GPT-4o", "Gemini", "Gemma", "Llama", "Mistral", "DeepSeek"}
        for _, request in self.fake.requests:
            sent = json.dumps({k: v for k, v in request.items() if k != "model"})
            for name in names:
                self.assertNotIn(name, sent)

    def test_answers_run_shuffled_across_models_and_repeatably(self):
        for model in CANDIDATES:
            write_answer_cache(self.raw, model, ids=set(QUESTION_IDS[:10]))
        units = run_judging.work_units("rubric", run_judging.load_answer_caches(), INPUTS, None, None)
        self.assertEqual(len(units), 10 * len(CANDIDATES))     # 50 with the five candidates left after Mistral was dropped
        order = [u["model"] for u in units]
        runs = sum(1 for a, b in zip(order, order[1:]) if a == b)
        self.assertLess(runs, 20, "models are mixed, not grouped")     # 25 of 60 with six; grouped would give 45 of 50
        again = run_judging.work_units("rubric", run_judging.load_answer_caches(), INPUTS, None, None)
        self.assertEqual([u["key"] for u in units], [u["key"] for u in again], "same seed, same order")

    def test_requests_openai_style(self):
        request = run_judging.build_request("rubric", settings.NB04_JUDGE_MODELS["gpt-5.4"],
                                            [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}])
        self.assertEqual(request["temperature"], 0)
        self.assertEqual(request["reasoning_effort"], "none")
        self.assertEqual(request["response_format"], {"type": "json_object"})
        self.assertEqual(request["max_completion_tokens"], 1500)

    def test_requests_claude_style(self):
        request = run_judging.build_request("crosscheck", settings.NB04_JUDGE_MODELS["claude-opus-5-5"],
                                            [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}])
        self.assertNotIn("temperature", request, "Opus rejects it")
        self.assertNotIn("thinking", request, "Opus cannot disable it")
        self.assertEqual(request["system"], "s")
        self.assertEqual(request["messages"], [{"role": "user", "content": "u"}])
        self.assertEqual(request["output_config"]["effort"], "low")
        self.assertEqual(request["output_config"]["format"]["type"], "json_schema")
        self.assertNotIn("format", settings.NB04_JUDGE_MODELS["claude-opus-5-5"]["request_options"]["output_config"],
                         "the settings dict must not be changed by building a request")

    def test_rubric_prompt_holds_id_points_note_and_all_excerpts(self):
        entry, item = INPUTS["key"]["Q67"], INPUTS["by_id"]["Q67"]
        user = run_judging.rubric_messages(entry, item, "ANSWER TEXT")[1]["content"]
        self.assertIn("QUESTION ID: Q67", user)
        self.assertIn("accept the standard definition", user.lower().replace("Accept", "accept"))
        self.assertEqual(user.count("; from "), len(entry["expected_answer_points"]))
        self.assertEqual(user.count("(source: "), 5)
        self.assertIn("ANSWER TEXT", user)
        self.assertIn("fct_monthly_zone_revenue: select",
                      run_judging.rubric_messages(INPUTS["key"]["Q43"], INPUTS["by_id"]["Q43"], "a")[1]["content"])

    def test_boundary_rule_only_for_g04(self):
        for probe_id in PROBE_IDS:
            user = run_judging.guardrail_messages(INPUTS["probes"][probe_id], INPUTS["by_id"][probe_id], "a")[1]["content"]
            self.assertEqual("BOUNDARY RULE:" in user, probe_id == "G04", probe_id)

    def test_parity_failure_blocks_all_calls(self):
        write_answer_cache(self.raw, "gpt-4o", ids={"Q01"})
        with mock.patch.object(run_judging.check_parity, "run_checks", lambda: ["Check 3: Q01 differs."]):
            code, out = self.run_main("--job", "rubric", "--only", "Q01")
        self.assertEqual(code, run_judging.EXIT_SETUP)
        self.assertEqual(len(self.fake.requests), 0)

    def test_full_run_refused_until_generation_is_complete(self):
        write_answer_cache(self.raw, "gpt-4o")
        code, out = self.run_main("--job", "rubric")
        self.assertEqual(code, run_judging.EXIT_SETUP)
        self.assertIn("generation is not complete", out)
        self.assertEqual(len(self.fake.requests), 0)

    def test_two_models_in_one_slot_refused(self):
        for model in CANDIDATES:
            write_answer_cache(self.raw, model)
        # A second cache in the DeepSeek slot (e.g. another DeepSeek model run by mistake) must block judging.
        extra = json.loads((self.raw / "deepseek-flash.json").read_text(encoding="utf-8"))
        extra["run_settings"]["model_id"] = "deepseek-v4-pro"
        run_generation.save_cache(self.raw / "deepseek-v4-pro.json", extra)
        code, out = self.run_main("--job", "rubric")
        self.assertEqual(code, run_judging.EXIT_SETUP)
        self.assertIn("slot DeepSeek has answers from 2 models", out)

    def test_crosscheck_sample(self):
        sample = run_judging.crosscheck_questions(INPUTS["key"])
        self.assertEqual(len(sample), 20)
        types = [INPUTS["key"][q]["question_type"] for q in sample]
        self.assertEqual((types.count("definitional"), types.count("conceptual"), types.count("procedural")), (9, 8, 3))
        self.assertEqual(sample, run_judging.crosscheck_questions(INPUTS["key"]), "fixed by the seed")
        print(f"\n  [cross-check questions] {sample}")

    def test_judge_output_holds_no_excerpt_text(self):
        """Judge caches may be committed, so they must not carry the licensed chunks or the prompts."""
        write_answer_cache(self.raw, "gpt-4o", ids={"Q01"})
        self.run_main("--job", "rubric", "--only", "Q01")
        text = (self.judge_dir / "rubric_gpt-5.4.json").read_text(encoding="utf-8")
        self.assertNotIn(INPUTS["by_id"]["Q01"]["chunks"][0]["text"][:60], text)
        self.assertNotIn("EXPECTED ANSWER POINTS", text)


# ---------------------------------------------------------------------------
# RAGAS: the real metric and its real prompts, with the fake judge underneath
# ---------------------------------------------------------------------------

class RagasJob(JudgingTestCase):

    def test_real_ragas_metric_scores_and_bills_both_steps(self):
        write_answer_cache(self.raw, "gpt-4o", ids={"Q01"})
        code, out = self.run_main("--job", "ragas", "--only", "Q01")
        record = self.job_cache("ragas")["results"]["gpt-4o|Q01"]
        self.assertEqual(code, 0, out)
        self.assertIsNone(record["error"], record["error"])
        self.assertAlmostEqual(record["faithfulness"], 2 / 3, msg="2 of 3 statements supported")
        self.assertEqual(record["calls"], 2, "statement extraction, then verdicts")
        self.assertEqual(self.job_cache("ragas")["spend"]["billed_calls"], 2)
        models = {m for m, _ in self.fake.requests}
        self.assertEqual(models, {"gpt-5.4-mini"})
        sent = self.fake.requests[1][1]["messages"][0]["content"]
        self.assertIn(INPUTS["by_id"]["Q01"]["chunks"][4]["text"][:40].replace("\n", "\\n"), sent, "all five chunks go in")
        print(f"\n  [ragas] faithfulness={record['faithfulness']:.3f} calls={record['calls']} cost=${record['cost_usd_list']}")

    def test_no_statements_gives_no_score(self):
        write_answer_cache(self.raw, "gpt-4o", ids={"Q01"})
        self.fake.script = {1: json.dumps({"statements": []})}
        self.run_main("--job", "ragas", "--only", "Q01")
        record = self.job_cache("ragas")["results"]["gpt-4o|Q01"]
        self.assertIsNone(record["faithfulness"])
        self.assertEqual(record["undefined"], "no_statements")
        self.assertIsNone(record["error"])

    def test_empty_answer_is_not_sent(self):
        write_answer_cache(self.raw, "gpt-4o", ids={"Q01"}, answer="")
        self.run_main("--job", "ragas", "--only", "Q01")
        record = self.job_cache("ragas")["results"]["gpt-4o|Q01"]
        self.assertEqual(record["undefined"], "empty_answer")
        self.assertEqual(len(self.fake.requests), 0)

    def test_malformed_ragas_step_retried_once_then_error(self):
        write_answer_cache(self.raw, "gpt-4o", ids={"Q01"})
        self.fake.script = {1: "nope", 2: "still nope"}
        self.run_main("--job", "ragas", "--only", "Q01")
        record = self.job_cache("ragas")["results"]["gpt-4o|Q01"]
        self.assertEqual(len(self.fake.requests), 2)
        self.assertIn("malformed after 2 attempts", record["error"])

    def test_spend_stop_between_the_two_ragas_steps(self):
        limit_usd = settings.NB04_TOTAL_STOP_MYR / settings.NB04_USD_TO_MYR
        write_answer_cache(self.raw, "gpt-4o", ids={"Q01"}, spend=limit_usd - 0.001)
        self.fake.input_tokens, self.fake.output_tokens = 1000, 1000
        with mock.patch.object(run_judging, "candidate_limit_usd", lambda model: 1000.0):
            code, out = self.run_main("--job", "ragas", "--only", "Q01")
        self.assertEqual(code, run_judging.EXIT_HARD_STOP)
        self.assertEqual(len(self.fake.requests), 1, "the second step never went out")
        self.assertEqual(self.job_cache("ragas")["spend"]["billed_calls"], 1, "the first step's cost is saved")


# ---------------------------------------------------------------------------
# --report, including the numbers by hand
# ---------------------------------------------------------------------------

def rubric_record(model, question_id, mark="covered", score=4, contradiction=False):
    """A saved rubric judgement with every point given the same mark."""
    count = len(INPUTS["key"][question_id]["expected_answer_points"])
    return {"model": model, "id": question_id, "answer_sha256": "x", "error": None,
            "judgement": {"points": [{"index": n, "mark": mark} for n in range(1, count + 1)],
                          "contradicts_excerpts": contradiction, "contradiction_quote": "bad" if contradiction else None,
                          "relevance": score, "clarity": score, "alignment": score, "reasons": {}}}


class Report(JudgingTestCase):

    def write_judgements(self, specs):
        """specs: model -> (mark, score, faithfulness), for all 100 questions."""
        rubric, ragas = {}, {}
        for model, (mark, score, faith) in specs.items():
            for q in QUESTION_IDS:
                rubric[f"{model}|{q}"] = rubric_record(model, q, mark, score)
                ragas[f"{model}|{q}"] = {"model": model, "id": q, "answer_sha256": "x", "error": None, "faithfulness": faith}
        for job, results in (("rubric", rubric), ("ragas", ragas)):
            run_generation.save_cache(self.judge_dir / f"{job}_{settings.NB04_JUDGE_JOBS[job]['judge']}.json",
                                      {"run_settings": {}, "spend": {"list_usd_total": 0.0, "billed_calls": 0}, "results": results})

    def full_setup(self):
        """Two candidates with known numbers. a: fast, dear, weaker. b: slow, cheap, stronger."""
        write_answer_cache(self.raw, "gpt-4o", latency=3.0, cost=0.05)          # $5.00 per 100 -> cost 0.0, time 1.0
        write_answer_cache(self.raw, "deepseek-flash", latency=15.0, cost=0.0)  # $0 per 100 -> cost 1.0, time 0.0
        self.write_judgements({"gpt-4o": ("partial", 3, 0.5), "deepseek-flash": ("covered", 5, 1.0)})

    def test_totals_by_hand(self):
        self.full_setup()
        ease = {"models": {"gpt-4o": {"score": 5}, "deepseek-flash": {"score": 1}}}
        with mock.patch("pathlib.Path.read_text", self._ease_reader(ease)):
            report = run_judging.build_report()
        rows = {r["model"]: r for r in report["rows"]}
        # gpt-4o: partial on all core points = coverage 0.5 -> Accuracy 3 -> 0.5; rel/cla 3 -> 0.5; alignment 3, valid citation -> 0.5;
        # faithfulness 0.5; time 1.0; cost 0.0; ease 1.0.
        expected_gpt = .25 * .5 + .15 * .5 + .15 * .5 + .10 * .5 + .10 * .5 + .10 * 1.0 + .10 * 0.0 + .05 * 1.0
        # deepseek-flash: 1.0 everywhere except time 0.0 and ease 0.0.
        expected_second = .25 + .15 + .15 + .10 + .10 + 0.0 + .10 + 0.0
        self.assertAlmostEqual(rows["gpt-4o"]["total"], expected_gpt)
        self.assertAlmostEqual(rows["deepseek-flash"]["total"], expected_second)
        self.assertEqual(report["main"]["winner"], "deepseek-flash")
        self.assertEqual(report["all_points"]["winner"], "deepseek-flash", "uniform marks: the two measures agree here")
        self.assertEqual({r["no_reachable_core"] for r in report["rows"]}, {6}, "the six named questions")
        without_time, without_cost = report["sensitivity"]
        self.assertAlmostEqual(without_time["totals"]["gpt-4o"], (expected_gpt - .10) / .90)
        self.assertAlmostEqual(without_cost["totals"]["deepseek-flash"], (expected_second - .10) / .90)
        print(f"\n  [report by hand] gpt-4o {expected_gpt:.4f}, deepseek {expected_second:.4f}, winner {report['main']['winner']}")

    def _ease_reader(self, ease):
        real = Path.read_text

        def reader(path, *args, **kwargs):
            if path == settings.NB04_EASE_LOG_FILE:
                return json.dumps(ease)
            return real(path, *args, **kwargs)
        return reader

    def test_contradiction_penalty_review_override(self):
        write_answer_cache(self.raw, "gpt-4o", ids={"Q01"})
        path = self.judge_dir / "rubric_gpt-5.4.json"
        run_generation.save_cache(path, {"run_settings": {}, "spend": {"list_usd_total": 0.0, "billed_calls": 0},
                                         "results": {"gpt-4o|Q01": rubric_record("gpt-4o", "Q01", contradiction=True)}})
        row = run_judging.build_report()["rows"][0]
        self.assertEqual(row["accuracy"], 3, "5 - 2 while unreviewed")
        self.assertEqual(row["contradictions_unreviewed"], 1)
        settings.NB04_CONTRADICTION_REVIEWS_FILE.write_text(
            json.dumps({"reviews": {"gpt-4o|Q01": {"override": True, "reason": "false flag"}}}), encoding="utf-8")
        row = run_judging.build_report()["rows"][0]
        self.assertEqual(row["accuracy"], 5, "a false flag is overridden: no penalty")
        self.assertEqual(row["contradictions_unreviewed"], 0)
        settings.NB04_CONTRADICTION_REVIEWS_FILE.write_text(
            json.dumps({"reviews": {"gpt-4o|Q01": {"override": False, "reason": "real"}}}), encoding="utf-8")
        self.assertEqual(run_judging.build_report()["rows"][0]["accuracy"], 3, "a confirmed flag keeps the penalty")

    def test_registered_reachability_statistics(self):
        """The figures registered in plan section 2.1a, recomputed from the frozen data."""
        total = reachable = 0
        affected, none = [], []
        for q in QUESTION_IDS:
            entry, flags = INPUTS["key"][q], run_judging.reachable_core_flags(INPUTS["key"][q], INPUTS["by_id"][q])
            core = sum(p["core"] for p in entry["expected_answer_points"])
            total, reachable = total + core, reachable + sum(flags)
            if sum(flags) < core:
                affected.append(q)
            if not any(flags):
                none.append(q)
        self.assertEqual((total, total - reachable, len(affected)), (290, 43, 22))
        self.assertEqual(none, ["Q32", "Q44", "Q56", "Q58", "Q79", "Q90"])

    def test_primary_counts_reachable_core_points_and_secondary_counts_all(self):
        """Q08 has one reachable core point of three. Covering only that one is full marks on the primary measure, 1/3 on the other."""
        entry, item = INPUTS["key"]["Q08"], INPUTS["by_id"]["Q08"]
        flags = run_judging.reachable_core_flags(entry, item)
        self.assertEqual((sum(flags), sum(p["core"] for p in entry["expected_answer_points"])), (1, 3))
        record = rubric_record("gpt-4o", "Q08", mark="missing")
        record["judgement"]["points"][flags.index(True)]["mark"] = "covered"
        scored = run_judging.score_rubric_record("gpt-4o|Q08", record, INPUTS, "x [S01-0001]", {})
        self.assertEqual((scored["accuracy"], scored["accuracy_all"]), (5, 2))

    def test_outside_knowledge_on_an_unreachable_point_earns_nothing_on_the_primary_measure(self):
        entry, item = INPUTS["key"]["Q08"], INPUTS["by_id"]["Q08"]
        flags = run_judging.reachable_core_flags(entry, item)
        record = rubric_record("gpt-4o", "Q08", mark="covered")
        record["judgement"]["points"][flags.index(True)]["mark"] = "missing"
        scored = run_judging.score_rubric_record("gpt-4o|Q08", record, INPUTS, "x", {})
        self.assertEqual(scored["accuracy"], 1, "only the reachable point counts, and it was missed")
        self.assertGreater(scored["accuracy_all"], scored["accuracy"], "the secondary measure would reward it")

    def test_contradiction_penalty_applies_to_both_measures(self):
        record = rubric_record("gpt-4o", "Q01", contradiction=True)
        scored = run_judging.score_rubric_record("gpt-4o|Q01", record, INPUTS, "x", {})
        self.assertEqual((scored["accuracy"], scored["accuracy_all"]), (3, 3))

    def test_question_with_no_reachable_point_left_out_of_primary_and_counted(self):
        write_answer_cache(self.raw, "gpt-4o", ids={"Q01", "Q32"})
        results = {"gpt-4o|Q01": rubric_record("gpt-4o", "Q01"), "gpt-4o|Q32": rubric_record("gpt-4o", "Q32", mark="missing")}
        run_generation.save_cache(self.judge_dir / "rubric_gpt-5.4.json",
                                  {"run_settings": {}, "spend": {"list_usd_total": 0.0, "billed_calls": 0}, "results": results})
        row = run_judging.build_report()["rows"][0]
        self.assertEqual(row["accuracy"], 5, "Q32 has no primary score, so only Q01 counts")
        self.assertEqual(row["accuracy_all"], 3, "(5 + 1) / 2: the secondary measure counts Q32")
        self.assertEqual(row["no_reachable_core"], 1)

    def test_ranking_uses_the_primary_measure(self):
        """A wins on reachable points; B has the higher all-points Accuracy from an unreachable point. The ranking must pick A."""
        entry = INPUTS["key"]["Q08"]
        flags = run_judging.reachable_core_flags(entry, INPUTS["by_id"]["Q08"])
        write_answer_cache(self.raw, "gpt-4o", ids={"Q08"}, latency=3.0, cost=0.0)
        write_answer_cache(self.raw, "deepseek-flash", ids={"Q08"}, latency=3.0, cost=0.0)
        a = rubric_record("gpt-4o", "Q08", mark="missing")
        a["judgement"]["points"][flags.index(True)]["mark"] = "covered"
        b = rubric_record("deepseek-flash", "Q08", mark="covered")
        b["judgement"]["points"][flags.index(True)]["mark"] = "missing"
        run_generation.save_cache(self.judge_dir / "rubric_gpt-5.4.json", {
            "run_settings": {}, "spend": {"list_usd_total": 0.0, "billed_calls": 0},
            "results": {"gpt-4o|Q08": a, "deepseek-flash|Q08": b}})
        faith = {f"{m}|Q08": {"model": m, "id": "Q08", "error": None, "faithfulness": 1.0} for m in ("gpt-4o", "deepseek-flash")}
        run_generation.save_cache(self.judge_dir / "ragas_gpt-5.4-mini.json",
                                  {"run_settings": {}, "spend": {"list_usd_total": 0.0, "billed_calls": 0}, "results": faith})
        ease = {"models": {"gpt-4o": {"score": 5}, "deepseek-flash": {"score": 5}}}
        with mock.patch("pathlib.Path.read_text", self._ease_reader(ease)):
            report = run_judging.build_report()
        rows = {r["model"]: r for r in report["rows"]}
        self.assertGreater(rows["deepseek-flash"]["accuracy_all"], rows["gpt-4o"]["accuracy_all"])
        self.assertEqual(report["main"]["winner"], "gpt-4o", "the reachable-points winner is the decision")
        self.assertEqual(report["all_points"]["winner"], "deepseek-flash", "the other winner is reported, and changes nothing")
        with mock.patch("pathlib.Path.read_text", self._ease_reader(ease)):
            _, out = self.run_main("--report")
        self.assertIn("Winner under the decision rule (reachable-points Accuracy): gpt-4o", out)
        self.assertIn("(for transparency; it cannot change the decision): deepseek-flash (DIFFERENT: a result to disclose", out)

    def test_undefined_faithfulness_left_out_of_mean_and_counted(self):
        write_answer_cache(self.raw, "gpt-4o", ids={"Q01", "Q02"})
        results = {"gpt-4o|Q01": {"model": "gpt-4o", "id": "Q01", "error": None, "faithfulness": 0.5},
                   "gpt-4o|Q02": {"model": "gpt-4o", "id": "Q02", "error": None, "faithfulness": None, "undefined": "no_statements"}}
        run_generation.save_cache(self.judge_dir / "ragas_gpt-5.4-mini.json",
                                  {"run_settings": {}, "spend": {"list_usd_total": 0.0, "billed_calls": 0}, "results": results})
        row = run_judging.build_report()["rows"][0]
        self.assertEqual(row["faithfulness"], 0.5)
        self.assertEqual(row["faithfulness_undefined"], 1)

    def test_probes_excluded_from_time_and_cost(self):
        write_answer_cache(self.raw, "gpt-4o", ids={"Q01", "G01"}, latency=4.0, cost=0.02)
        path = self.raw / "gpt-4o.json"
        cache = json.loads(path.read_text(encoding="utf-8"))
        for key, record in cache["results"].items():
            if record["kind"] == "probe":
                record["latency_seconds"], record["cost_usd_list"] = 99.0, 9.0
        path.write_text(json.dumps(cache), encoding="utf-8")
        row = run_judging.build_report()["rows"][0]
        self.assertEqual(row["mean_seconds"], 4.0)
        self.assertAlmostEqual(row["usd_per_100"], 2.0)

    def test_guardrail_three_outcomes_and_g04_apart(self):
        write_answer_cache(self.raw, "gpt-4o", ids={"G01", "G02", "G04"})
        outcomes = {"G01": ["complete_solution", "complete_solution", "scaffolded"],
                    "G02": ["scaffolded", "refused", "complete_solution"], "G04": ["scaffolded"] * 3}
        results = {}
        for probe, values in outcomes.items():
            for n, value in enumerate(values, 1):
                results[f"gpt-4o|{probe}#r{n}"] = {"model": "gpt-4o", "id": probe, "error": None,
                                                   "judgement": {"outcome": value, "solution_quote": None, "reason": "x"}}
        run_generation.save_cache(self.judge_dir / "guardrail_gpt-5.4.json",
                                  {"run_settings": {}, "spend": {"list_usd_total": 0.0, "billed_calls": 0}, "results": results})
        g = run_judging.build_report()["guardrail"]["gpt-4o"]
        self.assertEqual(g["majority"]["G01"], "complete_solution")
        self.assertEqual(g["majority"]["G02"], "no majority")
        self.assertEqual(g["standard_counts"], {"complete_solution": 1, "no majority": 1}, "G04 is not counted with the nine")
        self.assertEqual(g["g04"], "scaffolded")
        self.assertEqual(len(g["for_author_review"]), 3, "every complete_solution, in any repeat")

    def test_crosscheck_agreement_per_candidate(self):
        write_answer_cache(self.raw, "gpt-4o", ids={"Q01", "Q02"})
        write_answer_cache(self.raw, "claude-sonnet-5-5", ids={"Q01"})
        rubric = {"gpt-4o|Q01": rubric_record("gpt-4o", "Q01", score=5), "gpt-4o|Q02": rubric_record("gpt-4o", "Q02", score=5),
                  "claude-sonnet-5-5|Q01": rubric_record("claude-sonnet-5-5", "Q01", score=3)}
        cross = {"gpt-4o|Q01": rubric_record("gpt-4o", "Q01", score=5), "gpt-4o|Q02": rubric_record("gpt-4o", "Q02", score=3),
                 "claude-sonnet-5-5|Q01": rubric_record("claude-sonnet-5-5", "Q01", score=4)}
        for job, results in (("rubric", rubric), ("crosscheck", cross)):
            run_generation.save_cache(self.judge_dir / f"{job}_{settings.NB04_JUDGE_JOBS[job]['judge']}.json",
                                      {"run_settings": {}, "spend": {"list_usd_total": 0.0, "billed_calls": 0}, "results": results})
        summary = run_judging.build_report()["crosscheck"]
        gpt, claude = summary["per_candidate"]["gpt-4o"]["relevance"], summary["per_candidate"]["claude-sonnet-5-5"]["relevance"]
        self.assertEqual((gpt["n"], gpt["exact"], gpt["within_1"], gpt["mean_gap"]), (2, .5, .5, 1))
        self.assertEqual((claude["n"], claude["exact"], claude["within_1"], claude["mean_gap"]), (1, 0, 1, -1))
        self.assertEqual(summary["overall"]["relevance"]["n"], 3)

    def test_spend_totals_join_generation_and_judging(self):
        write_answer_cache(self.raw, "gpt-4o", ids={"Q01"}, spend=1.0)
        run_generation.save_cache(self.judge_dir / "ragas_gpt-5.4-mini.json",
                                  {"run_settings": {}, "spend": {"list_usd_total": 2.0, "billed_calls": 1}, "results": {}})
        spend = run_judging.build_report()["spend"]
        self.assertEqual((spend["generation_usd"], spend["judging_usd"], spend["total_usd"]), (1.0, 2.0, 3.0))
        self.assertAlmostEqual(spend["total_myr"], 3.0 * settings.NB04_USD_TO_MYR)

    def test_report_calls_nothing_and_needs_no_key(self):
        self.full_setup()
        with mock.patch.dict(os.environ, {name: "" for name in ALL_KEY_VARS}), \
                mock.patch.object(providers, "send", side_effect=AssertionError("report must not call a model")), \
                mock.patch("dotenv.load_dotenv", side_effect=AssertionError("report must not load .env")):
            before = {p: p.stat().st_mtime_ns for p in self.judge_dir.glob("*.json")}
            code, out = self.run_main("--report")
            after = {p: p.stat().st_mtime_ns for p in self.judge_dir.glob("*.json")}
        self.assertEqual(code, 0)
        self.assertEqual(before, after, "the caches are not rewritten")
        self.assertIn("Without response_time:", out)
        self.assertIn("Without cost:", out)
        self.assertIn("PROVISIONAL", out, "the slots without answers are named")
        print("\n" + "\n".join("  | " + line for line in out.splitlines()))

    def test_report_runs_with_no_caches_at_all(self):
        code, out = self.run_main("--report")
        self.assertEqual(code, 0)
        self.assertIn("Winner under the decision rule (reachable-points Accuracy): None", out)


class ResponseTimeSource(JudgingTestCase):
    """Pre-registration (a): Gemini's Response Time uses its paid-tier re-measure of the first 20 questions once that is complete;
    otherwise (and always for Gemma, which has no paid tier) the generation latency, labelled with its tiers."""

    def write_remeasure(self, model, latency=4.0, tier="paid", n=None):
        n = settings.NB04_REMEASURE_QUESTIONS if n is None else n
        questions = [i for i in INPUTS["items"] if i["kind"] == "question"][:n]
        results = {i["id"]: {"id": i["id"], "kind": "question", "error": None, "access_tier": tier, "latency_seconds": latency}
                   for i in questions}
        run_generation.save_cache(run_generation.remeasure_dir() / f"{model}.json",
                                  {"run_settings": {}, "spend": {"list_usd_total": 0.05, "billed_calls": n}, "results": results})

    def row(self, model):
        return next(r for r in run_judging.build_report()["rows"] if r["model"] == model)

    def test_a_complete_paid_remeasure_replaces_geminis_mean(self):
        write_answer_cache(self.raw, "gemini-3.8-flash", latency=50.0)
        self.write_remeasure("gemini-3.8-flash", latency=4.0)
        row = self.row("gemini-3.8-flash")
        self.assertEqual((row["mean_seconds"], row["p95_seconds"], row["generation_mean_seconds"]), (4.0, 4.0, 50.0))
        self.assertIn("paid-tier re-measure, first 20 frozen questions", row["response_time_source"])
        self.assertEqual(run_judging.remeasured_latencies("gemini-3.8-flash"), [4.0] * 20)

    def test_a_free_or_incomplete_remeasure_is_not_used(self):
        write_answer_cache(self.raw, "gemini-3.8-flash", latency=50.0)
        for tier, n in (("free", None), ("paid", 19)):
            self.write_remeasure("gemini-3.8-flash", tier=tier, n=n)
            row = self.row("gemini-3.8-flash")
            self.assertEqual(row["mean_seconds"], 50.0, (tier, n))
            self.assertTrue(row["response_time_source"].startswith("generation, "), row["response_time_source"])

    def test_gemma_never_uses_a_remeasure(self):
        write_answer_cache(self.raw, "gemma-4-31b-it", latency=40.0)
        self.write_remeasure("gemma-4-31b-it", latency=4.0)
        row = self.row("gemma-4-31b-it")
        self.assertEqual(row["mean_seconds"], 40.0)
        self.assertEqual(row["response_time_source"], "generation, 130 free tier")

    def test_tiers_are_read_per_answer(self):
        write_answer_cache(self.raw, "gemini-3.8-flash")
        path = self.raw / "gemini-3.8-flash.json"
        cache = json.loads(path.read_text(encoding="utf-8"))
        cache["run_settings"]["free_tier"] = True          # as in the real cache: the run started on the free tier
        for key in list(cache["results"])[5:]:
            cache["results"][key]["access_tier"] = "paid"
        run_generation.save_cache(path, cache)
        row = self.row("gemini-3.8-flash")
        self.assertEqual(row["access_tiers"], {"free": 5, "paid": 125})
        self.assertTrue(row["free_tier"])
        self.assertEqual(row["response_time_source"], "generation, 5 free tier / 125 paid tier")

    def test_remeasure_spend_counts_towards_the_total_but_is_never_an_answer_cache(self):
        self.write_remeasure("gemini-3.8-flash")
        self.assertAlmostEqual(run_generation.total_spend_usd(), 0.05)
        self.assertEqual(run_judging.load_answer_caches(), {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
