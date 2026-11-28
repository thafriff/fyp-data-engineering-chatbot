"""
Offline tests for batch_judging.py (STEP 3, half-price judging; prepared, not run). NOTHING HERE CAN REACH A REAL API: the network guard,
fake keys, fake judge and temporary folders come from test_run_judging.py; batches go to fake providers, and the two real provider
adapters are exercised with fake SDK clients.

What must hold: the requests are run_judging's own (same judge models, prompts and settings), the validation and the one malformed retry
are run_judging's, billing uses the official batch price (half the standard), the batch route is recorded and cannot be mixed with the
standard route in one cache, RAGAS takes two rounds, an interrupted run resumes its batch instead of paying for it twice, and a projected
overrun stops before anything is sent.

Usage (from the project root):  venv\\Scripts\\python evaluation\\notebook04\\test_batch_judging.py -v
"""

import contextlib
import io
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_run_judging as base  # noqa: E402  (installs the network guard first)
from test_run_judging import INPUTS, JudgingTestCase, write_answer_cache  # noqa: E402

import batch_judging  # noqa: E402
import run_judging  # noqa: E402
from config import settings  # noqa: E402


def tearDownModule():
    base.tearDownModule()


class FakeProvider:
    """A batch provider built on the fake judge: each request is answered as providers.send would answer it. polls_until_done
    makes a batch report "in_progress" that many times first; errors lists custom IDs to return as errored once."""

    def __init__(self, fake_judge, provider="openai", polls_until_done=0, errors=()):
        self.fake, self.provider, self.submitted, self.polls = fake_judge, provider, [], {}
        self.polls_until_done, self.errors = polls_until_done, set(errors)

    def submit(self, requests):
        self.submitted.append(dict(requests))
        return f"batch_{len(self.submitted)}"

    def poll(self, batch_id):
        self.polls[batch_id] = self.polls.get(batch_id, 0) + 1
        if self.polls[batch_id] <= self.polls_until_done:
            return "in_progress", None
        replies = {}
        for cid, body in self.submitted[int(batch_id.split("_")[1]) - 1].items():
            if cid in self.errors:
                self.errors.discard(cid)
                replies[cid] = {"error": "server_error"}
                continue
            result = self.fake({"model_id": body["model"]}, body)
            replies[cid] = {"text": result.text, "finish": result.finish, "input_tokens": result.input_tokens,
                            "output_tokens": result.output_tokens, "reasoning_tokens": 0, "served_model": body["model"], "error": None}
        return "completed", replies


class BatchTestCase(JudgingTestCase):
    def setUp(self):
        super().setUp()
        for patch in (mock.patch.object(settings, "NB04_BATCH_DIR", self.judge_dir / "batches"),
                      mock.patch.object(batch_judging.time, "sleep", lambda s: None),
                      mock.patch.object(batch_judging.check_parity, "run_checks", lambda: [])):
            patch.start()
            self.addCleanup(patch.stop)

    def run_batch(self, job, provider, **kwargs):
        out, code = io.StringIO(), 0
        with contextlib.redirect_stdout(out):
            try:
                batch_judging.run_batch_job(job, batches=provider, **kwargs)
            except SystemExit as stop:
                code = stop.code or 0
        return code, out.getvalue()


class SameJudgingOnlyThePriceChanges(BatchTestCase):

    def test_requests_are_run_judgings_own(self):
        for model in ("gpt-4o", "deepseek-flash"):
            write_answer_cache(self.raw, model, ids={"Q01"})
        provider = FakeProvider(self.fake)
        code, _ = self.run_batch("rubric", provider, only="Q01", wait=True)
        self.assertEqual(code, 0)
        sent = list(provider.submitted[0].values())
        judge = settings.NB04_JUDGE_MODELS["gpt-5.4"]
        expected = [run_judging.build_request("rubric", judge, run_judging.rubric_messages(
            INPUTS["key"]["Q01"], INPUTS["by_id"]["Q01"], base.fake_answer(INPUTS["by_id"]["Q01"]))) for _ in range(2)]
        self.assertEqual(sent, expected)
        self.assertEqual(sent[0]["model"], "gpt-5.4")
        self.assertEqual((sent[0]["temperature"], sent[0]["reasoning_effort"], sent[0]["response_format"]),
                         (0, "none", {"type": "json_object"}))

    def test_billed_at_the_batch_price_half_of_standard(self):
        write_answer_cache(self.raw, "gpt-4o", ids={"Q01"})
        self.run_batch("rubric", FakeProvider(self.fake), only="Q01", wait=True)
        cache = self.job_cache("rubric")
        batch_cost = (1000 * 1.25 + 100 * 7.50) / 1e6
        standard_cost = (1000 * 2.50 + 100 * 15.00) / 1e6
        self.assertAlmostEqual(cache["spend"]["list_usd_total"], batch_cost)
        self.assertAlmostEqual(batch_cost, standard_cost / 2)
        self.assertAlmostEqual(cache["spend"]["by_candidate"]["gpt-4o"], batch_cost)
        record = cache["results"]["gpt-4o|Q01"]
        self.assertIsNone(record["error"])
        self.assertEqual((record["calls"], record["pricing_route"]), (1, "batch"))
        self.assertEqual(len(record["judgement"]["points"]), len(INPUTS["key"]["Q01"]["expected_answer_points"]))
        self.assertEqual(self.fake.requests[0][1], run_judging.build_request(
            "rubric", settings.NB04_JUDGE_MODELS["gpt-5.4"], run_judging.rubric_messages(
                INPUTS["key"]["Q01"], INPUTS["by_id"]["Q01"], base.fake_answer(INPUTS["by_id"]["Q01"]))))

    def test_route_recorded_and_the_standard_runner_refuses_this_cache(self):
        write_answer_cache(self.raw, "gpt-4o", ids={"Q01"})
        self.run_batch("rubric", FakeProvider(self.fake), only="Q01", wait=True)
        run_settings = self.job_cache("rubric")["run_settings"]
        self.assertEqual((run_settings["pricing_route"], run_settings["input_usd_per_m"], run_settings["output_usd_per_m"]),
                         ("batch", 1.25, 7.50))
        self.assertEqual(run_settings["standard_prices"], [2.50, 15.00])
        before = len(self.fake.requests)
        code, out = self.run_main("--job", "rubric", "--only", "Q01")
        self.assertEqual(code, run_judging.EXIT_SETUP)
        self.assertIn("different settings", out)
        self.assertEqual(len(self.fake.requests), before, "the standard runner sent nothing")

    def test_malformed_reply_resubmitted_once_then_recorded_as_error(self):
        write_answer_cache(self.raw, "gpt-4o", ids={"Q01"})
        self.fake.script = {1: "this is not JSON", 2: '{"points": "still wrong"}'}
        provider = FakeProvider(self.fake)
        code, _ = self.run_batch("rubric", provider, only="Q01", wait=True)
        record = self.job_cache("rubric")["results"]["gpt-4o|Q01"]
        self.assertEqual(code, 0)
        self.assertEqual(len(provider.submitted), 2, "the retry is a second batch")
        self.assertIn("malformed after 2 attempts", record["error"])
        self.assertEqual((record["calls"], record["malformed_retries"]), (2, 1))

    def test_ragas_takes_two_rounds_and_scores_like_the_standard_route(self):
        write_answer_cache(self.raw, "gpt-4o", ids={"Q01"})
        provider = FakeProvider(self.fake)
        code, _ = self.run_batch("ragas", provider, only="Q01", wait=True)
        record = self.job_cache("ragas")["results"]["gpt-4o|Q01"]
        self.assertEqual(code, 0)
        self.assertEqual(len(provider.submitted), 2, "statements first, then NLI on them")
        self.assertAlmostEqual(record["faithfulness"], 2 / 3)
        self.assertEqual(record["calls"], 2)

    def test_guardrail_probe_repeats_in_one_batch(self):
        write_answer_cache(self.raw, "gpt-4o", ids={"G01"})
        provider = FakeProvider(self.fake)
        self.run_batch("guardrail", provider, only="G01", wait=True)
        results = self.job_cache("guardrail")["results"]
        self.assertEqual(len(provider.submitted), 1)
        self.assertEqual(len(provider.submitted[0]), 3)
        self.assertEqual({r["judgement"]["outcome"] for r in results.values()}, {"scaffolded"})


class SafetyOfTheBatchRoute(BatchTestCase):

    def test_an_interrupted_run_resumes_its_batch_and_pays_once(self):
        write_answer_cache(self.raw, "gpt-4o", ids={"Q01"})
        provider = FakeProvider(self.fake, polls_until_done=1)
        code, out = self.run_batch("rubric", provider, only="Q01", wait=False)
        self.assertEqual(code, batch_judging.EXIT_PENDING)
        self.assertIn("Submitted batch", out)
        code, out = self.run_batch("rubric", provider, only="Q01", wait=False)      # still running: nothing new is submitted
        self.assertEqual((code, len(provider.submitted)), (batch_judging.EXIT_PENDING, 1))
        code, out = self.run_batch("rubric", provider, only="Q01", wait=False)      # ended: collected, judged, done
        self.assertEqual((code, len(provider.submitted)), (0, 1))
        self.assertEqual(self.job_cache("rubric")["spend"]["billed_calls"], 1)

    def test_a_projected_overrun_stops_before_anything_is_sent(self):
        write_answer_cache(self.raw, "gpt-4o", ids={"Q01", "Q02"})
        provider = FakeProvider(self.fake)
        with mock.patch.object(batch_judging, "per_request_estimate_usd", lambda job: 10.0):     # $20 projected, limit $3.23
            code, out = self.run_batch("rubric", provider, only="Q01,Q02", wait=True)
        self.assertEqual(code, batch_judging.EXIT_HARD_STOP)
        self.assertIn("Nothing was sent", out)
        self.assertEqual(provider.submitted, [])

    def test_an_errored_request_is_resubmitted(self):
        write_answer_cache(self.raw, "gpt-4o", ids={"Q01"})
        request = run_judging.build_request("rubric", settings.NB04_JUDGE_MODELS["gpt-5.4"], run_judging.rubric_messages(
            INPUTS["key"]["Q01"], INPUTS["by_id"]["Q01"], base.fake_answer(INPUTS["by_id"]["Q01"])))
        provider = FakeProvider(self.fake, errors={batch_judging.custom_id("gpt-4o|Q01", request, 1)})
        code, _ = self.run_batch("rubric", provider, only="Q01", wait=True)
        self.assertEqual((code, len(provider.submitted)), (0, 2))
        cache = self.job_cache("rubric")
        self.assertIsNone(cache["results"]["gpt-4o|Q01"]["error"])
        self.assertEqual(cache["spend"]["billed_calls"], 1, "an errored request returns no tokens and is not billed")

    def test_custom_ids_fit_both_providers(self):
        cid = batch_judging.custom_id("deepseek-flash|G04#r3", {"model": "x", "messages": []}, 2)
        self.assertNotEqual(cid, batch_judging.custom_id("deepseek-flash|G04#r2", {"model": "x", "messages": []}, 2),
                            "identical requests from two judgements get their own calls")
        self.assertRegex(cid, r"^[a-zA-Z0-9_-]{1,64}$")


class ProviderAdapters(unittest.TestCase):
    """The two real adapters, driven by fake SDK clients shaped like the official ones."""

    def test_openai_jsonl_and_output_parsing(self):
        uploads = []
        output = "\n".join([
            json.dumps({"custom_id": "a-a1", "response": {"status_code": 200, "body": {
                "model": "gpt-5.4-2026-03-05", "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 900, "completion_tokens": 80, "completion_tokens_details": {"reasoning_tokens": 0}}}}}),
            json.dumps({"custom_id": "b-a1", "response": {"status_code": 500, "body": {"error": {"message": "x"}}}})])
        client = SimpleNamespace(
            files=SimpleNamespace(create=lambda file, purpose: uploads.append((file, purpose)) or SimpleNamespace(id="file_1"),
                                  content=lambda fid: SimpleNamespace(text=output)),
            batches=SimpleNamespace(create=lambda **kw: SimpleNamespace(id="batch_1", kw=kw),
                                    retrieve=lambda bid: SimpleNamespace(status="completed", output_file_id="out", error_file_id=None)))
        adapter = batch_judging.OpenAIBatches(client)
        self.assertEqual(adapter.submit({"a-a1": {"model": "gpt-5.4", "messages": []}}), "batch_1")
        name, data = uploads[0][0]
        line = json.loads(data.decode("utf-8"))
        self.assertEqual((uploads[0][1], line["method"], line["url"], line["custom_id"]), ("batch", "POST", "/v1/chat/completions", "a-a1"))
        status, replies = adapter.poll("batch_1")
        self.assertEqual(status, "completed")
        self.assertEqual((replies["a-a1"]["input_tokens"], replies["a-a1"]["finish"], replies["a-a1"]["error"]), (900, "stop", None))
        self.assertTrue(replies["b-a1"]["error"])

    def test_anthropic_requests_and_results_parsing(self):
        created = []
        results = [SimpleNamespace(model_dump=lambda: {"custom_id": "c-a1", "result": {"type": "succeeded", "message": {
                       "model": "claude-opus-5-5", "stop_reason": "end_turn", "content": [{"type": "text", "text": "{}"}],
                       "usage": {"input_tokens": 1200, "output_tokens": 300}}}}),
                   SimpleNamespace(model_dump=lambda: {"custom_id": "d-a1", "result": {"type": "expired"}})]
        batches = SimpleNamespace(create=lambda requests: created.append(requests) or SimpleNamespace(id="msgbatch_1"),
                                  retrieve=lambda bid: SimpleNamespace(processing_status="ended"),
                                  results=lambda bid: iter(results))
        adapter = batch_judging.AnthropicBatches(SimpleNamespace(messages=SimpleNamespace(batches=batches)))
        self.assertEqual(adapter.submit({"c-a1": {"model": "claude-opus-5-5"}}), "msgbatch_1")
        self.assertEqual(created[0], [{"custom_id": "c-a1", "params": {"model": "claude-opus-5-5"}}])
        status, replies = adapter.poll("msgbatch_1")
        self.assertEqual((status, replies["c-a1"]["finish"], replies["c-a1"]["output_tokens"]), ("ended", "stop", 300))
        self.assertTrue(replies["d-a1"]["error"])


class CreditFit(BatchTestCase):
    """Author, 2026-11-28: check the prepaid balance fits before each batch; if it runs short, judge as many as fit, stop, and say how
    much to top up. The ledger here sees only the test caches (the real key_check.json is hidden)."""

    def setUp(self):
        super().setUp()
        patch = mock.patch.object(batch_judging.credit_ledger, "KEY_CHECK_FILE", self.judge_dir / "no_key_check.json")
        patch.start()
        self.addCleanup(patch.stop)
        for model in ("gpt-4o", "deepseek-flash"):
            write_answer_cache(self.raw, model, ids={"Q01", "Q02"}, spend=0.01)
        self.reserve = batch_judging.per_request_estimate_usd("rubric") * settings.NB04_BATCH_CREDIT_MARGIN

    def run_batch(self, job, provider, **kwargs):
        return super().run_batch(job, provider, **{"allow_incomplete": True, **kwargs})

    def credit(self, usd):
        spent = sum(batch_judging.credit_ledger.spent_usd("openai").values())
        return mock.patch.dict(settings.NB04_PREPAID_CREDIT_USD, {"openai": spent + usd})

    def test_only_what_fits_is_sent_then_it_stops_with_the_top_up(self):
        provider = FakeProvider(self.fake)
        with self.credit(1.5 * self.reserve):
            code, out = self.run_batch("rubric", provider, wait=True)
        self.assertEqual(code, batch_judging.EXIT_CREDIT, out)
        self.assertEqual([len(b) for b in provider.submitted], [1, 1], "one request fitted each time; never more than the credit")
        self.assertIn("CREDIT STOP", out)
        self.assertIn("top up $", out)
        judged = [k for k, r in self.job_cache("rubric")["results"].items() if not r["error"]]
        self.assertEqual(len(judged), 2)
        with self.credit(1.0):                                     # after a top-up the same command finishes the job
            code, out = self.run_batch("rubric", provider, wait=True)
        self.assertEqual(code, 0, out)
        self.assertEqual(len([r for r in self.job_cache("rubric")["results"].values() if not r["error"]]), 4)
        self.assertEqual(sum(len(b) for b in provider.submitted), 4, "nothing was sent twice")

    def test_part_of_a_batch_is_sent_when_only_part_fits(self):
        provider = FakeProvider(self.fake)
        with self.credit(3.5 * self.reserve):
            code, out = self.run_batch("rubric", provider, wait=False)
        self.assertEqual(code, batch_judging.EXIT_PENDING)
        self.assertEqual(len(provider.submitted[0]), 3)
        self.assertIn("CREDIT: 3 of 4 requests fit", out)

    def test_nothing_is_sent_when_nothing_fits(self):
        provider = FakeProvider(self.fake)
        with self.credit(0.5 * self.reserve):
            code, out = self.run_batch("rubric", provider, wait=True)
        self.assertEqual((code, provider.submitted), (batch_judging.EXIT_CREDIT, []))

    def test_a_ragas_statements_request_reserves_its_nli_request(self):
        state = {"submitted": {"s1": {"unit": "gpt-4o|Q01"}, "s2": {"unit": "gpt-4o|Q02"}, "n1": {"unit": "gpt-4o|Q01"}},
                 "replies": {"s1": {"cost_usd": 0.001}}}
        self.assertEqual(batch_judging.requests_committed("ragas", "s2", state), 2)
        self.assertEqual(batch_judging.requests_committed("ragas", "n1", state), 1)
        self.assertEqual(batch_judging.requests_committed("rubric", "s2", state), 1)

    def test_a_used_up_credit_reply_stops_instead_of_resubmitting(self):
        class OutOfCredit(FakeProvider):
            def poll(self, batch_id):
                status, replies = super().poll(batch_id)
                return status, {cid: {"error": '{"code": "insufficient_quota", "message": "You exceeded your current quota"}'}
                                for cid in replies}
        provider = OutOfCredit(self.fake)
        code, out = self.run_batch("rubric", provider, wait=True)
        self.assertEqual(code, batch_judging.EXIT_CREDIT, out)
        self.assertEqual(len(provider.submitted), 1, "not resubmitted")
        state = batch_judging.load_state("rubric", "gpt-5.4")
        self.assertEqual(state["transport_failures"], {})
        self.assertIn("insufficient_quota", state["credit_exhausted"]["error"])
        code, out = self.run_batch("rubric", FakeProvider(self.fake), wait=True)     # after a top-up
        self.assertEqual(code, 0, out)
        self.assertIn("credit_exhausted_history", batch_judging.load_state("rubric", "gpt-5.4"))

    def test_a_refused_submission_stops(self):
        class Refused(FakeProvider):
            def submit(self, requests):
                raise RuntimeError("Error code: 429 - {'error': {'code': 'insufficient_quota'}}")
        code, out = self.run_batch("rubric", Refused(self.fake), wait=True)
        self.assertEqual(code, batch_judging.EXIT_CREDIT, out)
        self.assertEqual(batch_judging.load_state("rubric", "gpt-5.4")["batches"], [])

    def test_a_provider_without_prepaid_credit_is_not_limited(self):
        with mock.patch.dict(settings.NB04_PREPAID_CREDIT_USD, {}, clear=True):
            code, out = self.run_batch("rubric", FakeProvider(self.fake), wait=True)
        self.assertEqual(code, 0, out)

    def test_running_batches_of_another_job_are_reserved(self):
        for model in ("gpt-4o", "deepseek-flash"):
            write_answer_cache(self.raw, model, ids={"Q01", "Q02", "G01"}, spend=0.01)
        provider = FakeProvider(self.fake, polls_until_done=10**6)
        with self.credit(10.0):
            self.run_batch("guardrail", provider, wait=False)
            self.run_batch("rubric", provider, wait=False)
        self.assertEqual([len(b) for b in provider.submitted], [6, 4])
        guardrail = batch_judging.per_request_estimate_usd("guardrail") * settings.NB04_BATCH_CREDIT_MARGIN
        self.assertAlmostEqual(batch_judging.running_reserve_usd("openai"), 6 * guardrail + 4 * self.reserve)
        self.assertEqual(batch_judging.running_reserve_usd("anthropic"), 0.0)

    def test_the_credit_report_calls_nothing(self):
        with mock.patch.object(batch_judging.providers, "send", side_effect=AssertionError("no call")):
            report = batch_judging.top_up_report()
        self.assertIn("openai: credit $", report)
        self.assertIn("rubric (gpt-5.4): 0 of 4 judged", report)


class BatchSize(BatchTestCase):
    """OpenAI rejects a batch as a whole when the organisation's enqueued-token limit is passed (nothing runs, nothing is billed):
    later batches are then half the size. Any other rejection stops the run."""

    def setUp(self):
        super().setUp()
        for model in ("gpt-4o", "deepseek-flash"):
            write_answer_cache(self.raw, model, ids={"Q01", "Q02"}, spend=0.01)

    def run_batch(self, job, provider, **kwargs):
        return super().run_batch(job, provider, **{"allow_incomplete": True, **kwargs})

    def test_batches_are_capped_and_sent_one_after_another(self):
        provider = FakeProvider(self.fake)
        with mock.patch.object(settings, "NB04_BATCH_MAX_REQUESTS", 3):
            code, out = self.run_batch("rubric", provider, wait=True)
        self.assertEqual(code, 0, out)
        self.assertEqual([len(b) for b in provider.submitted], [3, 1])

    def test_an_enqueued_token_rejection_halves_the_batches(self):
        class Limited(FakeProvider):
            def poll(self, batch_id):
                if len(self.submitted[int(batch_id.split("_")[1]) - 1]) > 2:
                    return "failed", {batch_judging.BATCH_ERROR: "token_limit_exceeded: Enqueued token limit reached for gpt-5.4"}
                return super().poll(batch_id)
        provider = Limited(self.fake)
        with mock.patch.object(settings, "NB04_BATCH_MIN_REQUESTS", 1):
            code, out = self.run_batch("rubric", provider, wait=True)
        self.assertEqual(code, 0, out)
        self.assertEqual([len(b) for b in provider.submitted], [4, 2, 2])
        state = batch_judging.load_state("rubric", "gpt-5.4")
        self.assertEqual((state["max_requests"], state["transport_failures"]), (2, {}))
        self.assertEqual(len([r for r in self.job_cache("rubric")["results"].values() if not r["error"]]), 4)
        self.assertEqual(self.job_cache("rubric")["spend"]["billed_calls"], 4, "the rejected batch billed nothing")

    def test_a_rejection_at_the_minimum_size_stops_instead_of_looping(self):
        class AlwaysLimited(FakeProvider):
            def poll(self, batch_id):
                return "failed", {batch_judging.BATCH_ERROR: "token_limit_exceeded: Enqueued token limit reached"}
        provider = AlwaysLimited(self.fake)
        with mock.patch.object(settings, "NB04_BATCH_MIN_REQUESTS", 2):
            code, out = self.run_batch("rubric", provider, wait=True)
        self.assertEqual(code, batch_judging.EXIT_SETUP, out)
        self.assertEqual([len(b) for b in provider.submitted], [4, 2], "halved once, then stopped at the minimum")

    def test_any_other_rejection_stops(self):
        class Broken(FakeProvider):
            def poll(self, batch_id):
                return "failed", {batch_judging.BATCH_ERROR: "invalid_request: bad file"}
        code, out = self.run_batch("rubric", Broken(self.fake), wait=True)
        self.assertEqual(code, batch_judging.EXIT_SETUP)
        self.assertIn("rejected batch", out)
        self.assertEqual(batch_judging.load_state("rubric", "gpt-5.4")["rejections"][0]["error"], "invalid_request: bad file")

    def test_the_openai_adapter_reports_a_whole_batch_rejection(self):
        error = SimpleNamespace(code="token_limit_exceeded", message="Enqueued token limit reached")
        client = SimpleNamespace(batches=SimpleNamespace(retrieve=lambda bid: SimpleNamespace(
            status="failed", output_file_id=None, error_file_id=None, errors=SimpleNamespace(data=[error]))))
        status, replies = batch_judging.OpenAIBatches(client).poll("batch_1")
        self.assertEqual(status, "failed")
        self.assertEqual(replies, {batch_judging.BATCH_ERROR: "token_limit_exceeded: Enqueued token limit reached"})


class BatchPrices(unittest.TestCase):
    def test_batch_prices_are_half_the_standard(self):
        for name, (batch_in, batch_out) in settings.NB04_JUDGE_BATCH_PRICES.items():
            judge = settings.NB04_JUDGE_MODELS[name]
            self.assertAlmostEqual(batch_in, judge["input_usd_per_m"] / 2)
            self.assertAlmostEqual(batch_out, judge["output_usd_per_m"] / 2)

    def test_batch_estimates(self):
        expected = {"rubric": 2.6875, "ragas": 1.51875, "guardrail": 0.56625, "crosscheck": 1.096}     # five candidates
        for job, usd in expected.items():
            self.assertAlmostEqual(batch_judging.batch_estimate_usd(job), usd, places=6)


if __name__ == "__main__":
    unittest.main(verbosity=1)
