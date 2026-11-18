import contextlib
import io
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch
from panda_bench.runner import adapt_response, base_url, cents, execute, fingerprint, initialize, request_body, request_key
from panda_bench.storage import digest, read_json, write_json as persist_json


def response(amount="7", status="completed", text="Synthetic answer", **extra):
    return {"choices": [{"message": {"content": text}}], "panda": {"version": "panda.chat.v1", "status": status,
        "charge": {"schema_version": "panda.usd-amount.v1", "currency": "USD", "minor_units": amount}, **extra}}


def fixture():
    prompts = {"synthetic-1": {"prompt": "First synthetic request", "checklist": []}, "synthetic-2": {"prompt": "Second synthetic request", "checklist": []}}
    manifest = {"stage": "stage-4", "prompts": [{"id": p, "sha256": digest(v)} for p, v in prompts.items()], "judge": {}, "baseline": "gemini-4-argon"}
    return manifest, prompts


class RunTests(unittest.TestCase):
    def test_charge_units_and_invalid_values(self):
        self.assertEqual(cents(response("123")["panda"]["charge"]), 123)
        for invalid in (None, {}, {"currency": "USD", "minor_units": "1"}):
            with self.assertRaises(ValueError): cents(invalid)
        for invalid in (1, -1, "-1", "1.5", "nan", "Infinity", "01", True):
            obj = response()["panda"]["charge"]; obj["minor_units"] = invalid
            with self.assertRaises(ValueError): cents(obj)
        obj = response()["panda"]["charge"]; obj["currency"] = "EUR"
        with self.assertRaises(ValueError): cents(obj)

    def test_response_partial_failed_replay_and_whitelist(self):
        obj = response(status="partial", agents=[{"index": 1, "model": "deepseek-v4-pro", "status": "completed", "usage": {"input_tokens": 2, "output_tokens": 4, "secret": "never-persist"}, "headers": {"Authorization": "never-persist"}, "request_id": "never-persist"}], request_id="never-persist", run_id="never-persist", token="never-persist")
        out = adapt_response(obj)
        self.assertEqual(out["state"], "completed")
        self.assertEqual(out["panda"]["status"], "partial")
        self.assertNotIn("never-persist", json.dumps(out))
        self.assertEqual(adapt_response(response(status="failed"), 500)["status"], "failed")
        settled = adapt_response(response(replay=True), 409)
        self.assertEqual(settled["state"], "blocked")
        self.assertEqual(settled["charge_cents"], 7)
        obj = response(status="running", replay=True); obj["panda"]["charge"] = None
        self.assertIsNone(adapt_response(obj, 409)["charge_cents"])
        self.assertEqual(adapt_response({"error": "secret"}, 429)["state"], "blocked")
        self.assertEqual(adapt_response(response(text=""))["state"], "blocked")

    def test_built_in_panel_configuration_provenance(self):
        for name, combiner in (("panda", "deepseek-v4.1-flash"), ("panda-quality", "glm-5.3")):
            obj = response(panel={"name": name, "revision": "0123456789abcdef", "synthesizer_model": combiner, "private": "never-persist"},
                           agents=[{"index": 1, "model": "deepseek-v4.1-flash", "status": "completed"},
                                   {"index": 2, "model": "gemini-3.8-flash", "status": "completed"}],
                           synthesis={"model": combiner, "status": "completed", "included_agents": [1, 2], "request_id": "never-persist"})
            actual = adapt_response(obj)["panda"]
            self.assertEqual([a["model"] for a in actual["agents"]], ["deepseek-v4.1-flash", "gemini-3.8-flash"])
            self.assertEqual(actual["panel"], {"name": name, "revision": "0123456789abcdef", "synthesizer_model": combiner})
            self.assertEqual(actual["synthesis"]["included_agents"], [1, 2])
            self.assertNotIn("never-persist", json.dumps(actual))

    def test_key_binds_model_content_and_run(self):
        body = request_body("one", {"prompt": "synthetic"})
        keys = {request_key("run-a", "p", "one", body), request_key("run-a", "p", "two", body), request_key("run-b", "p", "one", body), request_key("run-a", "p", "one", {**body, "max_completion_tokens": 1})}
        self.assertEqual(len(keys), 4)
        self.assertEqual(request_key("run-a", "p", "one", body), request_key("run-a", "p", "one", body))

    def test_two_workers_budget_stop_and_safe_resume(self):
        source, prompts = fixture()
        models = ["panel:panda", "gemini-4-argon"]
        lock = threading.Lock(); calls = []; active = 0; high = 0
        scheduled = set(); admissions = []; current_limit = 10
        def record_persistence(path, value):
            if Path(path).name == "run-state.json":
                settled = sum(row["charge_cents"] or 0 for row in value["requests"])
                for row in value["requests"]:
                    if row["state"] == "submitting" and row["key"] not in scheduled:
                        scheduled.add(row["key"])
                        admissions.append((settled, current_limit))
            persist_json(path, value)
        def fake(body, key):
            nonlocal active, high
            with lock:
                active += 1; high = max(high, active); calls.append(key)
            time.sleep(.03)
            with lock: active -= 1
            return adapt_response(response())
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()), patch("panda_bench.runner.write_json", side_effect=record_persistence):
            output = Path(temp) / "run"
            manifest, ledger = initialize(source, models, "https://example.test/v1", output, prompts)
            self.assertEqual(execute(output, manifest, ledger, prompts, .10, fake), 2)
            self.assertEqual(manifest["run_state"], "budget_stopped")
            # One completion may open a slot while only 7 cents are settled.
            # Both 2-call and 3-call schedules satisfy the threshold contract.
            self.assertIn(len(calls), (2, 3))
            self.assertEqual(sum(r["charge_cents"] or 0 for r in ledger["requests"]), 7 * len(calls))
            self.assertEqual(sum(r["state"] == "pending" for r in ledger["requests"]), 4 - len(calls))
            self.assertTrue(all(settled < limit for settled, limit in admissions))
            current_limit = 50
            manifest, ledger = read_json(output / "manifest.json"), read_json(output / "run-state.json")
            self.assertEqual(execute(output, manifest, ledger, prompts, .50, fake), 0)
            self.assertEqual(len(calls), 4)
            self.assertEqual(len(set(calls)), 4)
            self.assertEqual(high, 2)
            self.assertEqual(sum(r["charge_cents"] for r in ledger["requests"]), 28)
            self.assertTrue(all(settled < limit for settled, limit in admissions))
            self.assertEqual(len(admissions), 4)
            self.assertEqual(execute(output, manifest, ledger, prompts, .50, fake), 0)
            self.assertEqual(len(calls), 4)
            self.assertEqual(len(manifest["judge"]["bundle_before_shuffle"]["synthetic-1"]), 2)

    def test_ambiguous_timeout_blocks_future_submission_and_resume(self):
        source, prompts = fixture()
        def failing(body, key): raise RuntimeError("never-persist-this-auth-value")
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            output = Path(temp) / "run"
            manifest, ledger = initialize(source, ["panel:panda", "gemini-4-argon"], "https://example.test/v1", output, prompts)
            self.assertEqual(execute(output, manifest, ledger, prompts, 5, failing), 2)
            self.assertEqual(manifest["run_state"], "blocked")
            self.assertEqual(sum(r["state"] == "pending" for r in ledger["requests"]), 2)
            self.assertNotIn("never-persist", (output / "run-state.json").read_text())
            with self.assertRaisesRegex(ValueError, "unknown"):
                execute(output, manifest, ledger, prompts, 5, failing)

    def test_resume_refuses_changed_content_and_endpoint_identity(self):
        source, prompts = fixture()
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "run"
            manifest, ledger = initialize(source, ["gemini-4-argon"], "https://example.test/v1", output, prompts)
            prompts["synthetic-1"]["prompt"] = "Changed"
            with self.assertRaisesRegex(ValueError, "content changed"):
                execute(output, manifest, ledger, prompts, 5, lambda *_: self.fail("called"))
            self.assertNotEqual(fingerprint(source, ["gemini-4-argon"], "https://example.test/v1"), fingerprint(source, ["gemini-4-argon"], "https://different.test/v1"))

    def test_base_url_normalization_rejects_credentials(self):
        self.assertEqual(base_url("https://app.panda.repair/v1/chat/completions/"), "https://app.panda.repair/v1")
        self.assertEqual(base_url("http://localhost:9999/v1"), "http://localhost:9999/v1")
        for invalid in ("https://secret@example.test/v1", "https://example.test/v1?key=secret", "http://example.test/v1"):
            with self.assertRaises(ValueError): base_url(invalid)


if __name__ == "__main__": unittest.main()
