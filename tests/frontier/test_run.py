"""Offline checks of which rows a ``--retry-failed`` pass sends again: synthetic rows, no network."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from panda_bench.frontier.run import Ledger, _transient


def call(role="single", outcome="text", reason="stop", model="@cf/zai-org/glm-5.3-flash", attempts=("ok",)):
    return {"model": model, "role": role, "outcome": outcome, "reason": reason, "attempts": list(attempts)}


LOST = call("member", "error", "transport", attempts=("http_500",) * 4)
ANSWERED = call("member", model="@cf/deepseek-ai/deepseek-v4.1-flash-1017")
PANEL = {"mode": "synthesis", "members": ["a", "b", "c"], "answered": ["b", "c"]}


class TransientTest(unittest.TestCase):
    def test_harness_error_is_sent_again(self):
        self.assertTrue(_transient({"status": "harness_error", "calls": []}))

    def test_failure_with_only_transport_errors_is_sent_again(self):
        row = {"status": "failed", "calls": [call(outcome="error", reason="transport", attempts=("http_503",) * 4)]}
        self.assertTrue(_transient(row))

    def test_model_caused_failure_is_kept(self):
        self.assertFalse(_transient({"status": "failed", "calls": [call(outcome="incomplete", reason="max_tokens")]}))
        self.assertFalse(_transient({"status": "failed", "calls": [call(outcome="refused", reason="prompt_blocked")]}))

    def test_answer_is_kept(self):
        self.assertFalse(_transient({"status": "completed", "calls": [call()]}))

    def test_panel_that_lost_a_member_to_a_provider_error_is_sent_again(self):
        row = {"status": "partial", "panel": PANEL, "calls": [LOST, ANSWERED, ANSWERED, call("synthesis")]}
        self.assertTrue(_transient(row))

    def test_panel_that_lost_a_member_to_the_model_is_kept(self):
        for lost in (call("member", "incomplete", "max_tokens"), call("member", "refused", "prompt_blocked")):
            row = {"status": "partial", "panel": PANEL, "calls": [lost, ANSWERED, ANSWERED, call("synthesis")]}
            self.assertFalse(_transient(row))

    def test_router_embedding_error_alone_does_not_count(self):
        embedding = call("router_embedding", "error", "transport", model="@cf/google/embeddinggemma-2")
        self.assertFalse(_transient({"status": "completed", "route": {"pick": "x"}, "calls": [embedding, call()]}))


class LedgerTest(unittest.TestCase):
    def test_retry_pass_reopens_only_transient_rows(self):
        rows = [
            {"setup": "panel:trio-glm", "item": "1", "status": "partial", "panel": PANEL,
             "calls": [LOST, ANSWERED, ANSWERED, call("synthesis")]},
            {"setup": "panel:trio-glm", "item": "2", "status": "completed", "panel": PANEL, "calls": [ANSWERED]},
            {"setup": "auto", "item": "3", "status": "failed",
             "calls": [call(outcome="error", reason="transport", attempts=("http_500",) * 4)]},
            # A retried row that answered replaces its failed one: the newest row wins.
            {"setup": "auto", "item": "4", "status": "failed",
             "calls": [call(outcome="error", reason="transport", attempts=("http_500",) * 4)]},
            {"setup": "auto", "item": "4", "status": "completed", "calls": [call()]},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "answers.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            self.assertEqual(Ledger(path).done, {("panel:trio-glm", "1"), ("panel:trio-glm", "2"), ("auto", "3"), ("auto", "4")})
            self.assertEqual(Ledger(path, retry_failed=True).done, {("panel:trio-glm", "2"), ("auto", "4")})


if __name__ == "__main__":
    unittest.main()
