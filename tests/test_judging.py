import contextlib
import copy
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from panda_bench import judging
from panda_bench.storage import ROOT, read_json, read_rows


def fixture():
    manifest = {"stage": "stage-4", "schema": "panda-bench.run/v1", "prompts": [{"id": "synthetic-1", "prompt_utf8_bytes": 5, "checklist_utf8_bytes": 2}],
        "judge": {"model": "claude-opus-5-5", "system": "Judge fairly.", "schema": {"type": "object"}, "effort": "medium", "max_tokens": 16000,
                  "rates_usd_per_million": {"input": 2, "output": 10}, "shuffle_seed_template": "stage4:{prompt_id}", "bundle_before_shuffle": {"synthetic-1": ["one", "two"]}}}
    answers = [{"prompt_id": "synthetic-1", "option": x, "text": x, "status": "ok", "grade": None} for x in ("one", "two")]
    return manifest, answers


class JudgeTests(unittest.TestCase):
    def test_full_archive_bundle_and_plan_no_keys(self):
        manifest = read_json(ROOT / "releases" / "stage-4" / "manifest.json")
        answers = read_rows(ROOT / "releases" / "stage-4" / "answers.jsonl")
        plan = judging.judge_plan(manifest, answers, "claude-opus-5-5")
        self.assertEqual(plan["request_count"], 150)
        self.assertEqual(plan["answer_count"], 1493)
        self.assertEqual(sorted(len(r["options"]) for r in plan["requests"]), [5, 9, 9] + [10] * 147)
        self.assertEqual(plan["missing_answers"], [])
        self.assertEqual(plan["output_allowance_usd"], 24)
        self.assertGreater(plan["conservative_cost_usd"], 24)
        self.assertEqual(plan["historical_reference_usd"], 10.07)
        self.assertEqual(plan, judging.judge_plan(manifest, answers, "claude-opus-5-5"))

    def test_request_settings_and_parser_preserve_history(self):
        manifest, answers = fixture()
        plan = judging.judge_plan(manifest, answers, "claude-opus-5-5")
        requests, mappings = judging.build_requests(manifest, answers, {"synthetic-1": {"prompt": "Hello", "checklist": ["Be brief"]}}, plan)
        params = requests[0]["params"]
        self.assertEqual(params["max_tokens"], 16000)
        self.assertEqual(params["output_config"]["effort"], "medium")
        self.assertIn("<user_request>\nHello\n</user_request>", params["messages"][0]["content"])
        self.assertIn("<checklist>\n- Be brief\n</checklist>", params["messages"][0]["content"])
        self.assertEqual(judging.parse_grades('{"grades":[{"label":"A","score":12},{"label":"B","score":-1},{"label":"Q","score":5}]}', {"A": "one", "B": "two"}), {"one": 10, "two": 1})
        self.assertEqual(judging.parse_grades('{"grades":[{"label":"A","score":"7"}]}', {"A": "one", "B": "two"}), {"one": 7})

    def test_refusal_excludes_prompt_and_incomplete_stream_blocks(self):
        mappings = {"safe-id": {"prompt_id": "synthetic-1", "labels": {"A": "one"}}}
        results = [{"custom_id": "safe-id", "result": {"type": "succeeded", "message": {"stop_reason": "refusal", "usage": {"input_tokens": 12, "output_tokens": 2}}}}]
        rows, excluded, usage = judging.collect_grades(results, mappings, [{"prompt_id": "synthetic-1", "option": "one", "status": "ok"}])
        self.assertIsNone(rows[0]["grade"])
        self.assertEqual(excluded[0]["prompt_id"], "synthetic-1")
        self.assertEqual(usage, {"input_tokens": 12, "output_tokens": 2})
        with self.assertRaises(ValueError):
            judging.collect_grades([], mappings, [])

    def test_budget_refuses_before_fetch_or_sdk(self):
        args = SimpleNamespace(source="stage-4", command="regrade", judge="claude-opus-5-5", max_cost_usd=15, plan=False)
        with patch("panda_bench.datasets.load_prompts", side_effect=AssertionError("No dataset/network needed")), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, "exceeds"):
                judging.command(args)

    def test_fake_batch_writes_only_whitelisted_state(self):
        manifest, answers = fixture()
        plan = judging.judge_plan(manifest, answers, "claude-opus-5-5")
        requests, mappings = judging.build_requests(manifest, answers, {"synthetic-1": {"prompt": "Hello", "checklist": []}}, plan)
        labels = mappings[requests[0]["custom_id"]]["labels"]
        response = {"custom_id": requests[0]["custom_id"], "result": {"type": "succeeded", "message": {"stop_reason": "end_turn", "usage": {"input_tokens": 10, "output_tokens": 8}, "content": [{"type": "text", "text": json.dumps({"grades": [{"label": l, "score": 8} for l in labels]})}]}}}
        remote = SimpleNamespace(messages=SimpleNamespace(batches=SimpleNamespace(create=lambda **kw: SimpleNamespace(id="never-persist-this", processing_status="ended"), results=lambda _: [response])))
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            judging.execute_batch(requests, mappings, manifest, answers, output, client=remote)
            self.assertEqual(read_json(output / "judge-state.json")["state"], "completed")
            self.assertNotIn("never-persist-this", "".join(p.read_text() for p in output.iterdir()))
            self.assertEqual(len(read_rows(output / "grades.jsonl")), 2)

    def test_unknown_override_refuses_unpriced_budget(self):
        manifest, answers = fixture()
        with self.assertRaisesRegex(ValueError, "rate table"):
            judging.judge_plan(manifest, answers, "unpriced-model")


if __name__ == "__main__":
    unittest.main()
