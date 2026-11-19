"""Offline release checks: a synthetic run whose every text field carries a marker, no network.

The release must carry none of the markers (question, rubric, reference, answer, judge reply,
member answers, error text, unknown nested fields), must refuse runs it cannot vouch for, and
must never leave a half-written release behind.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from panda_bench.frontier import judge as judging
from panda_bench.frontier import release
from panda_bench.frontier.core import Item, JudgeRequest, sha256_text

MARKERS = {
    "question": "QMARK-7f3a question text",
    "rubric": "RMARK-19c2 rubric line",
    "reference": "REFMARK-55d0",
    "answer": "AMARK-0b91 the model's answer",
    "judge": "JMARK-e4a7 the judge's reasoning",
    "member": "MMARK-3c10 a member's draft",
    "error": "EMARK-a2f4 a harness error",
    "nested": "NMARK-6d2e an unknown nested field",
}


class FakeBenchmark:
    name = "fake"
    max_output_tokens = 100

    def load(self, cache_dir):
        return [Item(id=f"f-{n}", messages=(("user", MARKERS["question"] + f" {n}"),),
                     key={"rubric": MARKERS["rubric"], "reference": MARKERS["reference"]},
                     public={"source_id": f"s{n}", "category": "c", "messages_sha256": "d" * 64}) for n in (1, 2)]

    def judge_requests(self, item, answer):
        return [JudgeRequest(name="grade", user=item.key["rubric"] + answer)] if answer else []

    def grade(self, item, answer, judged):
        reply = judged.get("grade") if answer else None
        return {"correct": bool(reply and reply.startswith("YES")), "failed_answer": answer is None,
                "failed_judge_calls": int(bool(answer) and reply is None)}

    def score(self, grades):
        return {"score": 100.0 * sum(g["correct"] for g in grades) / len(grades) if grades else None}


BENCH = FakeBenchmark()
SETUPS = [{"name": "single", "kind": "single", "models": ["m"]},
          {"name": "panel:p", "kind": "panel", "models": ["m", "n"], "synthesizer": "m"}]
PLAN = {"at": "2026-11-18T16:00:00Z", "limit": None, "judge": "claude-sonnet-5", "judge_settings": "s1",
        "harness": {"commit": "a" * 40, "dirty_files": 0}, "product_source": {"commit": "b" * 40, "dirty_files": 0},
        "benchmarks": {"fake": {"setups": SETUPS, "max_output_tokens": 100, "pins": {"DATA_REVISION": "c" * 40}}}}


def rows_of(path: Path):
    return [json.loads(line) for line in path.read_text().split("\n") if line]


def call(role, outcome="text", **extra):
    return {"model": "m", "role": role, "outcome": outcome, "reason": "stop", "finish": "stop", "verdict": "completed",
            "attempts": ["ok"], "seconds": 1.5, "wall_seconds": 2.0, "input_tokens": 10, "cached_input_tokens": 0,
            "output_tokens": 5, "list_cost_nanos": 1000, **extra}


def answer(setup, item, text, **extra):
    return {"benchmark": "fake", "setup": setup, "item": item, "status": "completed", "text": text, "cut": False,
            "run_id": "r", "at": "2026-11-18T17:00:00Z", "list_cost_nanos": 3000, "wall_seconds": 4.0,
            "max_output_tokens": 100, "setup_spec": {"name": setup, "secret": MARKERS["nested"]},
            "calls": [call("single", leak=MARKERS["nested"])], **extra}


def build_run(root: Path, judge_text=None, plan_rows=(PLAN,)) -> Path:
    run = root / "run"
    (run / "answers").mkdir(parents=True)
    (run / "judge").mkdir()
    (run / "plan.jsonl").write_text("".join(json.dumps(p) + "\n" for p in plan_rows))
    texts = {("single", "f-1"): MARKERS["answer"] + " 1", ("single", "f-2"): MARKERS["answer"] + " 2",
             ("panel:p", "f-1"): MARKERS["answer"] + " 3", ("panel:p", "f-2"): MARKERS["answer"] + " 4"}
    rows = [
        # A harness error first, then the redone answer: the newest row wins and both are counted.
        {"benchmark": "fake", "setup": "single", "item": "f-1", "status": "harness_error", "text": None, "calls": [],
         "error": MARKERS["error"], "at": "2026-11-18T16:30:00Z"},
        answer("single", "f-1", texts[("single", "f-1")], route={"pick": "m", "task": "t", "classification": "classified",
                                                              "mode": "balanced", "prompt": MARKERS["nested"]}),
        answer("single", "f-2", texts[("single", "f-2")]),
        answer("panel:p", "f-1", texts[("panel:p", "f-1")],
               panel={"members": ["m", "n"], "synthesizer": "m", "answered": [1, 2], "draft_order": [2, 1],
                      "mode": "synthesis", "synthesis_verdict": "completed", "drafts": [MARKERS["member"]]}),
        answer("panel:p", "f-2", texts[("panel:p", "f-2")]),
    ]
    (run / "answers" / "fake.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    judged = []
    for (setup, item), text in texts.items():
        cid = judging.custom_id("fake", setup, item, "grade", sha256_text(text))
        judged.append({"custom_id": cid, "benchmark": "fake", "setup": setup, "item": item, "name": "grade",
                       "answer_sha256": sha256_text(text), "batch": "msgbatch_x", "judge": "claude-sonnet-5",
                       "judge_settings": "s1", "result": "succeeded", "stop_reason": "end_turn",
                       "text": judge_text if judge_text is not None else "YES " + MARKERS["judge"],
                       "usage": {"input_tokens": 50, "output_tokens": 9, "note": MARKERS["nested"]}})
    (run / "judge" / "fake.jsonl").write_text("".join(json.dumps(r) + "\n" for r in judged))
    return run


def build_live(root: Path) -> Path:
    live = root / "live"
    (live / "answers").mkdir(parents=True)
    (live / "judge").mkdir()
    text = MARKERS["answer"] + " live"
    row = {"benchmark": "fake", "setup": "live:panel:p", "item": "f-1", "backend": "live", "status": "completed",
           "text": text, "model": "panel:p", "finish": "stop", "charge_cents": 1, "attempts": ["ok"],
           "idempotency_key": "k", "wall_seconds": 9.0, "usage": {"prompt_tokens": 10, "completion_tokens": 5},
           "panda": {"status": "completed", "mode": "panel", "route": None,
                     "panel": {"name": "p", "revision": "r1", "synthesizer_model": "m", "prompt": MARKERS["nested"]},
                     "agents": [{"index": 1, "model": "m", "status": "completed", "output_text": MARKERS["member"],
                                 "error": {"code": "x", "message": MARKERS["error"]}}],
                     "synthesis": {"model": "m", "status": "completed", "included_agents": [1], "error": None}}}
    (live / "answers" / "fake.jsonl").write_text(json.dumps(row) + "\n")
    cid = judging.custom_id("fake", "live:panel:p", "f-1", "grade", sha256_text(text))
    (live / "judge" / "fake.jsonl").write_text(json.dumps(
        {"custom_id": cid, "setup": "live:panel:p", "item": "f-1", "name": "grade", "text": "YES " + MARKERS["judge"],
         "result": "succeeded", "judge_settings": "s1"}) + "\n")
    return live


class ReleaseTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        for target in ("panda_bench.frontier.release.load_benchmark", "panda_bench.frontier.score.load_benchmark"):
            patcher = patch(target, return_value=BENCH)
            patcher.start()
            self.addCleanup(patcher.stop)

    def tearDown(self):
        self.tmp.cleanup()

    def release(self, run, **kwargs):
        return release.write_release(run, self.root / "releases" / "frontier-test", ["fake"], **kwargs)

    def published(self) -> str:
        out = self.root / "releases" / "frontier-test"
        return "\n".join(p.read_text() for p in sorted(out.rglob("*")) if p.is_file())

    def test_release_carries_no_marker(self):
        manifest = self.release(build_run(self.root), live_dir=build_live(self.root),
                                notes={"method": "the publisher's own prose"}, extra={"grader-check.json": {"x": 1}})
        text = self.published()
        for kind, marker in MARKERS.items():
            self.assertNotIn(marker.split()[0], text, kind)
        self.assertEqual(manifest["benchmarks"]["fake"]["redone"], {"single": 1})
        self.assertEqual(manifest["benchmarks"]["fake"]["judge_replies"], 4)
        self.assertTrue(manifest["benchmarks"]["fake"]["complete"])
        results = json.loads((self.root / "releases" / "frontier-test" / "fake" / "results.json").read_text())
        self.assertEqual(results["single"]["score"], 100.0)
        live = json.loads((self.root / "releases" / "frontier-test" / "live" / "results.json").read_text())
        self.assertEqual(live["fake"]["panel:p"], {"items": 1, "statuses": {"completed": 1}, "charge_cents": 1,
                                                   "paired_items": 1, "live_score": 100.0, "replay_score": 100.0})
        answers = rows_of(self.root / "releases" / "frontier-test" / "fake" / "answers.jsonl")
        first = next(a for a in answers if (a["setup"], a["item"]) == ("single", "f-1"))
        self.assertEqual(first["answer_sha256"], sha256_text(MARKERS["answer"] + " 1"))
        self.assertEqual(first["rows_written"], 2)
        self.assertEqual(first["retail_cents"], 1)

    def test_notes_cannot_replace_fixed_fields(self):
        manifest = self.release(build_run(self.root), notes={"judge": "someone else", "schema": "v0"})
        self.assertEqual(manifest["judge"], "claude-sonnet-5")
        self.assertEqual(manifest["schema"], release.SCHEMA)
        self.assertEqual(manifest["notes"]["judge"], "someone else")

    def test_long_text_in_an_allowed_field_is_refused(self):
        run = build_run(self.root)
        path = run / "answers" / "fake.jsonl"
        rows = rows_of(path)
        rows[1]["route"]["task"] = "x" * (release.MAX_STRING + 1)
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        with self.assertRaises(release.ReleaseError):
            self.release(run)

    def test_local_path_is_refused(self):
        run = build_run(self.root)
        path = run / "answers" / "fake.jsonl"
        rows = rows_of(path)
        rows[2]["calls"][0]["model"] = "/Users/someone/model"
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        with self.assertRaises(release.ReleaseError):
            self.release(run)

    def test_changed_design_is_refused(self):
        changed = json.loads(json.dumps(PLAN))
        changed["benchmarks"]["fake"]["max_output_tokens"] = 200
        with self.assertRaises(release.ReleaseError):
            self.release(build_run(self.root, plan_rows=(PLAN, changed)))

    def test_pilot_start_is_refused(self):
        with self.assertRaises(release.ReleaseError):
            self.release(build_run(self.root, plan_rows=(dict(PLAN, limit=5),)))

    def test_stray_setup_is_refused(self):
        run = build_run(self.root)
        with (run / "answers" / "fake.jsonl").open("a") as handle:
            handle.write(json.dumps(answer("ref:other", "f-1", "x")) + "\n")
        with self.assertRaises(release.ReleaseError):
            self.release(run)

    def test_incomplete_run_is_refused_and_old_release_survives(self):
        self.release(build_run(self.root))
        before = self.published()
        (self.root / "run2").mkdir()
        run = build_run(self.root / "run2", judge_text="")
        # An unusable judge reply leaves the scores incomplete.
        with patch.object(BENCH, "grade", lambda item, answer, judged: {"correct": False, "failed_answer": False,
                                                                        "failed_judge_calls": 1}):
            with self.assertRaises(release.ReleaseError):
                self.release(run)
        self.assertEqual(self.published(), before)
        leftovers = [p.name for p in (self.root / "releases").iterdir() if p.name != "frontier-test"]
        self.assertEqual(leftovers, [])

    def test_incomplete_run_can_be_released_on_request(self):
        run = build_run(self.root)
        path = run / "judge" / "fake.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in rows_of(path) if r["setup"] != "panel:p"))
        manifest = self.release(run, allow_incomplete=True)
        self.assertEqual(manifest["benchmarks"]["fake"]["unfinished"], ["panel:p"])

    def test_check_release_recomputes_scores_and_digests(self):
        self.release(build_run(self.root), live_dir=build_live(self.root))
        out = self.root / "releases" / "frontier-test"
        self.assertEqual(release.check_release(out), [])
        grades = out / "fake" / "grades.jsonl"
        rows = rows_of(grades)
        rows[0]["correct"] = False
        grades.write_text("".join(json.dumps(r) + "\n" for r in rows))
        problems = release.check_release(out)
        self.assertIn("fake/grades.jsonl: SHA-256 differs from the manifest", problems)
        self.assertIn(f"fake/{rows[0]['setup']}: score is 100.0, its grades give 50.0", problems)


if __name__ == "__main__":
    unittest.main()
