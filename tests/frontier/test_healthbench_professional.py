"""Offline protocol checks using synthetic text and unsent judge requests."""
from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from panda_bench.frontier.benchmarks import healthbench_professional as healthbench
from panda_bench.frontier.core import messages_digest


def example(source_id="synthetic-row", use_case="consult"):
    return {
        "id": source_id,
        "conversation": {"messages": [
            {"role": "system", "content": "Synthetic system instruction."},
            {"role": "developer", "content": "Synthetic developer instruction."},
            {"role": "user", "content": "Synthetic opening question."},
            {"role": "assistant", "content": "Synthetic prior response."},
            {"role": "user", "content": "  Synthetic final question.\n"},
        ]},
        "rubric_items": [
            {"points": 3, "criterion_text": "Synthetic first criterion."},
            {"points": 1, "criterion_text": "Synthetic second criterion."},
            {"points": -2, "criterion_text": "Synthetic undesirable criterion."},
        ],
        "use_case": use_case,
        "type": "good_faith",
        "difficulty": "typical",
        "specialty": "synthetic",
        "physician_response": "  Synthetic physician reference.\n",
    }


def fixture(*rows):
    return b"\n".join(json.dumps(row).encode("utf-8") for row in rows)


def judgments(*values):
    return {str(i): json.dumps({"explanation": "Private judge rationale.", "criteria_met": value})
            for i, value in enumerate(values)}


class HealthBenchProfessionalTests(unittest.TestCase):
    def setUp(self):
        self.bench = healthbench.BENCHMARK
        self.cache = Path("unused-in-memory-cache")
        self.row = example()
        with patch.object(healthbench, "fetch_pinned", return_value=fixture(self.row)):
            self.item, = self.bench.load(self.cache)

    def test_pin_source_order_ids_conversation_and_private_references(self):
        tail = example("second", "writing")
        absent = example("unanswered")
        del absent["physician_response"]
        with patch.object(healthbench, "fetch_pinned", return_value=fixture(self.row, tail, absent)) as fetch:
            items = self.bench.load(self.cache)
            references = healthbench.reference_answers(self.cache)
        self.assertEqual(fetch.call_count, 2)
        self.assertIn("/resolve/349962fd46dd02343a0d8a606491baf59154ea1a/", healthbench.DATA_URL)
        fetch.assert_called_with(healthbench.DATA_URL, self.cache / "healthbench_professional_eval.jsonl",
                                 "d44b08e6e952e04c945e2c406f02533d9e7a989a84e35820ee7efdff20c9e4e2")
        expected = ["hbp-" + hashlib.sha256(x.encode()).hexdigest()[:16]
                    for x in ("synthetic-row", "second", "unanswered")]
        self.assertEqual([item.id for item in items], expected)
        self.assertEqual(references, {key: self.row["physician_response"] for key in expected[:2]})
        messages = tuple((m["role"], m["content"]) for m in self.row["conversation"]["messages"])
        self.assertEqual(items[0].messages, messages)
        self.assertEqual(items[0].key["rubric_items"], self.row["rubric_items"])
        self.assertEqual(items[0].public, {"source_id": "synthetic-row", "category": "consult",
                                           "messages_sha256": messages_digest(messages)})
        self.assertEqual((self.bench.name, self.bench.max_output_tokens), ("healthbench-professional", 16384))

    def test_invalid_conversations_are_rejected_without_reordering(self):
        for turns in ([], [("assistant", "x")], [("user", "x"), ("system", "x"), ("user", "x")],
                      [("tool", "x"), ("user", "x")], [("user", {})]):
            with self.subTest(turns=turns):
                row = example()
                row["conversation"]["messages"] = [{"role": r, "content": t} for r, t in turns]
                with patch.object(healthbench, "fetch_pinned", return_value=fixture(row)):
                    with self.assertRaises(ValueError):
                        self.bench.load(self.cache)

    def test_official_template_and_each_criterion_prompt(self):
        expected_hash = "2adffd51fd259554ebcd036ad1072d4aa2b7ce3aec2bbffe36271f911632ed3c"
        self.assertEqual(hashlib.sha256(healthbench.GRADER_TEMPLATE.encode()).hexdigest(), expected_hash)
        answer = "Synthetic answer.\n"
        requests = self.bench.judge_requests(self.item, answer)
        transcript = ("system: Synthetic system instruction.\n\ndeveloper: Synthetic developer instruction."
                      "\n\nuser: Synthetic opening question.\n\nassistant: Synthetic prior response."
                      "\n\nuser:   Synthetic final question.\n\n\nassistant: Synthetic answer.\n")
        self.assertEqual([r.name for r in requests], ["0", "1", "2"])
        for request, criterion in zip(requests, self.row["rubric_items"]):
            rendered = healthbench.GRADER_TEMPLATE.replace("<<conversation>>", transcript).replace(
                "<<rubric_item>>", f"[{criterion['points']}] {criterion['criterion_text']}")
            self.assertEqual(request.user, rendered)
            self.assertIsNone(request.system)
            self.assertEqual(request.max_tokens, 1024)
            self.assertNotIn(self.row["physician_response"], request.user)
        self.assertIn("assistant: <<rubric_item>>", self.bench.judge_requests(self.item, "<<rubric_item>>")[0].user)

    def test_last_complete_object_handles_fences_prose_quotes_and_nested_objects(self):
        replies = judgments(False, False, False)
        last = {"criteria_met": True, "explanation": 'quote " and { }', "details": {"criteria_met": False}}
        replies["0"] = 'Preface {bad}. {"criteria_met": false}\n```json\n' + json.dumps(last) + '\n```'
        result = self.bench.grade(self.item, "x" * 2000, replies)
        self.assertEqual(result["criteria_met"], [True, False, False])
        self.assertEqual(result["failed_judge_calls"], 0)
        self.assertEqual(result["raw_score"], 0.75)
        self.assertEqual(result["length_adjusted_score"], 0.75)

    def test_invalid_or_missing_final_judge_verdict_fails_the_entire_task(self):
        for bad in (None, "", "bad JSON", "{}", "[]", '{"criteria_met":0}', '{"criteria_met":null}',
                    '{"criteria_met":"true"}', '{"criteria_met":true} {"explanation":"missing"}'):
            with self.subTest(reply=bad):
                replies = judgments(True, True, True)
                replies["2"] = bad
                grade = self.bench.grade(self.item, "short", replies)
                self.assertEqual(grade["criteria_met"], [True, True, None])
                self.assertEqual(grade["failed_judge_calls"], 1)
                self.assertFalse(grade["failed_answer"])
                self.assertIsNone(grade["answer_length"])
                self.assertEqual((grade["raw_score"], grade["length_adjusted_score"]), (0, 0))
        self.assertEqual(self.bench.grade(self.item, "short", {})["failed_judge_calls"], 3)

    def test_missing_or_blank_answer_has_no_judges_length_or_short_answer_bonus(self):
        for answer in (None, "", " \n"):
            with self.subTest(answer=answer):
                if answer is not None:
                    self.assertEqual(self.bench.judge_requests(self.item, answer), [])
                grade = self.bench.grade(self.item, answer, {})
                self.assertTrue(grade["failed_answer"])
                self.assertEqual(grade["failed_judge_calls"], 0)
                self.assertIsNone(grade["answer_length"])
                self.assertEqual((grade["raw_score"], grade["length_adjusted_score"]), (0, 0))

    def test_signed_weights_two_sided_length_and_aggregate_only_clipping(self):
        high = self.bench.grade(self.item, "é" * 1500, judgments(True, True, False))
        negative_row = example("negative", "research")
        # This second task has a different positive denominator. Each task gets one vote.
        negative_row["rubric_items"] = [{"points": 10, "criterion_text": "Reward."},
                                        {"points": -5, "criterion_text": "Penalty."}]
        with patch.object(healthbench, "fetch_pinned", return_value=fixture(negative_row)):
            negative_item, = self.bench.load(self.cache)
        low = self.bench.grade(negative_item, "z" * 3000, judgments(False, True))
        self.assertEqual((high["raw_score"], low["raw_score"]), (1.0, -0.5))
        self.assertAlmostEqual(high["length_adjusted_score"], 1.0147)
        self.assertAlmostEqual(low["length_adjusted_score"], -0.5294)
        score = self.bench.score([high, low])
        self.assertAlmostEqual(score["score"], 24.265)
        self.assertEqual(score["raw_score"], 25)
        self.assertEqual(score["mean_answer_length"], 2250)
        self.assertEqual(score["by_use_case"], {"consult": 100, "research": 0})
        self.assertEqual(score["by_type"], {"good_faith": 25})
        self.assertEqual(score["by_difficulty"], {"typical": 25})
        self.assertEqual(self.bench.score([high])["score"], 100)
        self.assertEqual(self.bench.score([low])["score"], 0)

    def test_failure_denominators_counts_and_publishable_outputs(self):
        answer = "Private synthetic answer."
        good = self.bench.grade(self.item, answer, judgments(True, False, False))
        failed_answer = self.bench.grade(self.item, None, {})
        failed_judges = self.bench.grade(self.item, answer, {"1": judgments(True)["0"]})
        result = self.bench.score([good, failed_answer, failed_judges])
        self.assertEqual((result["n"], result["failed_answers"], result["failed_judge_calls"]), (3, 1, 2))
        self.assertEqual(result["mean_answer_length"], len(answer))
        self.assertEqual(result["raw_score"], 25.0)
        self.assertEqual(result["by_type"]["good_faith"], 25.0)
        self.assertAlmostEqual(result["score"], 100 * good["length_adjusted_score"] / 3)
        self.assertIsNone(self.bench.score([failed_answer, failed_judges])["mean_answer_length"])
        self.assertEqual(self.bench.score([])["n"], 0)
        published = json.dumps([self.item.public, good, failed_answer, failed_judges, result], allow_nan=False)
        private = ([m["content"] for m in self.row["conversation"]["messages"]]
                   + [c["criterion_text"] for c in self.row["rubric_items"]]
                   + [answer, self.row["physician_response"], "Private judge rationale."])
        for text in private:
            self.assertNotIn(text.strip(), published)

    def test_zero_positive_denominator_is_rejected(self):
        row = example()
        row["rubric_items"] = [{"points": -1, "criterion_text": "Negative only."}]
        with patch.object(healthbench, "fetch_pinned", return_value=fixture(row)):
            item, = self.bench.load(self.cache)
        with self.assertRaisesRegex(ValueError, "positive criterion"):
            self.bench.grade(item, "answer", judgments(True))
