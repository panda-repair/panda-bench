"""Offline SimpleQA Verified checks: synthetic rows, unsent judge requests, no network."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import string
import unittest
from pathlib import Path
from unittest.mock import patch

from panda_bench.frontier.benchmarks import simpleqa_verified as sqa
from panda_bench.frontier.core import Item, messages_digest

COLUMNS = ["original_index", "problem", "answer", "topic", "answer_type", "multi_step", "requires_reasoning", "urls"]
SECRET_MARKERS = ("Synthetic question", "SYNTHETIC-GOLD", "{braces}")


def row(index, topic="Art"):
    return {
        "original_index": str(index),
        "problem": f"Synthetic question {index} with {{braces}}?",
        "answer": f"SYNTHETIC-GOLD {index} (acceptable range: anything between 1 and 2)",
        "topic": topic,
        "answer_type": "Number",
        "multi_step": "False",
        "requires_reasoning": "True",
        "urls": "['https://example.invalid/a', 'https://example.invalid/b']",
    }


def csv_bytes(rows):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


class SimpleQAVerifiedTests(unittest.TestCase):
    def setUp(self):
        self.bench = sqa.BENCHMARK
        self.cache = Path("unused-in-memory-cache")
        self.item = Item(
            id="sqav-synthetic",
            messages=(("user", "Synthetic question {target}?"),),
            key={"question": "Synthetic question {target}?", "target": "SYNTHETIC-GOLD"},
        )

    def load(self, rows):
        with patch.object(sqa, "fetch_pinned", return_value=csv_bytes(rows)) as fetch:
            items = self.bench.load(self.cache)
        fetch.assert_called_once_with(sqa.DATA_URL, self.cache / "simpleqa_verified.csv", sqa.DATA_SHA256)
        return items

    def grade(self, answer, reply):
        judged = {} if reply is ... else {"grade": reply}
        return self.bench.grade(self.item, answer, judged)

    def test_pins_and_settings(self):
        self.assertEqual((self.bench.name, self.bench.max_output_tokens), ("simpleqa-verified", 16384))
        self.assertEqual(sqa.DATA_URL, "https://huggingface.co/datasets/google/simpleqa-verified/resolve/"
                                       "0dc97e0d28d8233463e005cdc4475cc2a13ba2dc/simpleqa_verified.csv")
        self.assertEqual(sqa.DATA_SHA256, "b5db21155444763543fe31b67e7cf28ce2bb225742a5b889421c5f182e2f92f5")

    def test_grader_template_is_the_notebooks_text(self):
        template = sqa.GRADER_TEMPLATE
        # SHA-256 of the string the notebook builds in cell In[16], Version 9 (see the doc).
        self.assertEqual(hashlib.sha256(template.encode("utf-8")).hexdigest(),
                         "84c004ec4fcf8f0703bb0d734544036a72e847bfa7116429e5aee5e53ccc8cf3")
        self.assertEqual((len(template), template.count("\n") + 1), (7125, 84))
        self.assertTrue(template.startswith('Your job is to look at a question, a gold target, and a predicted '
                                            'answer, and then assign a grade of either ["CORRECT", "INCORRECT", '
                                            '"NOT_ATTEMPTED"].\n  First, I will give examples'))
        self.assertIn("  ```\n  Question: {question}\n  Gold target: {target}\n  Predicted answer: "
                      "{predicted_answer}\n  ```\n", template)
        self.assertTrue(template.endswith('  A: CORRECT\n  B: INCORRECT\n  C: NOT_ATTEMPTED\n\n  Just return '
                                          'the letters "A", "B", or "C", with no text around it.'))
        fields = [field for _, field, _, _ in string.Formatter().parse(template) if field is not None]
        self.assertEqual(fields, ["question", "target", "predicted_answer"])

    def test_load_sends_the_bare_question_and_keeps_text_private(self):
        rows = [row(5, "Politics"), row(8), row(4319, "Sports")]
        items = self.load(rows)
        self.assertEqual([item.id for item in items],
                         ["sqav-" + hashlib.sha256(r["original_index"].encode()).hexdigest()[:16] for r in rows])
        for source, item in zip(rows, items):
            self.assertEqual(item.messages, (("user", source["problem"]),))
            self.assertEqual(item.key, {"question": source["problem"], "target": source["answer"]})
            self.assertEqual(item.public, {
                "source_id": source["original_index"],
                "category": source["topic"],
                "answer_type": "Number",
                "messages_sha256": messages_digest(item.messages),
            })

    def test_one_judge_request_with_the_template_filled_once(self):
        answer = "  It is {target} and {question}.\nSecond line "
        request, = self.bench.judge_requests(self.item, answer)
        head, rest = sqa.GRADER_TEMPLATE.split("{question}")
        middle, rest = rest.split("{target}")
        before_answer, tail = rest.split("{predicted_answer}")
        expected = head + "Synthetic question {target}?" + middle + "SYNTHETIC-GOLD" + before_answer + answer + tail
        self.assertEqual((request.name, request.system, request.user), ("grade", None, expected))
        self.assertEqual(self.bench.judge_requests(self.item, " \n"), [])
        self.assertEqual(self.bench.judge_requests(self.item, ""), [])

    def test_parse_grade_matches_the_notebook_including_its_quirks(self):
        cases = {
            "A": "A", "B": "B", "C": "C", " B\n": "B", "**C**": "C", "The answer is B": "B",
            "CORRECT": "C",  # first capital A/B/C wins: the C of CORRECT
            "INCORRECT": "C",
            "NOT_ATTEMPTED": "A",  # the A of ATTEMPTED
            "correct": "A", "incorrect": "A",  # "CORRECT" is tested before "INCORRECT"
            "not_attempted": "C",
            "xyz": None, "": None, "a": None, "D": None,  # the notebook would default these to C
        }
        for reply, letter in cases.items():
            with self.subTest(reply=reply):
                self.assertEqual(sqa.parse_grade(reply), letter)

    def test_grade_paths(self):
        def expect(letter, verdict, failed_answer=False, failed_calls=0, bare=True):
            return {"grade_letter": letter, "verdict": verdict, "correct": verdict == "CORRECT",
                    "attempted": verdict in ("CORRECT", "INCORRECT"), "failed_answer": failed_answer,
                    "failed_judge_calls": failed_calls, "bare_judge_reply": bare}

        cases = [
            (("Paris", "A"), expect("A", "CORRECT")),
            (("Lyon", "B\n"), expect("B", "INCORRECT")),
            (("I don't know.", " C "), expect("C", "NOT_ATTEMPTED")),
            (("Paris", "Grade: A"), expect("A", "CORRECT", bare=False)),
            (("Paris", None), expect(None, None, failed_calls=1, bare=None)),  # failed judge call
            (("Paris", ...), expect(None, None, failed_calls=1, bare=None)),  # judge reply missing
            (("Paris", "unsure"), expect(None, None, failed_calls=1, bare=None)),  # unparseable reply
            ((None, ...), expect(None, "NOT_ATTEMPTED", failed_answer=True, bare=None)),  # no answer
            ((" \n", ...), expect(None, "NOT_ATTEMPTED", failed_answer=True, bare=None)),
        ]
        for (answer, reply), expected in cases:
            with self.subTest(answer=answer, reply=reply):
                grade = self.grade(answer, reply)
                self.assertEqual(grade, expected)
                self.assertEqual(json.loads(json.dumps(grade)), grade)

    def test_score_arithmetic_by_hand(self):
        replies = ["A", "A", "A", "A", "B", "B", "C", "C"]
        grades = [self.grade("an answer", reply) for reply in replies]
        grades += [self.grade(None, ...), self.grade("an answer", None)]
        grades[0] = self.grade("an answer", "Grade: A")
        result = self.bench.score(grades)
        # n = 10: 4 correct, 2 incorrect, 2 + 1 not attempted, 1 failed judge call.
        # accuracy 4/10, given attempted 4/6, F1 = 2(0.4)(2/3) / (0.4 + 2/3) = 0.5.
        self.assertAlmostEqual(result.pop("score"), 50.0)
        self.assertAlmostEqual(result.pop("accuracy"), 40.0)
        self.assertAlmostEqual(result.pop("correct_given_attempted"), 200.0 / 3)
        self.assertAlmostEqual(result.pop("attempted_rate"), 60.0)
        self.assertEqual(result, {"n": 10, "correct": 4, "incorrect": 2, "not_attempted": 3, "failed_answers": 1,
                                  "failed_judge_calls": 1, "nonbare_judge_replies": 1})

    def test_score_reproduces_a_published_row_and_edge_cases(self):
        # arXiv:2509.07968 Table 7, Gemini 2.5 Pro: accuracy 55.3, attempted 98.9, given attempted 55.9, F1 55.6.
        grades = ([self.grade("x", "A")] * 553 + [self.grade("x", "B")] * 436 + [self.grade("x", "C")] * 11)
        result = self.bench.score(grades)
        self.assertEqual([round(result[k], 1) for k in ("accuracy", "attempted_rate", "correct_given_attempted",
                                                         "score")], [55.3, 98.9, 55.9, 55.6])
        self.assertEqual(self.bench.score([self.grade("x", "C"), self.grade(None, ...)])["score"], 0.0)
        self.assertEqual(self.bench.score([self.grade("x", "A")])["score"], 100.0)
        empty = self.bench.score([])
        self.assertEqual((empty["score"], empty["n"], empty["attempted_rate"]), (0.0, 0, 0.0))

    def test_public_fields_and_grades_carry_no_question_or_answer_text(self):
        items = self.load([row(5), row(13)])
        for item in items:
            published = [json.dumps(item.public)]
            for answer, reply in (("SYNTHETIC-GOLD echoed", "A"), (None, ...), ("x", None), ("x", "unsure")):
                published.append(json.dumps(self.bench.grade(item, answer, {} if reply is ... else {"grade": reply})))
            published.append(json.dumps(self.bench.score([self.bench.grade(item, "x", {"grade": "B"})])))
            for text in published:
                for marker in SECRET_MARKERS:
                    self.assertNotIn(marker, text)


if __name__ == "__main__":
    unittest.main()
