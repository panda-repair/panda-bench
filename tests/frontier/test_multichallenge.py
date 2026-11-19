"""Offline tests for the MultiChallenge module: synthetic rows only, and no network."""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from panda_bench.frontier import core
from panda_bench.frontier.benchmarks import multichallenge as mc

try:
    import pyarrow  # noqa: F401
    HAVE_PYARROW = True
except ImportError:
    HAVE_PYARROW = False

IR, IM, RVE, SC = mc.AXES

# Every text carries a "zq-" marker so the leak tests can search for it.
ROWS = [
    {"QUESTION_ID": "syn-ir", "AXIS": IR, "PASS_CRITERIA": "YES",
     "TARGET_QUESTION": "zq-rubric-ir: Is every line of the response lowercase?",
     "CONVERSATION": [
         {"role": "user", "content": "  zq-user-ir-1: write only in lowercase from now on.\n"},
         {"role": "assistant", "content": "zq-assistant-ir: understood."},
         {"role": "user", "content": "zq-user-ir-2: describe a {braced} lighthouse."}]},
    {"QUESTION_ID": "syn-im", "AXIS": IM, "PASS_CRITERIA": "YES",
     "TARGET_QUESTION": "zq-rubric-im: Does the response avoid recipes with peanuts?",
     "CONVERSATION": [{"role": "user", "content": "zq-user-im: my guest cannot eat peanuts. Dessert ideas?"}]},
    {"QUESTION_ID": "syn-rve", "AXIS": RVE, "PASS_CRITERIA": "YES",
     "TARGET_QUESTION": "zq-rubric-rve: Does the edited draft keep the second paragraph unchanged?",
     "CONVERSATION": [
         {"role": "user", "content": "zq-user-rve-1: draft two paragraphs."},
         {"role": "assistant", "content": "zq-assistant-rve: first paragraph.\n\nsecond paragraph."},
         {"role": "user", "content": "zq-user-rve-2: shorten only the first paragraph."}]},
    {"QUESTION_ID": "syn-sc", "AXIS": SC, "PASS_CRITERIA": "NO",
     "TARGET_QUESTION": "zq-rubric-sc: Does the response drop its earlier advice?",
     "CONVERSATION": [
         {"role": "user", "content": "zq-user-sc-1: which step comes after wifi?"},
         {"role": "assistant", "content": "zq-assistant-sc: register the device next."},
         {"role": "user", "content": "zq-user-sc-2: so I just pick a book now, right?"}]},
]
COUNTS = {IR: 1, IM: 1, RVE: 1, SC: 1}


def github_bytes(rows):
    return b"".join(json.dumps(row).encode("utf-8") + b"\n" for row in rows)


def hf_bytes(rows, num_turns=None):
    import pyarrow as pa
    import pyarrow.parquet as pq

    conversation = pa.struct([("role", pa.list_(pa.string())), ("content", pa.list_(pa.string()))])
    table = pa.table({
        "question_id": pa.array([r["QUESTION_ID"] for r in rows], pa.string()),
        "axis": pa.array([r["AXIS"] for r in rows], pa.string()),
        "conversation": pa.array([
            {"role": [m["role"] for m in r["CONVERSATION"]], "content": [m["content"] for m in r["CONVERSATION"]]}
            for r in rows], conversation),
        "target_question": pa.array([r["TARGET_QUESTION"] for r in rows], pa.string()),
        "pass_criteria": pa.array([r["PASS_CRITERIA"] for r in rows], pa.string()),
        "num_turns": pa.array(num_turns or [len(r["CONVERSATION"]) for r in rows], pa.int32()),
    })
    sink = io.BytesIO()
    pq.write_table(table, sink)
    return sink.getvalue()


@contextlib.contextmanager
def offline_cache(github=None, hf=None, responses=()):
    """A temporary cache holding synthetic files, with the pins patched to their digests and
    any download attempt turned into a test failure."""
    def no_network(*args, **kwargs):
        raise AssertionError("a test tried to download")

    with tempfile.TemporaryDirectory() as tmp, contextlib.ExitStack() as stack:
        root = Path(tmp)
        stack.enter_context(patch("urllib.request.urlopen", no_network))
        stack.enter_context(patch.dict(mc.BENCHMARK.axis_counts, COUNTS))
        stack.enter_context(patch.dict(mc.REFERENCE_BENCHMARK.axis_counts, COUNTS))

        def put(relative, data):
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            return hashlib.sha256(data).hexdigest()

        if github is not None:
            stack.enter_context(patch.object(mc, "GITHUB_SHA256", put("github/" + mc.GITHUB_FILE, github)))
        if hf is not None:
            stack.enter_context(patch.object(mc, "HF_SHA256", put("huggingface/" + mc.HF_FILE, hf)))
        files = tuple(
            (model, model + ".jsonl", put("github/data/final_model_responses/" + model + ".jsonl", data))
            for model, data in responses)
        stack.enter_context(patch.object(mc, "REFERENCE_FILES", files))
        yield root


def load_github(rows, counts=None):
    with offline_cache(github=github_bytes(rows)) as root, patch.dict(mc.REFERENCE_BENCHMARK.axis_counts, counts or COUNTS):
        return mc.REFERENCE_BENCHMARK.load(root)


def judge_reply(verdict, reasoning="zq-judge-reasoning: the response quotes zq-user-ir-1."):
    return json.dumps({"reasoning": reasoning, "verdict": verdict})


class LoadTests(unittest.TestCase):
    def test_benchmarks_expose_the_protocol(self):
        self.assertEqual(mc.BENCHMARK.name, "multichallenge")
        self.assertEqual(mc.REFERENCE_BENCHMARK.name, "multichallenge-2025")
        for bench in (mc.BENCHMARK, mc.REFERENCE_BENCHMARK):
            self.assertEqual(bench.max_output_tokens, 16384)
            for method in ("load", "judge_requests", "grade", "score"):
                self.assertTrue(callable(getattr(bench, method)))
        self.assertEqual(sum(mc.HF_AXIS_COUNTS.values()), 266)
        self.assertEqual(sum(mc.GITHUB_AXIS_COUNTS.values()), 273)

    def test_messages_are_the_conversation_verbatim(self):
        items = load_github(ROWS)
        self.assertEqual([item.key["source_id"] for item in items], [row["QUESTION_ID"] for row in ROWS])
        for item, row in zip(items, ROWS):
            self.assertEqual(item.messages, tuple((m["role"], m["content"]) for m in row["CONVERSATION"]))
            self.assertEqual(item.messages[-1][0], "user")
            self.assertEqual(item.id, "mc25-" + hashlib.sha256(row["QUESTION_ID"].encode()).hexdigest()[:16])
            self.assertEqual(dict(item.key), {"source_id": row["QUESTION_ID"], "axis": row["AXIS"],
                                              "target_question": row["TARGET_QUESTION"], "pass_criteria": row["PASS_CRITERIA"]})
            self.assertEqual(dict(item.public), {
                "source_id": row["QUESTION_ID"],
                "category": row["AXIS"],
                "messages_sha256": core.messages_digest(item.messages),
                "rubric_sha256": core.sha256_text(row["TARGET_QUESTION"]),
            })
        self.assertEqual(items[0].messages[0][1], "  zq-user-ir-1: write only in lowercase from now on.\n")

    @unittest.skipUnless(HAVE_PYARROW, "pyarrow is needed to write the synthetic parquet file")
    def test_current_release_reads_the_parquet_layout(self):
        with offline_cache(hf=hf_bytes(ROWS)) as root:
            current = mc.BENCHMARK.load(root)
        reference = load_github(ROWS)
        self.assertEqual([i.messages for i in current], [i.messages for i in reference])
        self.assertEqual([dict(i.key) for i in current], [dict(i.key) for i in reference])
        self.assertEqual([i.id[len("mc-"):] for i in current], [i.id[len("mc25-"):] for i in reference])
        self.assertTrue(all(i.id.startswith("mc-") for i in current))

    @unittest.skipUnless(HAVE_PYARROW, "pyarrow is needed to write the synthetic parquet file")
    def test_current_release_checks_num_turns(self):
        with offline_cache(hf=hf_bytes(ROWS, num_turns=[3, 1, 3, 4])) as root:
            with self.assertRaises(ValueError):
                mc.BENCHMARK.load(root)

    def test_leading_system_message_is_kept_first(self):
        row = dict(ROWS[1], CONVERSATION=[{"role": "system", "content": "zq-system"}] + ROWS[1]["CONVERSATION"])
        items = load_github([ROWS[0], row, ROWS[2], ROWS[3]])
        self.assertEqual(items[1].messages[0], ("system", "zq-system"))

    def test_malformed_rows_are_refused(self):
        user, assistant = {"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}
        bad = [
            dict(ROWS[1], CONVERSATION=[user, assistant]),                    # ends with the assistant
            dict(ROWS[1], CONVERSATION=[assistant, user]),                    # starts with the assistant
            dict(ROWS[1], CONVERSATION=[user, user]),                         # does not alternate
            dict(ROWS[1], CONVERSATION=[user, {"role": "system", "content": "s"}, user]),
            dict(ROWS[1], CONVERSATION=[{"role": "tool", "content": "a"}]),
            dict(ROWS[1], CONVERSATION=[{"role": "user", "content": "a", "name": "x"}]),
            dict(ROWS[1], CONVERSATION=[{"role": "user", "content": None}]),
            dict(ROWS[1], CONVERSATION=[]),
            dict(ROWS[1], AXIS="RECOLLECTION"),
            dict(ROWS[1], PASS_CRITERIA="yes"),
            dict(ROWS[1], TARGET_QUESTION="  "),
            dict(ROWS[1], QUESTION_ID=7),
            {k: v for k, v in ROWS[1].items() if k != "TARGET_QUESTION"},
        ]
        for row in bad:
            with self.subTest(row=row), self.assertRaises(ValueError):
                load_github([ROWS[0], row, ROWS[2], ROWS[3]])
        with self.assertRaises(ValueError):
            load_github([ROWS[0], ROWS[0], ROWS[2], ROWS[3]])                 # duplicate question id

    def test_load_refuses_axis_counts_other_than_the_pinned_release(self):
        with self.assertRaises(ValueError):
            load_github(ROWS, counts={IR: 2, IM: 1, RVE: 1, SC: 0})

    def test_pins_are_exact(self):
        for digest in [mc.HF_SHA256, mc.GITHUB_SHA256, mc.JUDGE_PROMPT_SHA256] + [f[2] for f in mc.REFERENCE_FILES]:
            self.assertRegex(digest, r"\A[0-9a-f]{64}\Z")
        self.assertIn("/resolve/" + mc.HF_REVISION + "/", mc.HF_URL)
        self.assertRegex(mc.HF_REVISION, r"\A[0-9a-f]{40}\Z")
        self.assertRegex(mc.GITHUB_COMMIT, r"\A[0-9a-f]{40}\Z")
        self.assertTrue(mc.GITHUB_RAW.endswith("/" + mc.GITHUB_COMMIT + "/"))


class JudgeTests(unittest.TestCase):
    def setUp(self):
        self.items = load_github(ROWS)

    def test_judge_prompt_is_the_official_text(self):
        # SHA-256 of JUDGE_PROMPT in src/evaluator.py at the pinned commit.
        self.assertEqual(core.sha256_text(mc.JUDGE_PROMPT), "7519dc6e1af487f8a9850239b334f16238032994b763a8b5108ad67bd0d5be7f")
        self.assertEqual(mc.JUDGE_PROMPT_SHA256, core.sha256_text(mc.JUDGE_PROMPT))
        self.assertEqual(len(mc.JUDGE_PROMPT), 383)
        self.assertEqual(mc.JUDGE_PROMPT.count("{}"), 2)

    def test_one_request_with_the_filled_official_prompt(self):
        answer = "zq-answer: a {0} lighthouse {} stands tall.\n"
        requests = mc.REFERENCE_BENCHMARK.judge_requests(self.items[0], answer)
        self.assertEqual(len(requests), 1)
        request = requests[0]
        self.assertEqual(request.name, "verdict")
        self.assertEqual(request.max_tokens, 4096)
        self.assertEqual(request.system, mc.JUDGE_SYSTEM)
        expected = (
            "You are tasked with evaluating a model response to see if it meets a specific criteria.\n"
            "The criteria will always be YES/NO evaluation.\n\n"
            "The model response is as follows:\n<MODEL_RESPONSE>\n"
            "zq-answer: a {0} lighthouse {} stands tall.\n"
            "\n</MODEL_RESPONSE>\n\n"
            "The criteria that the model response must meet is as follows. Be VERY STRICT!:\n<CRITERIA>\n"
            "zq-rubric-ir: Is every line of the response lowercase?"
            "\n</CRITERIA>\n\n"
            'Print your reasoning followed by your verdict, either "YES" or "NO".'
        )
        self.assertEqual(request.user, expected)
        self.assertNotIn("zq-user", request.user)  # the judge sees only the answer and the rubric

    def test_no_judge_call_without_an_answer(self):
        self.assertEqual(mc.REFERENCE_BENCHMARK.judge_requests(self.items[0], None), [])
        self.assertEqual(mc.REFERENCE_BENCHMARK.judge_requests(self.items[0], " \n\t"), [])

    def test_system_message_carries_the_judge_response_schema(self):
        head, schema = mc.JUDGE_SYSTEM.split("\n", 1)
        self.assertIn("JSON", head)
        self.assertEqual(json.loads(schema), mc.JUDGE_RESPONSE_SCHEMA)
        self.assertEqual(mc.JUDGE_RESPONSE_SCHEMA["properties"]["verdict"]["enum"], ["YES", "NO"])

    def test_verdict_parsing_follows_the_official_pydantic_model(self):
        accepted = {
            '{"reasoning": "r", "verdict": "YES"}': "YES",
            ' \n{"reasoning": "r", "verdict": "NO"}\n': "NO",
            '{"reasoning": "", "verdict": "NO", "confidence": 0.9}': "NO",       # extra keys ignored
            '{"reasoning": "r", "verdict": "NO", "verdict": "YES"}': "YES",      # last key wins
            '{"reasoning": "r", "verdict": "Y\\u0045S"}': "YES",
            '```json\n{"reasoning": "r", "verdict": "YES"}\n```': "YES",         # the one tolerance
            '```\n{"reasoning": "r", "verdict": "NO"}```': "NO",
            # The stand-in judge's prose around the object: its last verdict object is read
            # (a documented deviation; see parse_verdict and docs/frontier/multichallenge.md).
            'Reasoning first. {"reasoning": "r", "verdict": "YES"}': "YES",
            '{"reasoning": "r", "verdict": "YES"}\nSo the verdict is YES.': "YES",
            '```json\n{"reasoning": "r", "verdict": "YES"}\n```\nDone.': "YES",
            'Draft {"reasoning": "a", "verdict": "NO"} then {"reasoning": "b", "verdict": "YES"}': "YES",
            # The official prompt's own form, a verdict line after the reasoning, without JSON.
            'Reasoning.\n\n**Verdict: YES**': "YES",
            'Reasoning.\n\n**Verdict:** NO': "NO",
            'At first the verdict looked like YES, but no.\nVerdict: NO': "NO",
            # The official prompt's words taken literally: reasoning, then a line of only the verdict.
            'The response meets it.\n\nYES': "YES",
            'A close call.\n\n**NO**': "NO",
            'Reasoning.\nYES.': "YES",
            # Invalid JSON whose verdict is still explicit: read by the verdict-line fallback.
            '{"reasoning": "r", "verdict": "YES"': "YES",
            '{"reasoning": "line one\nline two", "verdict": "YES"}': "YES",
        }
        for reply, verdict in accepted.items():
            with self.subTest(reply=reply):
                self.assertEqual(mc.parse_verdict(reply), verdict)
        rejected = [
            None, "", "YES", '"YES"', "null",
            '{"reasoning": "r", "verdict": "yes"}',
            '{"reasoning": "r", "verdict": " YES"}',
            '{"reasoning": "r", "verdict": true}',
            '{"reasoning": "r", "verdict": ["YES"]}',
            '{"reasoning": "r", "verdict": "MAYBE"}',
            '{"verdict": "YES"}',
            '{"reasoning": 5, "verdict": "YES"}',
            '{"reasoning": null, "verdict": "YES"}',
            '[{"reasoning": "r", "verdict": "YES"}]',
            "[" * 100000,
            'Prose only, the answer is YES.',
            'Reasoning.\n\n**YES/NO verdict:**',
            'Reasoning.\nThe answer is YES',
            '**YES**',
            'Prose {"verdict": "MAYBE", "reasoning": "r"}',
        ]
        for reply in rejected:
            with self.subTest(reply=reply if reply is None else reply[:60]):
                self.assertIsNone(mc.parse_verdict(reply))


class GradeTests(unittest.TestCase):
    def setUp(self):
        self.items = load_github(ROWS)
        self.bench = mc.REFERENCE_BENCHMARK

    def grade(self, index, answer, judged):
        return self.bench.grade(self.items[index], answer, judged)

    def test_yes_passes_and_no_fails(self):
        self.assertEqual(self.grade(0, "zq-answer", {"verdict": judge_reply("YES")}),
                         {"axis": IR, "verdict": "YES", "passed": True, "failed_answer": False, "failed_judge_calls": 0})
        self.assertEqual(self.grade(0, "zq-answer", {"verdict": judge_reply("NO")}),
                         {"axis": IR, "verdict": "NO", "passed": False, "failed_answer": False, "failed_judge_calls": 0})

    def test_pass_means_the_verdict_equals_the_pass_criteria(self):
        self.assertTrue(self.grade(3, "zq-answer", {"verdict": judge_reply("NO")})["passed"])
        self.assertFalse(self.grade(3, "zq-answer", {"verdict": judge_reply("YES")})["passed"])

    def test_missing_or_blank_answer_fails_without_a_judge_call(self):
        for answer in (None, "", "  \n"):
            with self.subTest(answer=answer):
                self.assertEqual(self.grade(1, answer, {}),
                                 {"axis": IM, "verdict": None, "passed": False, "failed_answer": True, "failed_judge_calls": 0})

    def test_failed_judge_call_is_marked_and_never_a_pass(self):
        for judged in ({"verdict": None}, {}, {"verdict": "The response passes. YES"}, {"other": judge_reply("YES")}):
            with self.subTest(judged=judged):
                self.assertEqual(self.grade(2, "zq-answer", judged),
                                 {"axis": RVE, "verdict": None, "passed": False, "failed_answer": False, "failed_judge_calls": 1})

    def test_grades_and_public_fields_hold_no_text(self):
        replies = [{"verdict": judge_reply("YES")}, {"verdict": judge_reply("NO")}, {"verdict": None},
                   {"verdict": "zq-unparseable judge text"}, {}]
        for index, item in enumerate(self.items):
            self.assertNotIn("zq-", json.dumps(dict(item.public)))
            for answer in ("zq-answer-text", None):
                for judged in replies:
                    grade = self.grade(index, answer, judged)
                    self.assertNotIn("zq-", json.dumps(grade))
                    self.assertEqual(set(grade), {"axis", "verdict", "passed", "failed_answer", "failed_judge_calls"})


def grades_for(axis, passes=0, fails=0, failed_answers=0, failed_judge_calls=0):
    ok = {"axis": axis, "failed_answer": False, "failed_judge_calls": 0}
    return ([dict(ok, verdict="YES", passed=True)] * passes + [dict(ok, verdict="NO", passed=False)] * fails
            + [dict(ok, verdict=None, passed=False, failed_answer=True)] * failed_answers
            + [dict(ok, verdict=None, passed=False, failed_judge_calls=1)] * failed_judge_calls)


class ScoreTests(unittest.TestCase):
    def test_hand_computed_example(self):
        grades = (grades_for(IR, passes=2, fails=1) + grades_for(IM, passes=1, fails=1, failed_answers=1, failed_judge_calls=1)
                  + grades_for(RVE, passes=2) + grades_for(SC, fails=1))
        result = mc.BENCHMARK.score(grades)
        # Axis rates 2/3, 1/4, 2/2 and 0/1; their mean is 23/48 = 0.479166...
        self.assertAlmostEqual(result["score"], 47.916667, places=5)
        # 1.96 * sqrt(2/27 + 3/64) / 4: the two non-degenerate axes' binomial variances.
        self.assertAlmostEqual(result["ci95"], 17.0411, places=3)
        self.assertEqual((result["n"], result["passed"], result["failed_answers"], result["failed_judge_calls"]), (10, 5, 1, 1))
        self.assertEqual(list(result["axes"]), list(mc.AXES))
        self.assertEqual(result["axes"][IM], {"n": 4, "passed": 1, "failed_answers": 1, "failed_judge_calls": 1, "score": 25.0})
        self.assertAlmostEqual(result["axes"][IR]["score"], 66.666667, places=5)
        self.assertEqual(result["axes"][RVE]["score"], 100.0)
        self.assertEqual(result["axes"][SC]["score"], 0.0)
        json.dumps(result)

    def test_graded_items_score_like_the_hand_example(self):
        items = load_github(ROWS)
        bench = mc.REFERENCE_BENCHMARK
        grades = [bench.grade(items[0], "a", {"verdict": judge_reply("YES")}),
                  bench.grade(items[1], None, {}),
                  bench.grade(items[2], "a", {"verdict": None}),
                  bench.grade(items[3], "a", {"verdict": judge_reply("NO")})]
        result = bench.score(grades)
        self.assertEqual(result["score"], 50.0)  # (1 + 0 + 0 + 1) / 4
        self.assertEqual((result["failed_answers"], result["failed_judge_calls"]), (1, 1))

    def test_paper_averages_reproduce_from_axis_counts(self):
        # arXiv:2501.17399v2 Table 5 on the 69/113/41/50 split.
        llama = grades_for(IR, 23, 46) + grades_for(IM, 17, 96) + grades_for(RVE, 10, 31) + grades_for(SC, 10, 40)
        mixtral = grades_for(IR, 9, 60) + grades_for(IM, 8, 105) + grades_for(RVE, 5, 36) + grades_for(SC, 6, 44)
        self.assertEqual(round(mc.REFERENCE_BENCHMARK.score(llama)["score"], 2), 23.19)
        self.assertEqual(round(mc.REFERENCE_BENCHMARK.score(mixtral)["score"], 2), 11.08)
        self.assertEqual(mc.REFERENCE_BENCHMARK.score(llama)["n"], 273)

    def test_failures_never_count_and_never_form_an_axis(self):
        corrupt = [{"axis": IR, "verdict": "YES", "passed": True, "failed_answer": False, "failed_judge_calls": 1},
                   {"axis": IR, "verdict": "YES", "passed": True, "failed_answer": True, "failed_judge_calls": 0}]
        result = mc.BENCHMARK.score(grades_for(IR, passes=2) + corrupt)
        self.assertEqual(result["axes"][IR]["passed"], 2)
        self.assertEqual(result["score"], 50.0)  # IR alone: 2 of 4
        self.assertEqual(set(result["axes"]), set(mc.AXES))

    def test_empty_and_partial_runs(self):
        empty = mc.BENCHMARK.score([])
        self.assertEqual((empty["score"], empty["ci95"], empty["n"]), (None, None, 0))
        partial = mc.BENCHMARK.score(grades_for(IR, passes=1) + grades_for(SC, fails=1))
        self.assertEqual(partial["score"], 50.0)  # the mean over the axes present, as the official code
        self.assertIsNone(partial["axes"][IM]["score"])

    def test_unknown_axis_is_refused(self):
        with self.assertRaises(ValueError):
            mc.BENCHMARK.score([{"axis": "NA", "passed": False, "failed_answer": False, "failed_judge_calls": 1}])


class ReferenceTests(unittest.TestCase):
    def test_reference_scores_match_the_paper_arithmetic(self):
        self.assertEqual(set(mc.REFERENCE_SCORES), {f[0] for f in mc.REFERENCE_FILES})
        self.assertEqual(len(mc.REFERENCE_SCORES), 12)
        for model, row in mc.REFERENCE_SCORES.items():
            with self.subTest(model=model):
                rates = [row["axes"][axis] for axis in mc.AXES]
                self.assertLessEqual(abs(sum(rates) / 4 - row["score"]), 0.0051)
                counts = [row["axes"][axis] * row["axis_n"][axis] / 100 for axis in mc.AXES]
                self.assertTrue(all(abs(c - round(c)) < 0.01 for c in counts))
                self.assertEqual(sum(round(c) for c in counts), row["passed"])
                self.assertEqual(sum(row["axis_n"].values()), 273)
                table5 = row["source"].endswith("Table 5")
                self.assertEqual(row["axis_n"], mc.GITHUB_AXIS_COUNTS if table5 else mc.PAPER_TABLE3_AXIS_N)
                self.assertEqual(row["compare_on"], "score" if table5 else "passed")
                self.assertIn("arXiv:2501.17399v2", row["source"])

    def test_recommended_models_spread_the_paper_scores(self):
        passes = [mc.REFERENCE_SCORES[m]["passed"] for m in mc.REFERENCE_RECOMMENDED]
        self.assertEqual(passes, sorted(passes))
        self.assertGreaterEqual(passes[-1] - passes[0], 60)
        self.assertFalse(any(m.startswith(("claude", "gpt-4o")) for m in mc.REFERENCE_RECOMMENDED))

    def stored(self, rows, responses):
        return b"".join(json.dumps({"QUESTION_ID": r["QUESTION_ID"], "RESPONSE": text}).encode() + b"\n"
                        for r, text in zip(rows, responses))

    def test_reference_responses_map_ids_to_texts(self):
        files = [("model-a", self.stored(reversed(ROWS), [["zq-a3"], ["zq-a2"], ["zq-a1"], ["zq-a0"]])),
                 ("model-b", self.stored(ROWS, [["zq-b0"], ["zq-b1"], ["zq-b2"], ["zq-b3"]]))]
        with offline_cache(github=github_bytes(ROWS), responses=files) as root:
            responses = mc.reference_responses(root)
            ids = [item.id for item in mc.REFERENCE_BENCHMARK.load(root)]
        self.assertEqual(list(responses), ["model-a", "model-b"])
        self.assertEqual(list(responses["model-a"]), ids)  # item order, whatever the file order
        self.assertEqual(list(responses["model-a"].values()), ["zq-a0", "zq-a1", "zq-a2", "zq-a3"])
        self.assertEqual(responses["model-b"][ids[2]], "zq-b2")

    def test_reference_responses_refuse_incomplete_files(self):
        bad = [
            self.stored(ROWS[:3], [["x"]] * 3),                                   # a conversation is missing
            self.stored(ROWS + ROWS[:1], [["x"]] * 5),                            # a conversation twice
            self.stored(ROWS, [["x"], ["x", "y"], ["x"], ["x"]]),                 # two attempts
            self.stored(ROWS, [["x"], "x", ["x"], ["x"]]),                        # not a list
            self.stored(ROWS, [["x"], [None], ["x"], ["x"]]),
            self.stored(ROWS, [["x"]] * 4) + json.dumps({"QUESTION_ID": "unknown", "RESPONSE": ["x"]}).encode(),
        ]
        for data in bad:
            with self.subTest(data=data[:40]), offline_cache(github=github_bytes(ROWS), responses=[("m", data)]) as root:
                with self.assertRaises(ValueError):
                    mc.reference_responses(root)


if __name__ == "__main__":
    unittest.main()
