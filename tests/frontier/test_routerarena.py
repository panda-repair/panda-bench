"""Offline tests for the RouterArena module: synthetic rows, no network, no model calls.

Two optional groups run only when their inputs exist: a fidelity check against a clone of
RouterArena at the pinned commit (``ROUTERARENA_CLONE``, default ``../cache/routerarena/repo``
next to this repository), and a container check (``PANDA_BENCH_DOCKER_TESTS=1``)."""
from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import pickle
import tempfile
import unittest
import zlib
from pathlib import Path
from unittest import mock

from panda_bench.frontier import core
from panda_bench.frontier.benchmarks import routerarena as ra

REPO_ROOT = Path(__file__).resolve().parents[2]
CLONE = Path(os.environ.get("ROUTERARENA_CLONE", str(REPO_ROOT.parent / "cache" / "routerarena" / "repo")))

QUESTION = "UNIQUE-QUESTION-TEXT-7f3a"
ANSWER_TEXT = "Unique_Answer_Text_9c1d"


def row(global_index, name, question=QUESTION, answer="B", options=None, context="", category="51 Mathematics"):
    return {"Global Index": global_index, "Dataset name": name, "Question": question, "Context": context,
            "Options": options, "Answer": answer, "Category": category, "Domain": "5 Science",
            "Difficulty": "easy", "Metadata": "{}", "Keywords": ""}


def item(global_index, name, answer="B", options=None, question=QUESTION, context="", lcb=None):
    return ra.make_item(row(global_index, name, question, answer, options, context), lcb)


def grade(it, answer):
    return ra.BENCHMARK.grade(it, answer, {})


class PromptTests(unittest.TestCase):
    def test_multiple_choice_prompt(self):
        prompt = ra.format_prompt(row("MMLUPro_1", "MMLUPro_biology", "Which?", options=["x", "y"]))
        self.assertEqual(
            prompt,
            "Please read the following multiple-choice questions and provide the most likely correct answer "
            "based on the options given.\n\nContext: None\n\nQuestion: Which?\n\nOptions: \nA. x\nB. y\n\n\n"
            "Provide the correct letter choice in \\boxed{X}, where X is the correct letter choice. Keep the "
            "explanation or feedback within 3 sentences.")

    def test_math_prompt_keeps_routerarenas_doubled_braces(self):
        prompt = ra.format_prompt(row("AIME_3", "AIME", "Find n.", context="ctx"))
        self.assertIn("Context: ctx\n\nQuestion: Find n.\n\n", prompt)
        self.assertIn("in \\boxed{{}} format", prompt)
        self.assertIn("For example: \\boxed{{42}}.", prompt)

    def test_braces_in_question_are_doubled_like_routerarena(self):
        prompt = ra.format_prompt(row("GSM8K_1", "GSM8K", "Let S = {1, 2} and T = {{3}}."))
        self.assertIn("Question: Let S = {{1, 2}} and T = {{3}}.\n", prompt)

    def test_special_templates(self):
        rc = ra.format_prompt(row("SuperGLUE-RC_1", "SuperGLUE-RC", "Q", options=["x"]))
        self.assertIn("Question: Q\n\nProvided Answer: \n\n Is the provided answer correct?", rc)
        wic = ra.format_prompt(row("SuperGLUE-Wic_1", "SuperGLUE-Wic", "S1 / S2", options=["Yes", "No"]))
        self.assertTrue(wic.startswith('Consider the word "" in the two sentences below.'))
        cloze = ra.format_prompt(row("SuperGLUE-ClozeTest_1", "SuperGLUE-ClozeTest", "P", options=["a", "b"]))
        self.assertIn("\n\nP\n\nOptions : A. a\nB. b\n\n\nProvide", cloze)

    def test_template_choice_by_dataset(self):
        self.assertEqual(ra.prompt_dataset(row("ChessInstruct_1", "ChessInstruct_mcq")), "ChessInstruct")
        self.assertEqual(ra.prompt_dataset(row("ChessInstruct_1", "ChessInstruct", options=["a"])), "ChessInstruct_mcq")
        self.assertEqual(ra.prompt_dataset(row("Ethics_virtue_4", "Ethics_virtue", options=["a"])), "Ethics_virtue")
        self.assertEqual(ra.prompt_dataset(row("Ethics_virtue_4", "", options=["a"])), "Ethics_virtue")
        self.assertEqual(ra.prompt_dataset(row("GeoGraphyData_100k_2", "GeoGraphyData_100k")), "GeoGraphyData")
        self.assertEqual(ra.prompt_dataset(row("OpenTDB_7", "OpenTDB_Entertainment: Books", options=["a"])), "OpenTDB")

    def test_livecodebench_variants(self):
        coding = row("LiveCodeBench_3", "LiveCodeBench", "Do it.")
        self.assertEqual(ra.format_prompt(coding, True), ra.LCB_STDIN_TEMPLATE.replace("{Question}", "Do it."))
        self.assertEqual(ra.format_prompt(coding, False), ra.LCB_FUNCTIONAL_TEMPLATE.replace("{Question}", "Do it."))
        with self.assertRaises(ValueError):
            ra.format_prompt(coding)

    def test_long_prompts_are_cut_like_routerarena(self):
        prompt = ra.format_prompt(row("NarrativeQA_1", "NarrativeQA", "q", context="x" * 20000))
        self.assertEqual(len(prompt), 10003)
        self.assertEqual(prompt[5000:5003], "...")
        self.assertTrue(prompt.startswith("Please read the following context"))
        self.assertTrue(prompt.endswith("within 3 sentences."))


class MetricMappingTests(unittest.TestCase):
    def test_grading_dataset_comes_from_the_global_index(self):
        cases = {
            "ArcMMLU_1": ("MMLUPro", "mcq_accuracy"),
            "ChessInstruct_3": ("ChessInstruct", "chess_accuracy"),
            "Ethics_justice_2": ("Ethics_commonsense", "mcq_accuracy"),
            "GeoGraphyData_100k_5": ("GeoGraphyData_100k", "exact_match"),
            "QANTA_9": ("QANTA", "exact_match"),
            "SuperGLUE-ClozeTest_9": ("SuperGLUE-ClozeTest", "superglue_clozetest"),
            "SuperGLUE-CausalReasoning_1": ("SuperGLUE-CausalReasoning", "mcq_accuracy"),
            "SuperGLUE-Wsc_1": ("SuperGLUE-Wsc", "superglue_exact_match"),
            "WMT19-de-en_1": ("WMT19-de-en", "meteor_score"),
            "NarrativeQA_1": ("NarrativeQA", "meteor_score"),
            "LiveCodeBench_5": ("LiveCodeBench", "code_accuracy"),
            "AsDiv_2": ("AsDiv", "math_metric"),
            "MATH-500_1": ("MATH-500", "math_metric"),
            "Unknown_1": ("Unknown", "mcq_accuracy"),
        }
        for global_index, expected in cases.items():
            self.assertEqual(ra.eval_metric(global_index), expected, global_index)


class ItemTests(unittest.TestCase):
    def test_item_fields(self):
        it = item("MMLUPro_12", "MMLUPro_law", answer="3", options=["a", "b", "c", "d"])
        self.assertEqual(it.id, "ra-" + hashlib.sha256(b"MMLUPro_12").hexdigest()[:16])
        self.assertEqual(len(it.messages), 1)
        self.assertEqual(it.messages[0][0], "user")
        self.assertEqual(it.key["answer"], "3")
        self.assertEqual(it.public["messages_sha256"], core.messages_digest(it.messages))
        self.assertEqual(it.public["metric"], "mcq_accuracy")

    def test_public_never_holds_question_or_answer(self):
        for it in (item("QANTA_1", "QANTA_History", answer=ANSWER_TEXT),
                   item("MMLUPro_2", "MMLUPro_law", answer="0", options=[ANSWER_TEXT, "other"]),
                   item("NarrativeQA_3", "NarrativeQA", answer=ANSWER_TEXT, context=QUESTION)):
            text = json.dumps(it.public)
            self.assertNotIn(QUESTION, text)
            self.assertNotIn(ANSWER_TEXT, text)

    def test_coding_item_needs_its_tests(self):
        with self.assertRaises(ValueError):
            item("LiveCodeBench_4", "LiveCodeBench", answer="4")
        lcb = {"is_stdin": True, "tests_path": "/x/LiveCodeBench_4.json", "tests_sha256": "ab", "n_tests": 3}
        it = item("LiveCodeBench_4", "LiveCodeBench", answer="4", lcb=lcb)
        self.assertTrue(it.messages[0][1].startswith(ra.LCB_STDIN_TEMPLATE.split("{")[0]))
        self.assertEqual((it.key["lcb_is_stdin"], it.key["lcb_n_tests"]), (True, 3))


class ExtractionTests(unittest.TestCase):
    def test_routerarena_extraction_quirks_are_kept(self):
        self.assertEqual(ra.extract_boxed_answer("so \\boxed{\\text{C}}.", "MCQ"), "C")
        self.assertEqual(ra.extract_boxed_answer("The answer is C", "MCQ"), "C")
        # "answer: <letter>" is matched case-insensitively, so this reads the "t" of "the".
        self.assertEqual(ra.extract_boxed_answer("Answer: the B", "MCQ"), "T")
        self.assertEqual(ra.extract_boxed_answer("first \\boxed{A} then \\boxed{B}", "MCQ"), "A")
        self.assertEqual(ra.extract_boxed_answer("leds it is", "SuperGLUE-ClozeTest"), "L")
        self.assertEqual(ra.extract_boxed_answer("   ", "MCQ"), "")

    def test_math_extraction(self):
        self.assertEqual(ra.extract_math_answer("so \\boxed{\\frac{3}{4}} done"), "\\frac{3}{4}")
        self.assertEqual(ra.extract_math_answer("we get 12 then 15"), "15")
        self.assertEqual(ra.extract_math_answer("The final answer: 7."), "7")

    def test_chess_extraction(self):
        self.assertEqual(ra.extract_chess_move("\\boxed{E2-E4}"), "e2e4")
        self.assertEqual(ra.extract_chess_move("I would play g1f3 here"), "g1f3")


class GradeTests(unittest.TestCase):
    def test_multiple_choice(self):
        it = item("MMLUPro_1", "MMLUPro_law", answer="2", options=["a", "b", "c"])
        self.assertEqual(grade(it, "\\boxed{C}")["score"], 1.0)
        self.assertEqual(grade(it, "\\boxed{C}")["letter"], "C")
        self.assertEqual(grade(it, "\\boxed{B}")["score"], 0.0)
        wrong_form = grade(it, "\\boxed{C. text}")
        self.assertEqual((wrong_form["score"], wrong_form["letter"]), (0.0, None))
        self.assertEqual(grade(item("ArcMMLU_1", "ArcMMLU", answer="B", options=["a", "b"]), "\\boxed{B}")["score"], 1.0)
        self.assertEqual(grade(item("Ethics_x_1", "Ethics_justice", answer="1.0", options=["a", "b"]), "B")["score"], 1.0)
        broken = grade(item("MMLU_1", "MMLU_management", answer="zz", options=["a"]), "\\boxed{A}")
        self.assertEqual((broken["score"], broken["reference_error"]), (0.0, True))

    def test_math(self):
        cases = [
            ("\\boxed{1,234}", "1234", 1.0, "decided"),
            ("\\boxed{50\\%}", "0.5", 1.0, "decided"),
            ("\\boxed{0.5}", "50", 1.0, "decided"),
            ("so x = 5, \\boxed{x = 5}", "5", 1.0, "decided"),
            ("\\boxed{(1, 2)}", "(1,2)", 1.0, "decided"),
            ("\\boxed{(C)}", "C", 1.0, "decided"),
            ("\\boxed{\\frac{3}{4}}", "\\frac{3}{4}", 1.0, "decided"),
            ("\\boxed{12}", "13", 0.0, "decided"),
            ("\\boxed{\\frac{1}{2}}", "0.5", 0.0, "unresolved"),
        ]
        for answer, reference, score, check in cases:
            result = grade(item("MATH_1", "MATH", answer=reference), answer)
            self.assertEqual((result["score"], result["math_check"]), (score, check), answer)

    def test_exact_match(self):
        it = item("QANTA_1", "QANTA_History", answer=ANSWER_TEXT)
        self.assertEqual(grade(it, "\\boxed{unique answer text 9c1d}")["score"], 1.0)
        self.assertEqual(grade(it, "\\boxed{Something else}")["score"], 0.0)

    def test_chess(self):
        move = item("ChessInstruct_5", "ChessInstruct", answer="e2e4")
        self.assertEqual(grade(move, "\\boxed{E2-E4}")["score"], 1.0)
        self.assertEqual(grade(move, "best is e2e4")["score"], 1.0)
        choice = item("ChessInstruct_6", "ChessInstruct_mcq", answer="1", options=["a", "b"])
        self.assertEqual(choice.key["metric"], "chess_accuracy")
        self.assertEqual(grade(choice, "\\boxed{B}")["score"], 1.0)
        self.assertEqual(grade(choice, "\\boxed{C}")["score"], 0.0)

    def test_meteor(self):
        it = item("WMT19-de-en_1", "WMT19-de-en", answer="the cat sat on a mat")
        result = grade(it, "Sure: \\boxed{the cat sat on the mat}")
        # matches 5 of 6 tokens each side, one chunk against five matched words: 5/6 * (1 - 0.02)
        self.assertAlmostEqual(result["score"], 5 / 6 * 0.98)
        self.assertIsNone(result["correct"])
        # RouterArena's penalty never vanishes for a multi-word exact copy.
        self.assertAlmostEqual(ra.meteor("a b c", "a b c"), 1 - 0.5 / 9)
        self.assertAlmostEqual(ra.meteor("猫在垫子上", "猫在垫子上"), 0.98)
        self.assertEqual(grade(it, "")["score"], 0.0)

    def test_superglue(self):
        numeric = item("SuperGLUE-QA_1", "SuperGLUE-QA", answer="1.0")
        self.assertEqual(grade(numeric, "\\boxed{B}")["score"], 1.0)
        self.assertEqual(grade(numeric, "\\boxed{1}")["score"], 1.0)
        self.assertEqual(grade(numeric, "\\boxed{0}")["score"], 0.0)
        yes_no = item("SuperGLUE-Wic_1", "SuperGLUE-Wic", answer="Yes", options=["Yes", "No"])
        self.assertEqual(grade(yes_no, "\\boxed{Yes}")["score"], 1.0)
        self.assertEqual(grade(yes_no, "\\boxed{A}")["score"], 1.0)
        self.assertEqual(grade(yes_no, "\\boxed{yes}")["score"], 0.0)  # RouterArena compares case-sensitively

    def test_clozetest(self):
        text = item("SuperGLUE-ClozeTest_1", "SuperGLUE-ClozeTest", answer="LEDs", options=["LEDs", "x"])
        self.assertEqual(grade(text, "\\boxed{LEDs}")["score"], 1.0)
        self.assertEqual(grade(text, "LEDs, clearly")["score"], 0.0)
        index = item("SuperGLUE-ClozeTest_2", "SuperGLUE-ClozeTest", answer="3", options=["a", "b", "c", "d"])
        self.assertEqual(grade(index, "\\boxed{D}")["score"], 1.0)
        out_of_range = item("SuperGLUE-ClozeTest_3", "SuperGLUE-ClozeTest", answer="30", options=["a"])
        self.assertEqual(grade(out_of_range, "\\boxed{30}")["score"], 1.0)

    def test_missing_answer_scores_zero(self):
        result = ra.BENCHMARK.grade(item("MMLUPro_1", "MMLUPro_law", answer="0", options=["a"]), None, {})
        self.assertEqual(result, {"dataset": "MMLUPro", "metric": "mcq_accuracy", "score": 0.0,
                                  "correct": False, "failed_answer": True})

    def test_no_judge_calls_and_failed_judge_replies_are_ignored(self):
        it = item("MMLUPro_1", "MMLUPro_law", answer="0", options=["a"])
        self.assertEqual(ra.BENCHMARK.judge_requests(it, "\\boxed{A}"), [])
        self.assertEqual(ra.BENCHMARK.grade(it, "\\boxed{A}", {"stray": None})["score"], 1.0)

    def test_scorer_exception_counts_wrong(self):
        it = item("QANTA_1", "QANTA_History", answer="x")
        with mock.patch.dict(ra.TEXT_SCORERS, {"exact_match": mock.Mock(side_effect=RuntimeError("boom"))}):
            result = grade(it, "\\boxed{x}")
        self.assertEqual((result["score"], result["scorer_error"]), (0.0, True))

    def test_grades_never_hold_question_reference_or_answer_text(self):
        cases = [
            (item("QANTA_1", "QANTA_History", answer=ANSWER_TEXT), "\\boxed{%s}" % ANSWER_TEXT),
            (item("MATH_1", "MATH", answer=ANSWER_TEXT), "\\boxed{%s}" % ANSWER_TEXT),
            (item("NarrativeQA_1", "NarrativeQA", answer=ANSWER_TEXT), "\\boxed{%s}" % ANSWER_TEXT),
            (item("ChessInstruct_1", "ChessInstruct", answer=ANSWER_TEXT), "\\boxed{%s}" % ANSWER_TEXT),
            (item("SuperGLUE-ClozeTest_1", "SuperGLUE-ClozeTest", answer=ANSWER_TEXT, options=[ANSWER_TEXT]),
             "\\boxed{%s}" % ANSWER_TEXT),
            (item("MMLUPro_1", "MMLUPro_law", answer="0", options=[ANSWER_TEXT]), "\\boxed{%s}" % ANSWER_TEXT),
        ]
        for it, answer in cases:
            text = json.dumps(grade(it, answer))
            self.assertNotIn(QUESTION, text)
            self.assertNotIn(ANSWER_TEXT, text)
            self.assertNotIn(ANSWER_TEXT.lower().replace("_", " "), text)


class CodeGradeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tests_path = Path(self.tmp.name) / "LiveCodeBench_7.json"
        self.tests_path.write_text(json.dumps([{"input": "1\n", "output": "1\n", "testtype": "stdin"}]))
        digest = hashlib.sha256(self.tests_path.read_bytes()).hexdigest()
        lcb = {"is_stdin": True, "tests_path": str(self.tests_path), "tests_sha256": digest, "n_tests": 1}
        self.item = item("LiveCodeBench_7", "LiveCodeBench", answer="7", lcb=lcb)

    def tearDown(self):
        self.tmp.cleanup()

    def test_last_block_is_post_processed_and_sent_to_the_sandbox(self):
        answer = "draft\n```python\nprint(0)\n```\nfinal\n```python\nprint(input())\n```"
        with mock.patch.object(ra, "run_code_sandbox", return_value={"status": "ran", "score": 1.0,
                                                                      "timed_out": False}) as sandbox:
            result = grade(self.item, answer)
        sandbox.assert_called_once_with("print(input())\n", self.tests_path.read_bytes(), False, 1)
        self.assertEqual((result["score"], result["correct"], result["code_check"]), (1.0, True, "ran"))

    def test_function_problems_use_extracted_inputs(self):
        lcb = {"is_stdin": False, "tests_path": str(self.tests_path),
               "tests_sha256": hashlib.sha256(self.tests_path.read_bytes()).hexdigest(), "n_tests": 1}
        it = item("LiveCodeBench_8", "LiveCodeBench", answer="8", lcb=lcb)
        with mock.patch.object(ra, "run_code_sandbox", return_value={"status": "ran", "score": 0.0,
                                                                      "timed_out": True}) as sandbox:
            result = grade(it, "```\ndef f(x):\n    return x\n```")
        self.assertTrue(sandbox.call_args[0][2])
        self.assertEqual((result["score"], result["timed_out"]), (0.0, True))

    def test_no_code_block_scores_zero_without_running_anything(self):
        with mock.patch.object(ra, "run_code_sandbox") as sandbox:
            result = grade(self.item, "def f(): pass")
        sandbox.assert_not_called()
        self.assertEqual((result["score"], result["code_check"]), (0.0, "no_code"))

    def test_sandbox_failures_are_marked(self):
        for status in ("sandbox_error", "unavailable"):
            with mock.patch.object(ra, "run_code_sandbox", return_value={"status": status}):
                result = grade(self.item, "```python\nimport numpy\nprint(1)\n```")
            self.assertEqual((result["score"], result["code_check"], result["third_party_import"]),
                             (0.0, status, True))

    def test_changed_tests_file_is_refused(self):
        self.tests_path.write_text("[]")
        with self.assertRaises(ValueError):
            grade(self.item, "```python\nprint(1)\n```")

    def test_no_docker_setting_disables_the_sandbox(self):
        with mock.patch.dict(os.environ, {ra.NO_DOCKER_ENV: "1"}):
            self.assertFalse(ra.docker_available())
            self.assertEqual(ra.run_code_sandbox("print(1)", b"[]", False, 1), {"status": "unavailable"})

    def test_private_tests_unpickle_without_globals(self):
        tests = [{"input": "1", "output": "2", "testtype": "stdin"}]
        encoded = base64.b64encode(zlib.compress(pickle.dumps(json.dumps(tests)))).decode()
        self.assertEqual(ra.translate_private_test_cases(encoded), tests)
        hostile = base64.b64encode(zlib.compress(pickle.dumps(os.getcwd))).decode()
        with self.assertRaises(pickle.UnpicklingError):
            ra.translate_private_test_cases(hostile)


class ScoreTests(unittest.TestCase):
    def test_hand_computed_example(self):
        grades = [
            {"dataset": "MMLUPro", "metric": "mcq_accuracy", "score": 1.0, "failed_answer": False},
            {"dataset": "MMLUPro", "metric": "mcq_accuracy", "score": 0.0, "failed_answer": False},
            {"dataset": "NarrativeQA", "metric": "meteor_score", "score": 0.5, "failed_answer": False},
            {"dataset": "MMLUPro", "metric": "mcq_accuracy", "score": 0.0, "failed_answer": True},
            {"dataset": "LiveCodeBench", "metric": "code_accuracy", "score": 0.0, "code_check": "sandbox_error",
             "third_party_import": True},
            {"dataset": "LiveCodeBench", "metric": "code_accuracy", "score": 1.0, "code_check": "ran",
             "third_party_import": True},
            {"dataset": "LiveCodeBench", "metric": "code_accuracy", "score": 0.0, "code_check": "ran",
             "third_party_import": True},
            {"dataset": "MATH", "metric": "math_metric", "score": 0.0, "math_check": "unresolved"},
            {"dataset": "LiveCodeBench", "metric": "code_accuracy", "score": 0.0, "code_check": "unavailable"},
        ]
        result = ra.BENCHMARK.score(grades)
        # 1 + 0 + 0.5 + 0 + 0 + 1 + 0 + 0 + 0 = 2.5 points over 9 items
        self.assertAlmostEqual(result["score"], 100 * 2.5 / 9)
        self.assertAlmostEqual(result["accuracy"], 2.5 / 9)
        self.assertEqual((result["n"], result["points"]), (9, 2.5))
        self.assertEqual(result["n_failed_answers"], 1)
        self.assertEqual(result["n_failed_judge_calls"], 0)
        self.assertEqual(result["n_code_sandbox_errors"], 1)
        self.assertEqual(result["n_code_ungraded"], 1)
        self.assertEqual(result["n_code_failed_third_party_import"], 1)
        self.assertEqual(result["n_math_unresolved"], 1)
        self.assertEqual(result["by_metric"]["mcq_accuracy"], {"n": 3, "score": 100 / 3})
        self.assertEqual(result["by_metric"]["code_accuracy"], {"n": 4, "score": 25.0})
        self.assertEqual(result["by_dataset"]["NarrativeQA"], {"n": 1, "score": 50.0})
        json.dumps(result)

    def test_empty(self):
        self.assertEqual(ra.BENCHMARK.score([])["score"], 0.0)


class CostTests(unittest.TestCase):
    EMBED = {"model": "@cf/google/embeddinggemma-2", "input_tokens": 30, "output_tokens": 0, "purpose": "routing"}

    def rows(self):
        return [("deepseek-v4.1-flash", 1000, 2000, [self.EMBED]),
                ("glm-5.3", 500, 1000, None),
                ("glm-5.3-flash", 100, 0, [self.EMBED])]  # no output tokens: a failed generation

    def test_routerarena_prices(self):
        summary = ra.cost_summary(self.rows())
        answers = (1000 * 0.21 + 2000 * 0.84 + 500 * 1.4 + 1000 * 4.4) / 1e6
        router = 30 * 0.0017 / 1e6
        self.assertAlmostEqual(summary["answer_cost"], answers)
        self.assertAlmostEqual(summary["router_cost"], router)
        self.assertAlmostEqual(summary["cost_per_1k"], 1000 * (answers + router) / 3)
        self.assertEqual((summary["n_queries"], summary["n_costed"], summary["unpriced_models"]), (3, 2, {}))
        self.assertAlmostEqual(ra.cost_per_1k(self.rows()), summary["cost_per_1k"])

    def test_panda_list_prices(self):
        answers = (1000 * 0.30 + 2000 * 1.20 + 500 * 1.40 + 1000 * 4.40) / 1e6
        router = 30 * 0.0017 / 1e6
        self.assertAlmostEqual(ra.cost_per_1k(self.rows(), ra.PANDA_LIST_PRICES), 1000 * (answers + router) / 3)

    def test_unpriced_model_pays_the_table_maximum(self):
        summary = ra.cost_summary([("mystery-model", 1000, 1000, None)], ra.ROUTERARENA_PRICES)
        self.assertAlmostEqual(summary["total_cost"], (1000 * 15.0 + 1000 * 75.0) / 1e6)
        self.assertEqual(summary["unpriced_models"], {"mystery-model": 1})
        embed_only = ra.cost_summary([("glm-5.3-flash", 0, 10, [self.EMBED])], ra.ROUTERARENA_PRICES)
        self.assertEqual(embed_only["unpriced_models"], {"google/embeddinggemma-2": 1})

    def test_mapping_rows_reasoning_and_failures(self):
        rows = [
            {"model": "z-ai/glm-5.3-flash_batch", "input_tokens": 100, "output_tokens": 100, "total_tokens": 300,
             "answer": "x"},
            {"model": "glm-5.3", "input_tokens": 100, "output_tokens": 100, "answer": None},
            {"model": "glm-5.3", "input_tokens": 100, "output_tokens": 100, "answer": "  "},
            {"model": "glm-5.3", "input_tokens": 100, "output_tokens": 100, "success": False},
        ]
        summary = ra.cost_summary(rows)
        self.assertAlmostEqual(summary["total_cost"], (100 * 0.15 + 200 * 0.5) / 1e6)
        self.assertEqual((summary["n_queries"], summary["n_costed"]), (4, 1))

    def test_malformed_router_usage_is_refused(self):
        for bad in ([{"model": "x", "input_tokens": -1, "output_tokens": 0}],
                    [{"model": "x", "input_tokens": True, "output_tokens": 0}],
                    [{"model": "", "input_tokens": 1, "output_tokens": 0}], {"model": "x"}):
            with self.assertRaises(ValueError):
                ra.cost_summary([("glm-5.3", 1, 1, bad)])

    def test_empty(self):
        self.assertEqual(ra.cost_per_1k([]), 0.0)

    def test_arena_score_matches_published_rows(self):
        # README leaderboard at the pinned commit: Divyam, GPT-5, Azure-Model-Router.
        self.assertAlmostEqual(ra.arena_score(0.66, 0.8083), 77.19, delta=0.02)
        self.assertAlmostEqual(ra.arena_score(10.02, 0.7396), 64.32, delta=0.02)
        self.assertAlmostEqual(ra.arena_score(0.75, 0.7293), 70.37, delta=0.02)
        self.assertAlmostEqual(ra.arena_score(500, 0.5), ra.arena_score(200, 0.5))
        for cost, accuracy in ((0, 0.5), (1, 1.5), (1, -0.1)):
            with self.assertRaises(ValueError):
                ra.arena_score(cost, accuracy)


class SubmissionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name)
        self.items = [item("MMLUPro_1", "MMLUPro_law", answer="0", options=["a"]),
                      item("QANTA_2", "QANTA_History", answer="x"),
                      item("WMT19-de-en_3", "WMT19-de-en", answer="y")]
        embed = {"model": "@cf/google/embeddinggemma-2", "input_tokens": 41, "output_tokens": 0,
                 "purpose": "routing embedding"}
        self.rows = [
            {"item": self.items[0], "model": "@cf/deepseek-ai/deepseek-v4.1-flash-1017", "answer": "\\boxed{A}",
             "input_tokens": 120, "output_tokens": 900, "router_usage": [embed]},
            {"item": self.items[1], "model": "glm-5.3", "answer": None, "input_tokens": 0, "output_tokens": 0,
             "error": "timeout", "router_usage": [embed]},
            {"item": self.items[2], "model": "z-ai/glm-5.3-flash", "answer": "", "input_tokens": 50,
             "output_tokens": 16384, "total_tokens": 16434},
        ]

    def tearDown(self):
        self.tmp.cleanup()

    def test_files_follow_routerarenas_formats(self):
        summary = ra.write_submission(self.out, self.rows, expected_rows=3)
        config = json.loads((self.out / "router_inference/config/panda-auto.json").read_text())
        self.assertEqual(list(config), ["pipeline_params"])
        params = config["pipeline_params"]
        self.assertEqual(params["router_name"], "panda-auto")
        self.assertEqual(params["router_cls_name"], "external_prediction_file")
        self.assertEqual(params["models"], ["deepseek/deepseek-v4.1-flash", "z-ai/glm-5.3", "z-ai/glm-5.3-flash"])
        predictions = json.loads((self.out / "router_inference/predictions/panda-auto.json").read_text())
        self.assertEqual(len(predictions), 3)
        first = predictions[0]
        self.assertEqual(list(first), ["global index", "prompt", "prediction", "generated_result", "cost",
                                       "accuracy", "for_optimality", "router_usage"])
        self.assertEqual(first["global index"], "MMLUPro_1")
        self.assertEqual(first["prompt"], self.items[0].messages[0][1])
        self.assertEqual(first["prediction"], "deepseek/deepseek-v4.1-flash")
        self.assertEqual(first["generated_result"], {
            "generated_answer": "\\boxed{A}", "success": True,
            "token_usage": {"input_tokens": 120, "output_tokens": 900, "total_tokens": 1020},
            "provider": "panda", "error": None})
        self.assertEqual((first["cost"], first["accuracy"], first["for_optimality"]), (None, None, False))
        self.assertEqual(first["router_usage"], [{"model": "google/embeddinggemma-2", "input_tokens": 41,
                                                  "output_tokens": 0, "purpose": "routing embedding"}])
        failed, empty = predictions[1]["generated_result"], predictions[2]["generated_result"]
        self.assertEqual((failed["success"], failed["generated_answer"], failed["error"]), (False, "", "timeout"))
        self.assertEqual((empty["success"], empty["error"]), (False, "empty answer"))
        self.assertNotIn("router_usage", predictions[2])
        self.assertEqual((summary["n_rows"], summary["n_failed"]), (3, 2))
        self.assertTrue(summary["pr_notes"])

    def test_default_expects_all_809_rows(self):
        with self.assertRaises(ValueError):
            ra.write_submission(self.out, self.rows)

    def test_bad_rows_are_refused(self):
        bad_cases = [
            [self.rows[0], dict(self.rows[0])],
            [dict(self.rows[0], model="gpt-6-sol")],
            [dict(self.rows[0], router_usage=[{"model": "x", "input_tokens": "1", "output_tokens": 0}])],
            [dict(self.rows[0], answer=42)],
        ]
        for rows in bad_cases:
            with self.assertRaises(ValueError):
                ra.write_submission(self.out, rows, expected_rows=len(rows))
        with self.assertRaises(ValueError):
            ra.write_submission(self.out, self.rows, router_name="bad name", expected_rows=3)

    def test_rows_without_items(self):
        rows = [{"global_index": "MMLUPro_1", "prompt": "p", "model": "glm-5.3", "answer": "a",
                 "input_tokens": 1, "output_tokens": 1}]
        ra.write_submission(self.out, rows, router_name="x", expected_rows=1)
        written = json.loads((self.out / "router_inference/predictions/x.json").read_text())
        self.assertEqual((written[0]["global index"], written[0]["prompt"]), ("MMLUPro_1", "p"))


def _clone_commit(clone):
    """The clone's checked-out commit, read from .git without running git."""
    try:
        head = (clone / ".git" / "HEAD").read_text().strip()
        if not head.startswith("ref: "):
            return head
        ref = clone / ".git" / head[5:]
        if ref.is_file():
            return ref.read_text().strip()
        for line in (clone / ".git" / "packed-refs").read_text().splitlines():
            if line.endswith(" " + head[5:]):
                return line.split()[0]
    except OSError:
        pass
    return ""


CLONE_AT_PIN = (CLONE / ".git").is_dir() and _clone_commit(CLONE) == ra.HARNESS_COMMIT


@unittest.skipUnless(CLONE_AT_PIN, "no RouterArena clone at the pinned commit")
class PinnedCloneTests(unittest.TestCase):
    def test_templates_and_metrics_match_the_configs(self):
        for path in sorted((CLONE / "config" / "eval_config" / "zero-shot").glob("*.json")):
            params = json.loads(path.read_text(encoding="utf-8"))["eval_params"]
            self.assertEqual(ra.CONFIG_METRICS[path.stem], params["eval_metrics"][0], path.stem)
            if path.stem == "LiveCodeBench":
                self.assertEqual(ra.LCB_STDIN_TEMPLATE, params["is_stdin_prompt"])
                self.assertEqual(ra.LCB_FUNCTIONAL_TEMPLATE, params["not_is_stdin_prompt"])
            else:
                self.assertEqual(ra.PROMPT_TEMPLATES[path.stem], params["prompt"], path.stem)

    def test_prices_match_the_table(self):
        data = (CLONE / "model_cost" / "model_cost.json").read_bytes()
        self.assertEqual(hashlib.sha256(data).hexdigest(), ra.MODEL_COST_JSON_SHA256)
        table = json.loads(data)
        for name, (price_in, price_out) in ra.ROUTERARENA_PRICES.items():
            self.assertEqual((table[name]["input_token_price_per_million"],
                              table[name]["output_token_price_per_million"]), (price_in, price_out), name)
        self.assertEqual(ra.ROUTERARENA_MAX_PRICE, (
            max(float(v["input_token_price_per_million"]) for v in table.values()),
            max(float(v["output_token_price_per_million"]) for v in table.values())))
        for name in ra.PROPOSED_PROFILES:
            self.assertNotIn(name, table)

    def test_submission_names_are_known_or_noted(self):
        source = (CLONE / "universal_model_names.py").read_text(encoding="utf-8")
        self.assertIn('"deepseek/deepseek-v4.1-flash"', source)
        self.assertIn('"z-ai/glm-5.3-flash"', source)
        self.assertNotIn('"z-ai/glm-5.3",', source)  # hence the pr_notes entry


@unittest.skipUnless(os.environ.get("PANDA_BENCH_DOCKER_TESTS") == "1", "set PANDA_BENCH_DOCKER_TESTS=1")
class DockerSandboxTests(unittest.TestCase):
    def test_stdin_and_function_problems(self):
        if not ra.docker_available():
            self.skipTest("Docker or the pinned image is not available")
        stdin_tests = json.dumps([{"input": "2 3\n", "output": "5\n", "testtype": "stdin"},
                                  {"input": "4 4\n", "output": "8\n", "testtype": "stdin"}]).encode()
        good = "a, b = map(int, input().split())\nprint(a + b)\n"
        self.assertEqual(ra.run_code_sandbox(good, stdin_tests, False, 2), {"status": "ran", "score": 1.0,
                                                                             "timed_out": False})
        self.assertEqual(ra.run_code_sandbox("print(5)\n", stdin_tests, False, 2)["score"], 0.0)
        guarded = "def main():\n    print(sum(map(int, input().split())))\nif __name__ == '__main__':\n    main()\n"
        self.assertEqual(ra.run_code_sandbox(guarded, stdin_tests, False, 2)["score"], 1.0)
        # reliability_guard removes os.system, as in RouterArena's harness; sys.exit ends the child.
        self.assertEqual(ra.run_code_sandbox("import os\nos.system('true')\nprint(5)\n", stdin_tests, False, 2)
                         ["score"], 0.0)
        self.assertEqual(ra.run_code_sandbox("import sys\nprint(5)\nsys.exit(0)\n", stdin_tests, False, 2)
                         ["score"], 0.0)
        network = "import socket\nsocket.create_connection(('1.1.1.1', 53), timeout=2)\nprint(5)\n"
        self.assertEqual(ra.run_code_sandbox(network, stdin_tests, False, 2)["score"], 0.0)
        function_tests = json.dumps([{"input": "[1, 2, 3]\n2", "output": "5", "testtype": "functional"}]).encode()
        solution = "def addLast(nums, k):\n    return nums[-1] + k\n"
        self.assertEqual(ra.run_code_sandbox(solution, function_tests, True, 1)["score"], 1.0)
        method = "class Solution:\n    def addLast(self, nums, k):\n        return nums[-1] + k\n"
        self.assertEqual(ra.run_code_sandbox(method, function_tests, True, 1)["score"], 0.0)


if __name__ == "__main__":
    unittest.main()
