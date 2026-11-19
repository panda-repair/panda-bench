"""RouterArena (ICLR 2026), ``sub_10`` split: 809 queries graded the way RouterArena grades them.

RouterArena (https://github.com/RouteWorks/RouterArena) ranks LLM routers by accuracy and by
cost per 1,000 queries.  Each query is one user message built from a per-dataset template;
the router names one model and that model's answer is scored by a per-dataset rule:

* ``mcq_accuracy``: the first ``\\boxed{...}`` (or a fallback pattern) must equal the letter;
* ``math_metric``: string, choice, numeric (with percent variants) and symbolic equivalence;
* ``exact_match``: QANTA-style normalised string equality;
* ``chess_accuracy``: normalised move equality;
* ``superglue_exact_match`` / ``superglue_clozetest``: label equality after a letter map;
* ``meteor_score``: RouterArena's own simplified METEOR, a fraction in [0, 1];
* ``code_accuracy``: LiveCodeBench tests, run here only inside a locked-down container.

Everything below is a line-by-line port of RouterArena's evaluator at commit
``HARNESS_COMMIT``; ``docs/frontier/routerarena.md`` lists the source of every part and the
few places where this port cannot match it (sympy's symbolic equivalence, the container's
limits).  No LLM judge is involved.

Besides ``BENCHMARK`` the module exposes:

* ``cost_per_1k(rows)`` / ``cost_summary(rows)``: dollars per 1,000 queries as RouterArena's
  evaluator computes them, router calls included;
* ``arena_score(cost_per_1k, accuracy)``: the board's "Acc-Cost Arena" number;
* ``write_submission(out_dir, rows)``: the router config and the ``sub_10`` prediction file in
  RouterArena's formats, for a later pull request (nothing is submitted).
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import os
import pickle
import re
import shutil
import subprocess
import threading
import uuid
import zlib
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from panda_bench.frontier.core import Item, JudgeRequest, fetch_pinned, messages_digest, sha256_text

NAME = "routerarena"
# RouterArena's inference pipeline sets no output bound on the OpenRouter/OpenAI-compatible
# path that models like ours take (llm_inference/model_inference.py), so this uses the
# runner's ceiling (see the protocol document).
MAX_OUTPUT_TOKENS = 16384

# --- Pins -------------------------------------------------------------------------------

HARNESS_URL = "https://github.com/RouteWorks/RouterArena"
HARNESS_COMMIT = "00cb25a470165b846a24bf3f2112d22dfc5758ce"  # main on 2026-11-16

DATASET_REPO = "RouteWorks/RouterArena"
DATASET_REVISION = "a4a062ce3313b56bb09c042e1bc37b61d34e3bd8"
SUB10_PATH = "data/sub_10-00000-of-00001.parquet"
SUB10_SHA256 = "e7b4b7579e98df0b83202324357d0b91af78666ddbf1038a513d19edaf58efd5"
SUB10_SIZE = 809
# The full split is read only to rebuild the LiveCodeBench matching exactly as prep_datasets.py
# does (it maps LiveCodeBench rows against all 385 full-split coding questions).
FULL_PATH = "data/full-00000-of-00001.parquet"
FULL_SHA256 = "358a6bb4160320b5cb5b68d6d0a885c89b128b06d3dea267d77ed6daeebe4369"

# RouterArena's coding questions are LiveCodeBench release_v2 problems; their prompt variant
# (stdin or function) and their hidden tests come from this dataset.
LCB_REPO = "lighteval/code_generation_lite"
LCB_REVISION = "89e5fc5c2a8e748f50e95bc7235fab2372d49bfa"
LCB_SHARD_0_SHA256 = "5939509129bd25dc49a16f276005ad006c810aba528bc153e9ec8155edd617e0"
LCB_SHARD_1_SHA256 = "18e6ed06641db1bc355e4f2ff150fecbd86d5cf5fe634db974f3777911c25b66"
LCB_SHARD_2_SHA256 = "24ec55689daad5da2abc86fe1fac4cd3893783d97db27ff26316b639cbe777a5"
LCB_SHARD_3_SHA256 = "e0cdb54a3f965e86cbd909fb7f4c3fa3b304472878bb0c5805026045989bdbbe"
LCB_SHARDS = (
    ("release_v2/test-00000-of-00004.parquet", LCB_SHARD_0_SHA256),
    ("release_v2/test-00001-of-00004.parquet", LCB_SHARD_1_SHA256),
    ("release_v2/test-00002-of-00004.parquet", LCB_SHARD_2_SHA256),
    ("release_v2/test-00003-of-00004.parquet", LCB_SHARD_3_SHA256),
)
# Digest of the manifest this module derives from the four shards for the 38 coding questions
# of sub_10; it pins every derived test file (each listed with its own SHA-256).
LCB_MANIFEST_SHA256 = "de299fbe80dde8e43f408c818b6ee8056602c39efb33c14a4a7809e939930803"
LCB_SHARD_MAX_BYTES = 1_000_000_000

# The container that runs model-written code (RouterArena's CI uses Python 3.11; this is the
# closest image present locally).  ``--pull never``: grading never downloads anything.
SANDBOX_IMAGE_SHA256 = "a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1"  # Python 3.12.15
SANDBOX_IMAGE = "python:3.12-slim@sha256:" + SANDBOX_IMAGE_SHA256
SANDBOX_FLAGS = (
    "--network", "none", "--read-only", "--memory", "512m", "--cpus", "1", "--pids-limit", "128",
    "--tmpfs", "/tmp:rw,nosuid,nodev,size=64m", "--cap-drop", "ALL",
    "--security-opt", "no-new-privileges", "--user", "65534:65534", "--pull", "never",
)
# RouterArena's per-test budget: check_correctness(..., timeout=6) and a join of
# (timeout + 1) * tests + 5 seconds; the container gets extra slack to start and load tests.
LCB_TIMEOUT = 6
SANDBOX_SLACK_SECONDS = 90
# Set to "1" to never start a container (code items are then reported as ungraded).
NO_DOCKER_ENV = "PANDA_BENCH_NO_DOCKER"

_HF = "https://huggingface.co/datasets/{repo}/resolve/{revision}/{path}"

# --- Prompts: config/eval_config/zero-shot/*.json at HARNESS_COMMIT, verbatim ----------------

# Template per prompt dataset (scripts/process_datasets/prep_datasets.py picks it).
PROMPT_TEMPLATES: Dict[str, str] = {
    "AIME": "Please solve the following mathematical problem step by step. \n\nContext: {Context}\n\nQuestion: {Question}\n\nProvide your final answer in \\boxed{{{{}}}} format, where the content inside the braces is the exact mathematical expression or number. For example: \\boxed{{{{42}}}}. Keep your explanation clear, concise, and within 3 sentences.",
    "ArcMMLU": "Please read the following multiple-choice questions and provide the most likely correct answer based on the options given.\n\nContext: {Context}\n\nQuestion: {Question}\n\nOptions: \n{Options}\n\nProvide the correct letter choice in \\boxed{{X}}, where X is the correct letter choice. Keep the explanation or feedback within 3 sentences.",
    "AsDiv": "Please solve the following mathematical problem step by step. \n\nContext: {Context}\n\nQuestion: {Question}\n\nProvide your final answer in \\boxed{{{{}}}} format, where the content inside the braces is the exact mathematical expression or number. For example: \\boxed{{{{42}}}}. Keep your explanation clear, concise, and within 3 sentences.",
    "ChessInstruct": "You are given a question about chess moves. Analyze the context carefully and provide the correct move.\n\nContext:\n{Context}\n\nQuestion:\n{Question}\n\nInstructions:\n- State the correct chess move clearly.\n- Format your final answer as \\boxed{{X}}, where X is the move in standard coordinate notation (e.g., a2a3).\n- Keep your explanation or reasoning within 2\u20133 sentences.",
    "ChessInstruct_mcq": "Please read the following question about chess moves and provide the correct answer.\n\nContext: {Context}\n\nQuestion: {Question}\n\nOptions: \n{Options}\n\nProvide the correct answer in \\boxed{{X}}, where X is the correct letter choice. Keep the explanation or feedback within 3 sentences.",
    "Ethics_commonsense": "Please read the following multiple-choice questions and determine whether the action (in question) is morally acceptable or not.\n\nContext: {Context}\n\nQuestion: {Question}\n\nOptions: \n{Options}\n\nProvide the correct letter choice in \\boxed{{X}}, where X is the correct letter choice. Keep the explanation or feedback within 3 sentences.",
    "Ethics_deontology": "Please read the following multiple-choice questions and consider which option best follows deontological ethics.\n\nContext: {Context}\n\nQuestion: {Question}\n\nOptions: \n{Options}\n\nProvide the correct letter choice in \\boxed{{X}}, where X is the correct letter choice. Keep the explanation or feedback within 3 sentences.",
    "Ethics_justice": "Please read the following multiple-choice questions and determine whether the action (in question) is morally acceptable or not.\n\nContext: {Context}\n\nQuestion: {Question}\n\nOptions: \n{Options}\n\nProvide the correct letter choice in \\boxed{{X}}, where X is the correct letter choice. Keep the explanation or feedback within 3 sentences.",
    "Ethics_virtue": "Please read the following multiple-choice questions and determine which virtue or vice best describes the action in the scenario.\n\nContext: {Context}\n\nQuestion: {Question}\n\nOptions: \n{Options}\n\nProvide the correct letter choice in \\boxed{{X}}, where X is the correct letter choice. Keep the explanation or feedback within 3 sentences.",
    "FinQA": "Please solve the following mathematical problem step by step. Provide the final answer.\n\nContext: {Context}\n\nQuestion: {Question}\n\nProvide your final answer in \\boxed{{{{}}}} format, where the content inside the braces is the exact mathematical expression or number. For example: \\boxed{{{{42}}}}. Keep your explanation clear, concise, and within 3 sentences.",
    "GPQA": "Please read the following multiple-choice questions and provide the most likely correct answer based on the options given.\n\nContext: {Context}\n\nQuestion: {Question}\n\nOptions: \n{Options}\n\nProvide the correct letter choice in \\boxed{{X}}, where X is the correct letter choice. Keep the explanation or feedback within 3 sentences.",
    "GSM8K": "Please solve the following mathematical problem and provide the final answer.\n\nContext: {Context}\n\nQuestion: {Question}\n\nProvide your final answer in \\boxed{{{{}}}} format, where the content inside the braces is the exact mathematical expression or number. For example: \\boxed{{{{42}}}}. Keep your explanation clear, concise, and within 3 sentences.",
    "GeoBench": "Please read the following multiple-choice questions and provide the most likely correct answer based on the options given.\n\nContext: {Context}\n\nQuestion: {Question}\n\nOptions: \n{Options}\n\nProvide the correct letter choice in \\boxed{{X}}, where X is the correct letter choice. Keep the explanation or feedback within 3 sentences.",
    "GeoGraphyData": "Please read the following question and provide the correct answer.\n\nContext: {Context}\n\nQuestion: {Question}\n\nProvide the correct answer in \\boxed{{X}}, where X is the correct and the most common answer to the question. Keep the explanation or feedback within 3 sentences.",
    "MATH": "Please solve the following mathematical problem and provide the final answer.\n\nContext: {Context}\n\nQuestion: {Question}\n\nProvide your final answer in \\boxed{{{{}}}} format, where the content inside the braces is the exact mathematical expression or number. For example: \\boxed{{{{42}}}}. Keep your explanation clear, concise, and within 3 sentences.",
    "MMLU": "Please read the following multiple-choice questions and provide the most likely correct answer based on the options given.\n\nContext: {Context}\n\nQuestion: {Question}\n\nOptions: \n{Options}\n\nProvide the correct letter choice in \\boxed{{X}}, where X is the correct letter choice. Keep the explanation or feedback within 3 sentences.",
    "MMLUPro": "Please read the following multiple-choice questions and provide the most likely correct answer based on the options given.\n\nContext: {Context}\n\nQuestion: {Question}\n\nOptions: \n{Options}\n\nProvide the correct letter choice in \\boxed{{X}}, where X is the correct letter choice. Keep the explanation or feedback within 3 sentences.",
    "MathQA": "Please read the following multiple-choice questions and provide the most likely correct answer based on the options given.\n\nContext: {Context}\n\nQuestion: {Question}\n\nOptions: \n{Options}\n\nProvide the correct letter choice in \\boxed{{X}}, where X is the correct letter choice. Keep the explanation or feedback within 3 sentences.",
    "MedMCQA": "Please read the following multiple-choice questions and provide the most likely correct answer based on the options given.\n\nContext: {Context}\n\nQuestion: {Question}\n\nOptions: \n{Options}\n\nProvide the correct letter choice in \\boxed{{X}}, where X is the correct letter choice. Keep the explanation or feedback within 3 sentences.",
    "MusicTheoryBench": "Please read the following multiple-choice questions and provide the most likely correct answer based on the options given.\n\nContext: {Context}\n\nQuestion: {Question}\n\nOptions: \n{Options}\n\nProvide the correct letter choice in \\boxed{{X}}, where X is the correct letter choice. Keep the explanation or feedback within 3 sentences.",
    "NarrativeQA": "Please read the following context and answer the question based on its content.\n\nContext: {Context}\n\nQuestion: {Question}\n\nProvide your final answer in \\boxed{{{{}}}} format. Keep your explanation clear, concise, and within 3 sentences.",
    "OpenTDB": "Please read the following multiple-choice questions and provide the most likely correct answer based on the options given.\n\nContext: {Context}\n\nQuestion: {Question}\n\nOptions: \n{Options}\n\nProvide the correct letter choice in \\boxed{{X}}, where X is the correct letter choice. Keep the explanation or feedback within 3 sentences.",
    "PubMedQA": "Please read the following multiple-choice questions and provide the most likely correct answer based on the options given.\n\nContext: {Context}\n\nQuestion: {Question}\n\nOptions: \n{Options}\n\nProvide the correct letter choice in \\boxed{{X}}, where X is the correct letter choice. Keep the explanation or feedback within 3 sentences.",
    "QANTA": "Please read the following question and provide the correct answer.\n\nContext: {Context}\n\nQuestion: {Question}\n\nProvide the correct answer in \\boxed{{X}}, where X is the correct and the most common answer to the question. Keep the explanation or feedback within 3 sentences.",
    "SocialiQA": "Please read the following multiple-choice questions and provide the most likely correct answer based on the options given.\n\nContext: {Context}\n\nQuestion: {Question}\n\nOptions: \n{Options}\n\nProvide the correct letter choice in \\boxed{{X}}, where X is the correct letter choice. Keep the explanation or feedback within 3 sentences.",
    "SuperGLUE-CausalReasoning": "Please read the following multiple-choice questions and provide the most likely correct answer based on the options given.\n\nContext: {Context}\n\nQuestion: {Question}\n\nOptions: \n{Options}\n\nProvide the correct letter choice in \\boxed{{X}}, where X is the correct letter choice. Keep the explanation or feedback within 3 sentences.",
    "SuperGLUE-ClozeTest": "Read the following passage and answer the question by choosing the best option. Provide only the text of the correct option as your answer.\n\n{Question}\n\nOptions : {Options}\n\nProvide the correct letter choice in \\boxed{{X}}, where X is the contetn of the correct option. ",
    "SuperGLUE-Entailment": "You are a Natural Language Inference expert. Analyze the given \"Premise\" and \"Hypothesis\".\n\nYou must determine if the Premise entails the Hypothesis. First, explain your reasoning in one or two sentences on why the Hypothesis can or cannot be inferred from the Premise. Then, provide your final judgment: `0.0` for entailment, `1.0` for no entailment.\n\nQuestion: {Question}\n\nProvide the correct letter choice in \\boxed{{X}}, where X is the correct choice. ",
    "SuperGLUE-QA": "You are a reading comprehension assistant. Your task is to answer a question based on the provided context.\n\nFirst, provide your final judgment: `1` for correct, `0` for incorrect.\n\nContext: {Context}\n\nQuestion: {Question}\n\nProvide the correct letter choice in \\boxed{{X}}, where X is the correct choice. ",
    "SuperGLUE-RC": "Your task is to evaluate if the \"Provided Answer\" is a correct response to the \"Question\" based on the \"Paragraph\".\\n- Output 1 if the answer is correct.- Output 0 if the answer is incorrect.\n\nQuestion: {Question}\n\nProvided Answer: {Answer}\n\n Is the provided answer correct? Provide the correct letter choice in \\boxed{{X}}, where X is the correct choice. ",
    "SuperGLUE-Wic": "Consider the word \"{Context}\" in the two sentences below. Does the word have the same meaning in both sentences? Answer with only \"Yes\" or \"No\".\n\n{Question}\n\nProvide the correct letter choice in \\boxed{{X}}, where X is the correct choice. ",
    "SuperGLUE-Wsc": "In the \"Text\" below, does the pronoun (\"span2_text\") refer to the noun phrase (\"span1_text\")?\n\nAnswer with only \"Yes\" or \"No\".\n\n{Question}\n\nProvide the correct letter choice in \\boxed{{X}}, where X is the correct choice. ",
    "WMT19-cs-en": "Translate the following sentence from Englisb to Czech.\n\n{Question}\n\nProvide your final answer in \\boxed{{{{}}}} format.",
    "WMT19-de-en": "Translate the following sentence from English to German.\n\n{Question}\n\nProvide your final answer in \\boxed{{{{}}}} format.",
    "WMT19-fi-en": "Translate the following sentence from English to Finnish.\n\n{Question}\n\nProvide your final answer in \\boxed{{{{}}}} format.",
    "WMT19-gu-en": "Translate the following sentence from English to Gujarati.\n\n{Question}\n\nProvide your final answer in \\boxed{{{{}}}} format.",
    "WMT19-kk-en": "Translate the following sentence from English to Kazakh.\n\n{Question}\n\nProvide your final answer in \\boxed{{{{}}}} format.",
    "WMT19-lt-en": "Translate the following sentence from English to Lithuanian.\n\n{Question}\n\nProvide your final answer in \\boxed{{{{}}}} format.",
    "WMT19-ru-en": "Translate the following sentence from English to Russian.\n\n{Question}\n\nProvide your final answer in \\boxed{{{{}}}} format.",
    "WMT19-zh-en": "Translate the following sentence from English to Chinese.\n\n{Question}\n\nProvide your final answer in \\boxed{{{{}}}} format.",
}
LCB_STDIN_TEMPLATE = "Generate an executable Python function generated from the given prompt. The function should take stdin as input and print the output. Simply call the function after the definition. Question:\n{Question}"
LCB_FUNCTIONAL_TEMPLATE = "Generate an executable Python function generated from the given prompt. Return the function body without invoking it at the final solution. Question:\n{Question}"

# Each config's first ``eval_metrics`` entry (every config at the pin lists exactly one).
CONFIG_METRICS: Dict[str, str] = {
    "AIME": "math_metric", "ArcMMLU": "mcq_accuracy", "AsDiv": "math_metric",
    "ChessInstruct": "chess_accuracy", "ChessInstruct_mcq": "mcq_accuracy",
    "Ethics_commonsense": "mcq_accuracy", "Ethics_deontology": "mcq_accuracy",
    "Ethics_justice": "mcq_accuracy", "Ethics_virtue": "mcq_accuracy", "FinQA": "math_metric",
    "GPQA": "mcq_accuracy", "GSM8K": "math_metric", "GeoBench": "mcq_accuracy",
    "GeoGraphyData": "exact_match", "LiveCodeBench": "code_accuracy", "MATH": "math_metric",
    "MMLU": "mcq_accuracy", "MMLUPro": "mcq_accuracy", "MathQA": "mcq_accuracy",
    "MedMCQA": "mcq_accuracy", "MusicTheoryBench": "mcq_accuracy", "NarrativeQA": "meteor_score",
    "OpenTDB": "mcq_accuracy", "PubMedQA": "mcq_accuracy", "QANTA": "exact_match",
    "SocialiQA": "mcq_accuracy", "SuperGLUE-CausalReasoning": "mcq_accuracy",
    "SuperGLUE-ClozeTest": "superglue_clozetest", "SuperGLUE-Entailment": "superglue_exact_match",
    "SuperGLUE-QA": "superglue_exact_match", "SuperGLUE-RC": "superglue_exact_match",
    "SuperGLUE-Wic": "superglue_exact_match", "SuperGLUE-Wsc": "superglue_exact_match",
    "WMT19-cs-en": "meteor_score", "WMT19-de-en": "meteor_score", "WMT19-fi-en": "meteor_score",
    "WMT19-gu-en": "meteor_score", "WMT19-kk-en": "meteor_score", "WMT19-lt-en": "meteor_score",
    "WMT19-ru-en": "meteor_score", "WMT19-zh-en": "meteor_score",
}

# llm_evaluation/evaluate_models.py, ModelEvaluator.determine_dataset_from_global_index.
# Grading picks the dataset from the global index, not from the prompt template: every
# ChessInstruct item (multiple-choice ones included) is graded with chess_accuracy.
_EVAL_DATASET = {
    "AIME": "AIME", "ArcMMLU": "MMLUPro", "AsDiv": "AsDiv", "ChessInstruct": "ChessInstruct",
    "Ethics": "Ethics_commonsense", "FinQA": "FinQA", "GeoBench": "GeoBench",
    "GeoGraphyData": "GeoGraphyData_100k", "GPQA": "GPQA", "GSM8K": "GSM8K",
    "LiveCodeBench": "LiveCodeBench", "MATH": "MATH", "MathQA": "MathQA", "MedMCQA": "MedMCQA",
    "MMLUPro": "MMLUPro", "MMLU": "MMLU", "MusicTheoryBench": "MusicTheoryBench",
    "NarrativeQA": "NarrativeQA", "OpenTDB": "OpenTDB", "PubMedQA": "PubMedQA", "QANTA": "QANTA",
    "SocialiQA": "SocialiQA", "SuperGLUE": "SuperGLUE-RC", "WMT19": "WMT19-cs-en",
}
# llm_evaluation/eval_reasoning.py, dataset2metric (used only when a dataset has no config).
_FALLBACK_METRICS = {
    "lsat-ar": "mcq_exact_match", "AIME2024": "math_metric", "MATH-500": "math_metric",
    "MMLUPro": "mcq_accuracy", "livecodebench": "code_accuracy", "OpenTDB": "mcq_accuracy",
    "MATH": "math_metric", "NarrativeQA": "meteor_score",
}

_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def escape_format_braces(text: Any) -> Any:
    """prep_datasets.escape_format_braces, verbatim.  Its output is inserted into the template
    as a ``str.format`` argument, so RouterArena's prompts carry every brace of the question
    doubled (and a doubled one unchanged); this port keeps that."""
    if not isinstance(text, str):
        return text
    result = ""
    i = 0
    while i < len(text):
        if text[i] == "{":
            if i + 1 < len(text) and text[i + 1] == "{":
                result += "{{"
                i += 2
            else:
                result += "{{"
                i += 1
        elif text[i] == "}":
            if i + 1 < len(text) and text[i + 1] == "}":
                result += "}}"
                i += 2
            else:
                result += "}}"
                i += 1
        else:
            result += text[i]
            i += 1
    return result


def _safe_format_prompt(template: str, **kwargs: Any) -> str:
    return template.format(**{k: escape_format_braces(v) for k, v in kwargs.items()})


def prompt_dataset(row: Mapping[str, Any]) -> str:
    """The template a dataset row gets (prep_datasets.build_formatted_prompts_from_router_eval_benchmark)."""
    if not row.get("Dataset name"):
        parts = row["Global Index"].split("_")
        if parts[0] in ("Ethics", "ChessInstruct") and len(parts) >= 2:
            full = parts[0] + "_" + parts[1]
        else:
            full = parts[0]
    else:
        full = row.get("Dataset name")
    options = row.get("Options")
    has_options = options is not None and len(options) > 0
    if "Ethics" in full:
        return full
    if "ChessInstruct" in full:
        return "ChessInstruct_mcq" if has_options else "ChessInstruct"
    return str(full).split("_", 1)[0]


def format_prompt(row: Mapping[str, Any], is_stdin: Optional[bool] = None) -> str:
    """The exact user message RouterArena sends for a dataset row (``prompt_formatted``).

    ``is_stdin`` is LiveCodeBench's own flag (any public test of type "stdin") and is
    required for coding rows."""
    base = prompt_dataset(row)
    options = row.get("Options")
    options_str = ""
    if options:
        for i, option in enumerate(options):
            letter = _LETTERS[i] if i < len(_LETTERS) else "-"
            options_str += "%s. %s\n" % (letter, option)
    question = row.get("Question", "")
    context = row.get("Context", "")
    context_for_prompt = context if context != "" else "None"
    if base == "LiveCodeBench":
        if is_stdin is None:
            raise ValueError("a LiveCodeBench row needs its is_stdin flag")
        template = LCB_STDIN_TEMPLATE if is_stdin else LCB_FUNCTIONAL_TEMPLATE
        prompt = _safe_format_prompt(template, Question=question)
    elif base == "SuperGLUE-RC":
        prompt = _safe_format_prompt(PROMPT_TEMPLATES[base], Question=question, Answer="")
    elif base == "SuperGLUE-Wic":
        prompt = _safe_format_prompt(PROMPT_TEMPLATES[base], Question=question, Context=context)
    elif not options:
        prompt = _safe_format_prompt(PROMPT_TEMPLATES[base], Context=context_for_prompt, Question=question)
    else:
        prompt = _safe_format_prompt(
            PROMPT_TEMPLATES[base], Context=context_for_prompt, Question=question, Options=options_str
        )
    if len(prompt) > 10000:
        prompt = "%s...%s" % (prompt[:5000], prompt[-5000:])
    if not prompt:
        raise ValueError("empty prompt for " + str(row.get("Global Index")))
    return prompt


def eval_metric(global_index: str) -> Tuple[str, str]:
    """(dataset, scorer) RouterArena's evaluator uses for a global index."""
    prefix = global_index.split("_")[0] if "_" in global_index else global_index
    dataset = _EVAL_DATASET.get(prefix, prefix)
    config = "GeoGraphyData" if dataset == "GeoGraphyData_100k" else dataset
    metric = CONFIG_METRICS.get(config)
    if metric is None:
        metric = _FALLBACK_METRICS.get(dataset, "mcq_accuracy")
    return dataset, metric


# --- Answer extraction: llm_evaluation/enhanced_extractor.py, verbatim ------------------------

_PATTERNS = {
    "mcq_letter": r"\b([A-Z])\b(?!\w)",
    "mcq_option": r"(?:option|choice|select)[:\s]*([A-Z])",
    "mcq_letter_alt": r"(?:the\s+)?(?:answer|choice)\s*[:\-]\s*([A-Z])",
    "mcq_answer_colon": r"answer\s*:\s*([A-Z])",
    "math_answer": r"(?:the\s+)?(?:final\s+)?answer\s*[:\-]\s*([^\n]+)",
    "math_result": r"(?:result|solution)\s*[:\-]\s*([^\n]+)",
    "math_number": r"(?:answer|result)\s*[:\-]\s*(\d+(?:\.\d+)?)",
    "math_is": r"(?:the\s+)?(?:answer|result)\s+is\s+([^\n]+)",
    "python_code": r"```python\s*\n(.*?)\n```",
    "code_block": r"```(?:python)?\s*\n(.*?)\n```",
    "code_function": r"def\s+\w+\([^)]*\):\s*\n(.*?)(?=\n\w|\n```|\Z)",
    "quoted": r'["\']([^"\']+)["\']',
    "parentheses": r"\(([^)]+)\)",
    "brackets": r"\[([^\]]+)\]",
    "colon_answer": r"(?:answer|result|solution)[:\s]+([^\n]+)",
    "chess_move": r"\b([a-h][1-8][a-h][1-8])\b",
    "chess_move_alt": r"(?:move|play)[:\s]*([a-h][1-8][a-h][1-8])",
}
_CLEANUP_PATTERNS = [
    r"^(?:the\s+)?(?:answer|result|solution)\s*[:\-]\s*",
    r"^(?:option|choice)\s*[:\-]\s*",
    r"^(?:letter|choice)\s*[:\-]\s*",
    r"\s*[:\-]\s*(?:is\s+)?the\s+(?:answer|result|solution)\.?$",
    r"\s*[:\-]\s*(?:is\s+)?correct\.?$",
    r"^[:\-\s]+",
    r"[:\-\s]+$",
    r"\.$",
    r'^["\']|["\']$',
    r"^(?:the\s+)?(?:final\s+)?answer\s+is\s+",
    r"^(?:the\s+)?answer\s+is\s+",
    r"^(?:the\s+)?result\s+is\s+",
]


def _has_boxed(text: str) -> bool:
    return bool(text) and bool(re.search(r"\\boxed\{", text))


def _standard_boxed(text: str) -> Optional[str]:
    text = text.replace("\\text{", "")
    matches = re.findall(r"\\boxed\{([^}]+)\}", text)
    if matches:
        return matches[0].strip()
    return None


def _standard_math_boxed(text: str) -> Optional[str]:
    start_match = re.search(r"\\boxed\{", text)
    if start_match:
        start_pos = start_match.end() - 1
        brace_count = 0
        for i, char in enumerate(text[start_pos:], start_pos):
            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0:
                    return text[start_pos + 1:i].strip()
    return None


def _clean_answer(answer: str) -> str:
    if not answer:
        return answer
    cleaned = answer.strip()
    for pattern in _CLEANUP_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
    cleaned = " ".join(cleaned.split())
    return cleaned.strip()


def _code_answer(text: str) -> Optional[str]:
    matches = re.findall(_PATTERNS["python_code"], text, re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    matches = re.findall(_PATTERNS["code_block"], text, re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    matches = re.findall(_PATTERNS["code_function"], text, re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    return None


def _mcq_answer(text: str) -> Optional[str]:
    matches = re.findall(_PATTERNS["mcq_answer_colon"], text, re.IGNORECASE)
    if matches:
        return matches[0].upper()
    matches = re.findall(_PATTERNS["mcq_option"], text, re.IGNORECASE)
    if matches:
        return matches[0].upper()
    matches = re.findall(_PATTERNS["mcq_letter_alt"], text, re.IGNORECASE)
    if matches:
        return matches[0].upper()
    matches = re.findall(_PATTERNS["mcq_letter"], text)
    if matches:
        return matches[-1]
    return None


_MATH_PATTERN_KEYS = ("math_answer", "math_result", "math_number", "math_is")


def _math_pattern_answer(text: str) -> Optional[str]:
    for key in _MATH_PATTERN_KEYS:
        matches = re.findall(_PATTERNS[key], text, re.IGNORECASE)
        if matches:
            cleaned = _clean_answer(matches[0].strip())
            if cleaned:
                return cleaned
    return None


def _general_answer(text: str) -> Optional[str]:
    matches = re.findall(_PATTERNS["quoted"], text)
    if matches:
        return matches[-1].strip()
    matches = re.findall(_PATTERNS["parentheses"], text)
    if matches:
        return matches[-1].strip()
    matches = re.findall(_PATTERNS["brackets"], text)
    if matches:
        return matches[-1].strip()
    matches = re.findall(_PATTERNS["colon_answer"], text, re.IGNORECASE)
    if matches:
        cleaned = _clean_answer(matches[-1].strip())
        if cleaned:
            return cleaned
    return None


def _enhanced_answer(text: str, dataset: Optional[str]) -> Optional[str]:
    if dataset == "LiveCodeBench":
        return _code_answer(text)
    if dataset == "SuperGLUE-ClozeTest":
        return text[0].upper()
    for extractor in (_mcq_answer, _math_pattern_answer, _general_answer):
        result = extractor(text)
        if result:
            return result
    return None


def _fallback_answer(text: str) -> str:
    matches = re.findall(r"\b([A-Z])\b", text)
    if matches:
        return matches[-1]
    return text.strip()


def extract_boxed_answer(text: str, dataset: Optional[str] = None) -> str:
    """EnhancedAnswerExtractor.extract_boxed_answer."""
    if not text or not text.strip():
        return ""
    text = text.strip()
    if _has_boxed(text):
        result = _standard_boxed(text)
        if result:
            return result
    result = _enhanced_answer(text, dataset)
    if result:
        return result
    return _fallback_answer(text)


def extract_math_answer(text: str) -> str:
    """EnhancedAnswerExtractor.extract_math_answer."""
    if not text or not text.strip():
        return ""
    text = text.strip()
    if _has_boxed(text):
        result = _standard_math_boxed(text)
        if result:
            return result
    result = _math_pattern_answer(text)
    if result:
        return result
    for pattern in (r"\b(\d+(?:\.\d+)?)\b", r"\b(\d+(?:\.\d+)?)\s*$"):
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1]
    return text.strip()


def _extractor_chess_normalize(move: str) -> str:
    if not move:
        return ""
    move = str(move).strip().lower()
    move = move.replace("-", "").replace(" ", "").replace("_", "")
    return re.sub(r"[^a-z0-9]", "", move)


def extract_chess_move(text: str) -> str:
    """EnhancedAnswerExtractor.extract_chess_move."""
    if not text or not text.strip():
        return ""
    text = text.strip()
    if _has_boxed(text):
        result = _standard_boxed(text)
        if result:
            return _extractor_chess_normalize(result)
    matches = re.findall(_PATTERNS["chess_move"], text, re.IGNORECASE)
    if matches:
        return _extractor_chess_normalize(matches[-1])
    matches = re.findall(_PATTERNS["chess_move_alt"], text, re.IGNORECASE)
    if matches:
        return _extractor_chess_normalize(matches[-1])
    return _extractor_chess_normalize(_fallback_answer(text))


# --- Scorers: llm_evaluation/metrics.py and metric_utils.py ----------------------------------
# Each takes one generated answer and one reference and returns (score, publishable extras),
# as metrics.<scorer>([answer], [reference]) does through ModelEvaluator._evaluate_single_entry.


_UPPER = frozenset(_LETTERS)
_LOWER = frozenset(_LETTERS.lower())


def _letter_of(gt: Any) -> str:
    """mcq_accuracy's reading of a reference: a letter, or a 0-based option index."""
    if gt in _UPPER:
        return gt
    if gt in _LOWER:
        return gt.upper()
    gt_int = int(float(gt))
    if gt_int in range(0, 26):
        return _LETTERS[gt_int]
    raise ValueError("ground truth index out of range")


def _public_letter(extracted: str) -> Optional[str]:
    return extracted if extracted in _UPPER else None


def score_mcq_accuracy(answer: str, gt: str) -> Tuple[float, Dict[str, Any]]:
    try:
        gt_letter = _letter_of(gt)
        extracted = extract_boxed_answer(answer, "MCQ")
    except Exception:
        # metrics.mcq_accuracy catches any per-question error and counts it wrong.
        return 0.0, {"reference_error": True}
    return float(extracted == gt_letter), {"letter": _public_letter(extracted)}


def score_mcq_exact_match(answer: str, gt: str) -> Tuple[float, Dict[str, Any]]:
    extracted = extract_boxed_answer(answer, "MCQ")
    return float(extracted == gt), {"letter": _public_letter(extracted)}


def _choice_answer_clean(pred: str) -> str:
    cleaned = pred.strip("\n").rstrip(".").rstrip("/").strip(" ").lstrip(":")
    matches = re.findall(r"\b(A|B|C|D|E)\b", cleaned.upper())
    if matches:
        result = matches[-1]
    else:
        result = cleaned.strip().strip(".")
    return result.rstrip(".").rstrip("/")


def _parse_digits(num: Any) -> Optional[float]:
    normalized = str(num).replace(",", "")
    try:
        return float(normalized)
    except Exception:
        if normalized.endswith("%"):
            normalized = normalized[:-1]
            if normalized.endswith("\\"):
                normalized = normalized[:-1]
            try:
                return float(normalized) / 100
            except Exception:
                pass
    return None


def _is_digit(num: Any) -> bool:
    return _parse_digits(num) is not None


def _numeric_equal(prediction: float, reference: float) -> bool:
    return math.isclose(reference, prediction, rel_tol=1e-4)


def _str_to_pmatrix(input_str: str) -> str:
    stripped = input_str.strip()
    pmatrix_list = []
    for m in re.findall(r"\{.*,.*\}", stripped):
        content = m.strip("{}")
        pmatrix_list.append(r"\begin{pmatrix}" + content.replace(",", "\\") + r"\end{pmatrix}")
    return ", ".join(pmatrix_list)


def _symbolic_equal(a: str, b: str) -> Optional[bool]:
    """metric_utils.symbolic_equal needs sympy and latex2sympy2.  Identical strings parse to
    identical expressions, so that case is True; anything else is left undecided (None)."""
    if a == b:
        return True
    return None


def _all3(values: Iterable[Optional[bool]]) -> Optional[bool]:
    values = list(values)
    if any(v is False for v in values):
        return False
    if any(v is None for v in values):
        return None
    return True


def math_equal(prediction: Any, reference: Any, include_percentage: bool = True,
               is_close: bool = True) -> Optional[bool]:
    """metrics.math_equal, branch by branch.  True or False wherever the official function
    decides without sympy; None where its verdict would come from sympy's symbolic_equal."""
    if prediction is None or reference is None:
        return False
    if str(prediction.strip().lower()) == str(reference.strip().lower()):
        return True
    if reference in ["A", "B", "C", "D", "E"] and _choice_answer_clean(prediction) == reference:
        return True
    try:
        if _is_digit(prediction) and _is_digit(reference):
            pred_val = _parse_digits(prediction)
            ref_val = _parse_digits(reference)
            if pred_val is not None and ref_val is not None:
                gt_result = [ref_val / 100, ref_val, ref_val * 100] if include_percentage else [ref_val]
                for item in gt_result:
                    try:
                        if is_close:
                            if _numeric_equal(pred_val, item):
                                return True
                        elif item == pred_val:
                            return True
                    except Exception:
                        continue
                return False
    except Exception:
        pass
    if not prediction and prediction not in [0, False]:
        return False

    reference = str(reference).strip()
    prediction = str(prediction).strip()
    if "pmatrix" in prediction and "pmatrix" not in reference:
        reference = _str_to_pmatrix(reference)
    pred_str, ref_str = prediction, reference
    if (prediction.startswith("[") and prediction.endswith("]") and not reference.startswith("(")) or (
        prediction.startswith("(") and prediction.endswith(")") and not reference.startswith("[")
    ):
        pred_str = pred_str.strip("[]()")
        ref_str = ref_str.strip("[]()")
    for s in ["{", "}", "(", ")"]:
        ref_str = ref_str.replace(s, "")
        pred_str = pred_str.replace(s, "")
    if pred_str.lower() == ref_str.lower():
        return True

    unresolved = False
    if (re.match(r"(\(|\[).+(\)|\])", prediction) is not None
            and re.match(r"(\(|\[).+(\)|\])", reference) is not None):
        pred_parts = prediction[1:-1].split(",")
        ref_parts = reference[1:-1].split(",")
        if len(pred_parts) == len(ref_parts):
            verdict = _all3([math_equal(pred_parts[i], ref_parts[i], include_percentage, is_close)
                             for i in range(len(pred_parts))])
            if verdict is True:
                return True
            unresolved = unresolved or verdict is None
    if ((prediction.startswith("\\begin{pmatrix}") or prediction.startswith("\\begin{bmatrix}"))
            and (prediction.endswith("\\end{pmatrix}") or prediction.endswith("\\end{bmatrix}"))
            and (reference.startswith("\\begin{pmatrix}") or reference.startswith("\\begin{bmatrix}"))
            and (reference.endswith("\\end{pmatrix}") or reference.endswith("\\end{bmatrix}"))):
        cut = slice(len("\\begin{pmatrix}"), -len("\\end{pmatrix}"))
        pred_lines = [line.strip() for line in prediction[cut].split("\\\\") if line.strip()]
        ref_lines = [line.strip() for line in reference[cut].split("\\\\") if line.strip()]
        matched: Optional[bool] = True
        if len(pred_lines) == len(ref_lines):
            for pred_line, ref_line in zip(pred_lines, ref_lines):
                pred_parts = pred_line.split("&")
                ref_parts = ref_line.split("&")
                if len(pred_parts) != len(ref_parts):
                    matched = False
                    break
                verdict = _all3([math_equal(pred_parts[i], ref_parts[i], include_percentage, is_close)
                                 for i in range(len(pred_parts))])
                if verdict is False:
                    matched = False
                    break
                if verdict is None:
                    matched = None
        else:
            matched = False
        if matched is True:
            return True
        unresolved = unresolved or matched is None
    if prediction.count("=") == 1 and reference.count("=") == 1:
        pred = prediction.split("=")
        pred_expr = "%s - (%s)" % (pred[0].strip(), pred[1].strip())
        ref = reference.split("=")
        ref_expr = "%s - (%s)" % (ref[0].strip(), ref[1].strip())
        verdicts = [_symbolic_equal(pred_expr, ref_expr), _symbolic_equal("-(%s)" % pred_expr, ref_expr)]
        if True in verdicts:
            return True
        unresolved = unresolved or None in verdicts
    elif (prediction.count("=") == 1 and len(prediction.split("=")[0].strip()) <= 2
          and "=" not in reference):
        verdict = math_equal(prediction.split("=")[1], reference, include_percentage, is_close)
        if verdict is True:
            return True
        unresolved = unresolved or verdict is None
    elif (reference.count("=") == 1 and len(reference.split("=")[0].strip()) <= 2
          and "=" not in prediction):
        verdict = math_equal(prediction, reference.split("=")[1], include_percentage, is_close)
        if verdict is True:
            return True
        unresolved = unresolved or verdict is None
    verdict = _symbolic_equal(prediction, reference)
    if verdict is True:
        return True
    if verdict is None or unresolved:
        return None
    return False


def score_math_metric(answer: str, gt: str) -> Tuple[float, Dict[str, Any]]:
    try:
        extracted = extract_math_answer(answer)
        if len(extracted.split("=")) > 1:
            extracted = extracted.split("=")[-1].strip()
        verdict = math_equal(extracted, gt)
    except Exception:
        return 0.0, {"math_check": "decided"}
    if verdict is None:
        return 0.0, {"math_check": "unresolved"}
    return float(verdict), {"math_check": "decided"}


def _normalize_qanta_answer(answer: Any) -> str:
    if not answer:
        return ""
    answer = str(answer).strip().replace("_", " ").lower()
    return " ".join(answer.split())


def score_exact_match(answer: str, gt: str) -> Tuple[float, Dict[str, Any]]:
    extracted = extract_boxed_answer(answer, "QANTA")
    return float(_normalize_qanta_answer(extracted) == _normalize_qanta_answer(gt)), {}


def _normalize_chess_move(move: Any) -> str:
    if not move:
        return ""
    move = str(move).strip().lower()
    if len(move) == 1 and move.isalpha():
        return move
    if move.isdigit():
        numeric_val = int(move)
        if 0 <= numeric_val <= 25:
            return chr(ord("a") + numeric_val)
        return move
    move = move.replace("-", "").replace(" ", "").replace("_", "")
    return re.sub(r"[^a-z0-9]", "", move)


def score_chess_accuracy(answer: str, gt: str) -> Tuple[float, Dict[str, Any]]:
    extracted = extract_chess_move(answer)
    return float(_normalize_chess_move(extracted) == _normalize_chess_move(gt)), {}


def _meteor_tokens(text: str) -> List[str]:
    if not text:
        return []
    text = re.sub(r"[^\w\s]", " ", text.lower())
    if re.search(r"[\u4e00-\u9fff]", text):
        tokens = [char for char in text if char.strip() and char != " "]
    else:
        tokens = text.split()
    return [token for token in tokens if token.strip()]


def _matching_chunks(tokens: Sequence[str], matched_words: set) -> int:
    chunks = 0
    in_chunk = False
    for token in tokens:
        if token in matched_words:
            if not in_chunk:
                chunks += 1
                in_chunk = True
        else:
            in_chunk = False
    return chunks


def meteor(prediction: str, ground_truth: str) -> float:
    """metrics._calculate_single_meteor: RouterArena's own simplified METEOR (no stemming or
    synonyms), with the fragmentation penalty capped at 1 (RouterArena #210)."""
    pred_tokens = _meteor_tokens(prediction)
    gt_tokens = _meteor_tokens(ground_truth)
    if not pred_tokens or not gt_tokens:
        return 0.0
    pred_counter = Counter(pred_tokens)
    gt_counter = Counter(gt_tokens)
    matches = 0
    for word in pred_counter:
        if word in gt_counter:
            matches += min(pred_counter[word], gt_counter[word])
    precision = matches / len(pred_tokens) if len(pred_tokens) > 0 else 0.0
    recall = matches / len(gt_tokens) if len(gt_tokens) > 0 else 0.0
    if precision == 0 and recall == 0:
        return 0.0
    f_score = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    matched_words = set(pred_tokens) & set(gt_tokens)
    if not matched_words:
        penalty = 1.0
    else:
        min_chunks = min(_matching_chunks(pred_tokens, matched_words), _matching_chunks(gt_tokens, matched_words))
        total_matches = len(matched_words)
        if min_chunks == total_matches:
            penalty = 0.0
        else:
            penalty = min(1.0, 0.5 * (min_chunks / total_matches) ** 2)
    return f_score * (1 - penalty)


def score_meteor(answer: str, gt: str) -> Tuple[float, Dict[str, Any]]:
    if not answer or not gt:
        return 0.0, {}
    return meteor(extract_boxed_answer(answer), gt), {}


_LETTER_TO_NUMBER = {"A": "0.0", "0": "0.0", "FALSE": "0.0", "False": "0.0", "false": "0.0",
                     "B": "1.0", "1": "1.0", "TRUE": "1.0", "True": "1.0", "true": "1.0"}
_LETTER_TO_YESNO = {"A": "Yes", "X": "Yes", "1": "Yes", "TRUE": "Yes", "True": "Yes", "true": "Yes",
                    "B": "No", "Y": "No", "0": "No", "FALSE": "No", "False": "No", "false": "No"}


def _convert_superglue_answer(extracted: str, ground_truth: str) -> str:
    if ground_truth in ["0.0", "1.0", "0", "1"]:
        return _LETTER_TO_NUMBER.get(extracted.upper(), extracted)
    if ground_truth in ["Yes", "No", "yes", "no", "YES", "NO"]:
        return _LETTER_TO_YESNO.get(extracted.upper(), extracted)
    return extracted


def score_superglue_exact_match(answer: str, gt: str) -> Tuple[float, Dict[str, Any]]:
    if not answer or not gt:
        return 0.0, {}
    extracted = extract_boxed_answer(answer, "SuperGLUE")
    return float(_convert_superglue_answer(extracted, gt) == gt), {}


def score_superglue_clozetest(answer: str, gt: str) -> Tuple[float, Dict[str, Any]]:
    try:
        if gt in _UPPER:
            gt_letter = gt
        elif gt in _LOWER:
            gt_letter = gt.upper()
        else:
            # A reference that is not an index in range is the option's text, compared as is.
            try:
                gt_int = int(float(gt))
                if gt_int in range(0, 26):
                    gt_letter = _LETTERS[gt_int]
                else:
                    raise ValueError("ground truth index out of range")
            except (ValueError, TypeError):
                gt_letter = gt
        extracted = extract_boxed_answer(answer, "SuperGLUE-ClozeTest")
    except Exception:
        return 0.0, {"reference_error": True}
    return float(extracted == gt_letter), {}


TEXT_SCORERS = {
    "mcq_accuracy": score_mcq_accuracy,
    "mcq_exact_match": score_mcq_exact_match,
    "math_metric": score_math_metric,
    "exact_match": score_exact_match,
    "chess_accuracy": score_chess_accuracy,
    "meteor_score": score_meteor,
    "superglue_exact_match": score_superglue_exact_match,
    "superglue_clozetest": score_superglue_clozetest,
}


# --- LiveCodeBench: livecodebench_util.py, run only inside the container --------------------


def has_code(response: str) -> List[str]:
    return re.findall(r"```(?:[a-zA-Z]*)\n(.*?)```", response, re.DOTALL)


def post_process_code(code: str) -> str:
    code = code.split("</code>")[0]
    code = code.replace("```python", "")
    code = code.split("```")[0]
    return code.replace("<code>", "")


# The program the container runs.  The test functions are livecodebench_util.py's, verbatim;
# the outer process stands in for unsafe_lcb_runTests: it forks one child that runs every test
# in order and stops at the first failure, waits (timeout + 1) * tests + 5 seconds, and scores
# 1 only if every test reported a pass.  The child cannot write to the container's stdout, and
# the verdict line carries a nonce that only the host and the parent know.
SANDBOX_HARNESS = r'''
import builtins, copy, faulthandler, io, json, os, re, select, shutil, signal, subprocess, sys, time
from io import StringIO

LCB_MEMORY_HEADROOM_BYTES = 2 * 1024 ** 3
MAIN_GUARD = re.compile(r"""__name__\s*==\s*(['"])__main__\1|(['"])__main__\2\s*==\s*__name__""")


def prepare_test_input_output_functional(test_case, is_extracted):
    if not is_extracted:
        test_input = test_case["input"]
        test_output = test_case["output"]
        return test_input, test_output
    else:
        input_str = test_case["input"]
        expected_output = test_case["output"].strip()
        inputs = []
        if "=" in input_str:
            if "," in input_str:
                input_parts = [part.strip() for part in input_str.split(",")]
                for part in input_parts:
                    key, value = part.split("=")
                    key = key.strip()
                    value = value.strip()
                    try:
                        value = int(value)
                    except ValueError:
                        try:
                            value = float(value)
                        except ValueError:
                            value = value.strip('"')
                    inputs.append(value)
            else:
                key, value = input_str.split("=")
                key = key.strip()
                value = value.strip()
                try:
                    value = int(value)
                except ValueError:
                    try:
                        value = float(value)
                    except ValueError:
                        value = value.strip('"')
                inputs.append(value)
        else:
            input_parts = input_str.split("\n")
            for part in input_parts:
                part = part.strip()
                if part:
                    if part.startswith('"') and part.endswith('"'):
                        inputs.append(part.strip('"'))
                    elif part.startswith("[") and part.endswith("]"):
                        inputs.append(json.loads(part))
                    else:
                        try:
                            inputs.append(int(part))
                        except ValueError:
                            try:
                                inputs.append(float(part))
                            except ValueError:
                                inputs.append(part)
        try:
            expected_output = json.loads(expected_output)
        except json.JSONDecodeError:
            expected_output = expected_output.strip()
        return inputs, expected_output


def prepare_test_input_output_std(test_case):
    test_input = test_case["input"]
    test_output = test_case["output"].strip()
    if test_output.endswith("-"):
        test_output = test_output[: test_output.rfind("-")].rstrip()
    return test_input, test_output


def run_test_func(completion, is_extracted, test_input, test_output):
    namespace = {}
    if not is_extracted:
        exec(completion, namespace)
        func_name = completion.split("(")[0].split()[-1]
        if isinstance(test_input, dict):
            args = []
            kwargs = test_input
        else:
            args = [test_input]
            kwargs = {}
        output = io.StringIO()
        sys.stdout = output
        try:
            result_output = namespace[func_name](*args, **kwargs)
            if result_output != test_output:
                return False, result_output
        except Exception as e:
            return False, f"Error: {str(e)}"
        finally:
            sys.stdout = sys.__stdout__
        return True, result_output
    else:
        exec(completion, namespace)
        func_name = completion.split("(")[0].split()[-1]
        output = io.StringIO()
        sys.stdout = output
        try:
            result_output = namespace[func_name](*test_input)
            if result_output != test_output:
                return False, result_output
        except Exception as e:
            return False, str(e)
        finally:
            sys.stdout = sys.__stdout__
        return True, result_output


def run_test_std(completion, test_input, test_output):
    sys.stdin = io.TextIOWrapper(io.BytesIO(test_input.encode("utf-8")), encoding="utf-8", newline="\n")
    output = StringIO()
    sys.stdout = output
    if MAIN_GUARD.search(completion):
        completion = '__name__ = "__main__"\n' + completion
    namespace = {}
    exec(completion, namespace)
    output_value = output.getvalue().strip()
    return output_value == test_output, output_value


def limit_test_memory(headroom_bytes):
    if headroom_bytes <= 0:
        return
    try:
        import resource
        with open("/proc/self/status") as status:
            current = next(int(line.split()[1]) * 1024 for line in status if line.startswith("VmSize:"))
        limit = current + headroom_bytes
        _, hard = resource.getrlimit(resource.RLIMIT_AS)
        if hard != resource.RLIM_INFINITY:
            limit = min(limit, hard)
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    except (ImportError, OSError, StopIteration, ValueError):
        pass


def reliability_guard():
    faulthandler.disable()
    os.environ["OMP_NUM_THREADS"] = "1"
    setattr(builtins, "exit", None)
    setattr(builtins, "quit", None)
    for name in ["kill", "system", "putenv", "remove", "removedirs", "rmdir", "fchdir", "setuid",
                 "fork", "forkpty", "killpg", "rename", "renames", "truncate", "replace", "unlink",
                 "fchmod", "fchown", "chmod", "chown", "chroot", "getcwd", "chdir", "lchflags",
                 "lchmod", "lchown"]:
        try:
            setattr(os, name, None)
        except Exception:
            pass
    for name in ["rmtree", "move", "chown"]:
        try:
            setattr(shutil, name, None)
        except Exception:
            pass
    setattr(subprocess, "Popen", None)
    for name in ["ipdb", "joblib", "resource", "psutil", "tkinter"]:
        sys.modules[name] = None


def run_tests_for_one_example(test_cases, completion, out, is_extracted):
    test_type = test_cases[0]["testtype"]
    limit_test_memory(LCB_MEMORY_HEADROOM_BYTES)
    reliability_guard()
    for test_case in test_cases:
        try:
            if test_type == "functional":
                test_input, test_output = prepare_test_input_output_functional(test_case, is_extracted)
                passed, _ = run_test_func(completion, is_extracted, copy.deepcopy(test_input), copy.deepcopy(test_output))
            else:
                test_input, test_output = prepare_test_input_output_std(test_case)
                passed, _ = run_test_std(completion, copy.deepcopy(test_input), copy.deepcopy(test_output))
        except Exception:
            passed = False
        out.write(json.dumps({"passed": bool(passed)}) + "\n")
        out.flush()
        if not passed:
            return


def main():
    data = sys.stdin.buffer.read()
    header, _, tests_raw = data.partition(b"\n")
    del data
    payload = json.loads(header.decode("utf-8"))
    tests = json.loads(tests_raw.decode("utf-8"))
    del header, tests_raw
    nonce = payload.pop("nonce")
    completion, is_extracted, timeout = payload["completion"], payload["is_extracted"], payload["timeout"]
    results, timed_out = [], False
    if tests:
        read_end, write_end = os.pipe()
        pid = os.fork()
        if pid == 0:
            try:
                os.close(read_end)
                devnull = os.open(os.devnull, os.O_RDWR)
                for fd in (0, 1, 2):
                    os.dup2(devnull, fd)
                try:
                    with open("/proc/self/oom_score_adj", "w") as handle:
                        handle.write("1000")
                except OSError:
                    pass
                out = os.fdopen(write_end, "w")
                run_tests_for_one_example(tests, completion, out, is_extracted)
                out.flush()
            except BaseException:
                pass
            finally:
                os._exit(0)
        os.close(write_end)
        deadline = time.monotonic() + (timeout + 1) * len(tests) + 5
        buffer = b""
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                timed_out = True
                break
            ready, _, _ = select.select([read_end], [], [], left)
            if not ready:
                continue
            chunk = os.read(read_end, 1 << 16)
            if not chunk:
                break
            buffer += chunk
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        os.waitpid(pid, 0)
        for line in buffer.decode("utf-8", "replace").split("\n"):
            try:
                results.append(json.loads(line)["passed"] is True)
            except Exception:
                pass
    padded = results + [False] * max(0, len(tests) - len(results))
    verdict = {"score": float(bool(padded and all(padded))), "tests": len(tests),
               "reported": len(results), "timed_out": timed_out}
    sys.__stdout__.write(nonce + json.dumps(verdict) + "\n")
    sys.__stdout__.flush()


main()
'''

_docker_state: Dict[str, Optional[bool]] = {"ok": None}
_docker_lock = threading.Lock()


def docker_available() -> bool:
    """Whether code items can be graded: Docker answers and the pinned image is present."""
    if os.environ.get(NO_DOCKER_ENV) == "1" or shutil.which("docker") is None:
        return False
    with _docker_lock:
        if _docker_state["ok"] is None:
            try:
                done = subprocess.run(["docker", "image", "inspect", SANDBOX_IMAGE],
                                      capture_output=True, timeout=60)
                _docker_state["ok"] = done.returncode == 0
            except (OSError, subprocess.SubprocessError):
                _docker_state["ok"] = False
        return bool(_docker_state["ok"])


def run_code_sandbox(completion: str, tests: bytes, is_extracted: bool, n_tests: int) -> Dict[str, Any]:
    """Run RouterArena's LiveCodeBench test loop on ``completion`` inside the container.

    ``tests`` is the problem's hidden-tests JSON.  The harness, the answer and the tests go in
    through the container's stdin, so nothing is mounted and nothing is written on the host.
    Returns ``{"status": "ran", "score": 0.0 or 1.0, "timed_out": bool}``, or
    ``{"status": "sandbox_error"}`` when the container gave no verdict (a failure of this
    grader, not of the answer), or ``{"status": "unavailable"}`` without Docker."""
    if not docker_available():
        return {"status": "unavailable"}
    name = "panda-bench-ra-" + uuid.uuid4().hex[:16]
    nonce = "RA" + uuid.uuid4().hex
    header = json.dumps({"nonce": nonce, "completion": completion, "is_extracted": is_extracted,
                         "timeout": LCB_TIMEOUT})
    command = (["docker", "run", "--rm", "-i", "--name", name] + list(SANDBOX_FLAGS)
               + [SANDBOX_IMAGE, "python", "-I", "-B", "-c", SANDBOX_HARNESS])
    limit = (LCB_TIMEOUT + 1) * n_tests + 5 + SANDBOX_SLACK_SECONDS
    try:
        done = subprocess.run(command, input=header.encode("utf-8") + b"\n" + tests, capture_output=True,
                              timeout=limit)
    except subprocess.TimeoutExpired:
        try:
            subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=60)
        except (OSError, subprocess.SubprocessError):
            pass
        return {"status": "sandbox_error", "reason": "container_timeout"}
    except (OSError, subprocess.SubprocessError):
        return {"status": "sandbox_error", "reason": "docker_failed"}
    for line in reversed(done.stdout.decode("utf-8", "replace").splitlines()):
        if line.startswith(nonce):
            try:
                verdict = json.loads(line[len(nonce):])
                return {"status": "ran", "score": float(verdict["score"]), "timed_out": bool(verdict["timed_out"])}
            except (ValueError, KeyError, TypeError):
                break
    return {"status": "sandbox_error", "reason": "no_verdict", "exit_code": done.returncode}


# Packages in RouterArena's evaluator environment (uv.lock at the pin) that a solution may
# import.  The official harness forks its evaluator, so they import there; the container has
# only the standard library, so a solution that needs one fails here.
EVALUATOR_ONLY_PACKAGES = ("numpy", "networkx", "pandas", "sympy", "mpmath", "sortedcontainers", "regex", "torch")
_THIRD_PARTY_IMPORT = re.compile(
    r"^\s*(?:import|from)\s+(?:%s)\b" % "|".join(EVALUATOR_ONLY_PACKAGES), re.MULTILINE)


def score_code_accuracy(answer: str, key: Mapping[str, Any]) -> Tuple[float, Dict[str, Any]]:
    """metrics.code_accuracy for one answer: the last fenced block, post-processed, against the
    problem's hidden tests (function-call tests for function problems, stdin otherwise)."""
    blocks = has_code(answer)
    if not blocks:
        return 0.0, {"code_check": "no_code"}
    completion = post_process_code(blocks[-1])
    tests_path = Path(key["lcb_tests_path"])
    tests = tests_path.read_bytes() if tests_path.is_file() else b""
    if hashlib.sha256(tests).hexdigest() != key["lcb_tests_sha256"]:
        raise ValueError("LiveCodeBench tests file is missing or changed: " + str(tests_path))
    third_party = bool(_THIRD_PARTY_IMPORT.search(completion))
    result = run_code_sandbox(completion, tests, not key["lcb_is_stdin"], int(key["lcb_n_tests"]))
    if result["status"] != "ran":
        return 0.0, {"code_check": result["status"], "third_party_import": third_party}
    return result["score"], {"code_check": "ran", "timed_out": result["timed_out"],
                             "third_party_import": third_party}


# --- LiveCodeBench data: scripts/process_datasets/prep_datasets.py --------------------------


class _NoGlobalsUnpickler(pickle.Unpickler):
    """LiveCodeBench pickles a JSON string; refusing every global makes loading inert."""

    def find_class(self, module: str, name: str) -> Any:
        raise pickle.UnpicklingError("refused global %s.%s" % (module, name))


def translate_private_test_cases(encoded: str) -> List[Dict[str, Any]]:
    raw = zlib.decompress(base64.b64decode(encoded))
    return json.loads(_NoGlobalsUnpickler(io.BytesIO(raw)).load())


def _has_test_type(tests: str, test_type: str) -> bool:
    try:
        for test in json.loads(tests):
            if test.get("testtype") == test_type:
                return True
        return False
    except (json.JSONDecodeError, TypeError, AttributeError):
        return False


def _read_parquet(data: bytes, columns: Optional[Sequence[str]] = None) -> Any:
    import pyarrow as pa
    import pyarrow.parquet as pq

    return pq.read_table(pa.BufferReader(data), columns=list(columns) if columns else None)


def _derived_dir(cache_dir: Path) -> Path:
    return cache_dir / "lcb-derived" / LCB_REVISION[:12]


def _load_derived(cache_dir: Path, wanted: Sequence[str]) -> Optional[Dict[str, Dict[str, Any]]]:
    directory = _derived_dir(cache_dir)
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        return None
    data = manifest_path.read_bytes()
    if hashlib.sha256(data).hexdigest() != LCB_MANIFEST_SHA256:
        return None
    problems = json.loads(data.decode("utf-8"))["problems"]
    out = {}
    for global_index in wanted:
        entry = problems.get(global_index)
        if entry is None:
            return None
        path = directory / entry["tests_file"]
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != entry["tests_sha256"]:
            return None
        out[global_index] = dict(entry, tests_path=str(path))
    return out


def _derive_lcb(cache_dir: Path, wanted: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    """Match RouterArena's coding questions to LiveCodeBench rows as prep_datasets.py does and
    write each wanted problem's decoded hidden tests, plus a manifest, under the cache."""
    full = fetch_pinned(_HF.format(repo=DATASET_REPO, revision=DATASET_REVISION, path=FULL_PATH),
                        cache_dir / DATASET_REVISION / FULL_PATH, FULL_SHA256)
    question_to_global_idx: Dict[str, str] = {}
    for row in _read_parquet(full, ["Global Index", "Question"]).to_pylist():
        if "LiveCodeBench" in row["Global Index"]:
            question_to_global_idx[row["Question"]] = row["Global Index"]
    del full
    wanted_set = set(wanted)
    found: Dict[str, List[Dict[str, Any]]] = {}
    position = 0
    for shard, digest in LCB_SHARDS:
        data = fetch_pinned(_HF.format(repo=LCB_REPO, revision=LCB_REVISION, path=shard),
                            cache_dir / "lighteval-code_generation_lite" / LCB_REVISION / shard,
                            digest, max_bytes=LCB_SHARD_MAX_BYTES)
        meta = _read_parquet(data, ["question_content", "question_id", "public_test_cases"]).to_pylist()
        hits = []
        for offset, row in enumerate(meta):
            prompt = row["question_content"]
            global_idx = question_to_global_idx.get(prompt)
            if global_idx is None:
                prompt_start = prompt[:100]
                for question, candidate in question_to_global_idx.items():
                    if question.startswith(prompt_start) or prompt_start in question:
                        global_idx = candidate
                        break
            if global_idx in wanted_set:
                hits.append((offset, global_idx, row))
        if hits:
            private = _read_parquet(data, ["private_test_cases"]).column(0)
            for offset, global_idx, row in hits:
                tests = translate_private_test_cases(private[offset].as_py())
                found.setdefault(global_idx, []).append({
                    "lcb_index": position + offset,
                    "question_id": row["question_id"],
                    "is_stdin": _has_test_type(row["public_test_cases"], "stdin"),
                    "tests": tests,
                })
            del private
        position += len(meta)
        del data, meta
    directory = _derived_dir(cache_dir)
    directory.mkdir(parents=True, exist_ok=True)
    problems = {}
    for global_index in sorted(wanted_set):
        matches = found.get(global_index, [])
        if len(matches) != 1:
            raise ValueError("expected one LiveCodeBench row for %s, found %d" % (global_index, len(matches)))
        match = matches[0]
        blob = json.dumps(match["tests"], ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        tests_file = global_index + ".json"
        (directory / tests_file).write_bytes(blob)
        problems[global_index] = {
            "lcb_index": match["lcb_index"], "question_id": match["question_id"],
            "is_stdin": match["is_stdin"], "n_tests": len(match["tests"]),
            "tests_file": tests_file, "tests_sha256": hashlib.sha256(blob).hexdigest(),
        }
    manifest = {"source": {"repo": LCB_REPO, "revision": LCB_REVISION, "config": "release_v2"},
                "problems": problems}
    data = (json.dumps(manifest, sort_keys=True, indent=1) + "\n").encode("utf-8")
    (directory / "manifest.json").write_bytes(data)
    if hashlib.sha256(data).hexdigest() != LCB_MANIFEST_SHA256:
        raise ValueError("derived LiveCodeBench manifest does not match its pinned digest "
                         + hashlib.sha256(data).hexdigest())
    return {gi: dict(entry, tests_path=str(directory / entry["tests_file"])) for gi, entry in problems.items()}


def lcb_problems(cache_dir: Path, wanted: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    """Per coding question: LiveCodeBench's is_stdin flag and the derived hidden-tests file."""
    cache_dir = Path(cache_dir)
    return _load_derived(cache_dir, wanted) or _derive_lcb(cache_dir, wanted)


# --- The benchmark ----------------------------------------------------------------------------


class RouterArenaSub10:
    name = NAME
    max_output_tokens = MAX_OUTPUT_TOKENS

    def load(self, cache_dir: Path) -> List[Item]:
        cache_dir = Path(cache_dir)
        data = fetch_pinned(_HF.format(repo=DATASET_REPO, revision=DATASET_REVISION, path=SUB10_PATH),
                            cache_dir / DATASET_REVISION / SUB10_PATH, SUB10_SHA256)
        rows = _read_parquet(data).to_pylist()
        if len(rows) != SUB10_SIZE:
            raise ValueError("sub_10 has %d rows, expected %d" % (len(rows), SUB10_SIZE))
        coding = [row["Global Index"] for row in rows if prompt_dataset(row) == "LiveCodeBench"]
        lcb = lcb_problems(cache_dir, coding) if coding else {}
        return [make_item(row, lcb.get(row["Global Index"])) for row in rows]

    def judge_requests(self, item: Item, answer: str) -> List[JudgeRequest]:
        return []  # RouterArena grades every query with rules; there is no LLM judge.

    def grade(self, item: Item, answer: Optional[str], judged: Mapping[str, Optional[str]]) -> Dict[str, Any]:
        key = item.key
        metric = key["metric"]
        grade: Dict[str, Any] = {"dataset": key["eval_dataset"], "metric": metric}
        if answer is None:
            # RouterArena scores a query without a valid generation as 0 and keeps it in the
            # denominator (llm_evaluation/run.py, router_metrics).
            grade.update(score=0.0, correct=False, failed_answer=True)
            return grade
        if metric == "code_accuracy":
            score, extras = score_code_accuracy(answer, key)
        else:
            try:
                score, extras = TEXT_SCORERS[metric](answer, key["answer"])
            except Exception:
                # ModelEvaluator._evaluate_single_entry turns a scorer exception into 0.0.
                score, extras = 0.0, {"scorer_error": True}
        grade.update(score=float(score), failed_answer=False, **extras)
        grade["correct"] = None if metric == "meteor_score" else score >= 1.0
        return grade

    def score(self, grades: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        n = len(grades)
        points = sum(float(g.get("score") or 0.0) for g in grades)
        accuracy = points / n if n else 0.0
        by_metric: Dict[str, List[float]] = {}
        by_dataset: Dict[str, List[float]] = {}
        for g in grades:
            by_metric.setdefault(str(g.get("metric")), []).append(float(g.get("score") or 0.0))
            by_dataset.setdefault(str(g.get("dataset")), []).append(float(g.get("score") or 0.0))

        def table(groups: Mapping[str, List[float]]) -> Dict[str, Dict[str, Any]]:
            return {k: {"n": len(v), "score": 100.0 * sum(v) / len(v)} for k, v in sorted(groups.items())}

        return {
            "score": 100.0 * accuracy,
            "accuracy": accuracy,
            "n": n,
            "points": points,
            "n_failed_answers": sum(1 for g in grades if g.get("failed_answer")),
            "n_failed_judge_calls": 0,
            "n_code_sandbox_errors": sum(1 for g in grades if g.get("code_check") == "sandbox_error"),
            "n_code_ungraded": sum(1 for g in grades if g.get("code_check") == "unavailable"),
            "n_code_failed_third_party_import": sum(
                1 for g in grades
                if g.get("code_check") == "ran" and g.get("third_party_import") and not g.get("score")),
            "n_math_unresolved": sum(1 for g in grades if g.get("math_check") == "unresolved"),
            "by_metric": table(by_metric),
            "by_dataset": table(by_dataset),
        }


def make_item(row: Mapping[str, Any], lcb: Optional[Mapping[str, Any]] = None) -> Item:
    """One dataset row as an Item; ``lcb`` is the coding question's entry from lcb_problems."""
    global_index = row["Global Index"]
    eval_dataset, metric = eval_metric(global_index)
    is_coding = prompt_dataset(row) == "LiveCodeBench"
    if is_coding and lcb is None:
        raise ValueError("LiveCodeBench row not found for " + global_index)
    prompt = format_prompt(row, bool(lcb["is_stdin"]) if is_coding else None)
    messages = (("user", prompt),)
    key: Dict[str, Any] = {"global_index": global_index, "eval_dataset": eval_dataset, "metric": metric,
                           "answer": row["Answer"]}
    if metric == "code_accuracy":
        if lcb is None:
            raise ValueError("no LiveCodeBench tests for " + global_index)
        key.update(lcb_tests_path=str(lcb["tests_path"]), lcb_tests_sha256=lcb["tests_sha256"],
                   lcb_is_stdin=bool(lcb["is_stdin"]), lcb_n_tests=int(lcb["n_tests"]))
    public = {
        "global_index": global_index,
        "dataset": row.get("Dataset name") or "",
        "eval_dataset": eval_dataset,
        "metric": metric,
        "category": row.get("Category") or "",
        "domain": row.get("Domain") or "",
        "difficulty": row.get("Difficulty") or "",
        "messages_sha256": messages_digest(messages),
    }
    return Item(id="ra-" + sha256_text(global_index)[:16], messages=messages, key=key, public=public)


BENCHMARK = RouterArenaSub10()


# --- Cost: llm_evaluation/run.py (compute_row_cost, router_metrics) --------------------------

# model_cost/model_cost.json at HARNESS_COMMIT (prices synced from OpenRouter on 2026-11-16,
# "as charged, discounts included"), USD per million tokens (input, output), for the names a
# Panda submission uses.  A name missing from the table is charged the table's highest prices.
ROUTERARENA_PRICES: Dict[str, Tuple[float, float]] = {
    "deepseek/deepseek-v4.1-flash": (0.21, 0.84),
    "deepseek/deepseek-v4.1-flash@high": (0.21, 0.84),
    "deepseek/deepseek-v4.1-flash@low": (0.21, 0.84),
    "z-ai/glm-5.3": (1.4, 4.4),
    "z-ai/glm-5.3@high": (1.4, 4.4),
    "z-ai/glm-5.3@low": (1.4, 4.4),
    "z-ai/glm-5.3-flash": (0.15, 0.5),
    "glm-5.3-flash": (0.15, 0.5),
    "z-ai/glm-5.3-flash@default": (0.15, 0.5),
}
ROUTERARENA_MAX_PRICE = (15.0, 75.0)  # anthropic/claude-opus-4.1, the table's top on both sides
MODEL_COST_JSON_SHA256 = "fd9d27b76d74fcf2a7422a9e8cd1b08c2cdfd6b1cc0968887fa9262a7aedc21d"
# Not in RouterArena's table: a submission must add a profile (with a link to the published
# price) to model_cost/model_profiles.yaml, or each embedding call is charged $15/$75 per 1M.
PROPOSED_PROFILES: Dict[str, Tuple[float, float]] = {
    "google/embeddinggemma-2": (0.0017, 0.0),
}
# Panda's list prices (USD per million tokens), for a view at Panda's own prices.
PANDA_LIST_PRICES: Dict[str, Tuple[float, float]] = {
    "deepseek/deepseek-v4.1-flash": (0.30, 1.20),
    "z-ai/glm-5.3": (1.40, 4.40),
    "z-ai/glm-5.3-flash": (0.15, 0.50),
    "google/embeddinggemma-2": (0.0017, 0.0),
}

# Panda and Workers AI names -> the names a RouterArena submission uses.
ROUTERARENA_NAMES: Dict[str, str] = {
    "deepseek-v4.1-flash": "deepseek/deepseek-v4.1-flash",
    "@cf/deepseek-ai/deepseek-v4.1-flash-1017": "deepseek/deepseek-v4.1-flash",
    "deepseek/deepseek-v4.1-flash": "deepseek/deepseek-v4.1-flash",
    "glm-5.3": "z-ai/glm-5.3",
    "@cf/zai-org/glm-5.3": "z-ai/glm-5.3",
    "z-ai/glm-5.3": "z-ai/glm-5.3",
    "glm-5.3-flash": "z-ai/glm-5.3-flash",
    "@cf/zai-org/glm-5.3-flash": "z-ai/glm-5.3-flash",
    "z-ai/glm-5.3-flash": "z-ai/glm-5.3-flash",
    "embeddinggemma-2": "google/embeddinggemma-2",
    "@cf/google/embeddinggemma-2": "google/embeddinggemma-2",
    "google/embeddinggemma-2": "google/embeddinggemma-2",
}
ANSWER_MODELS = ("deepseek/deepseek-v4.1-flash", "z-ai/glm-5.3", "z-ai/glm-5.3-flash")


def routerarena_name(model: str) -> str:
    return ROUTERARENA_NAMES.get(model, model)


def validate_router_usage(value: Any) -> List[str]:
    """global_utils/router_usage.validate_router_usage, verbatim in effect."""
    if not isinstance(value, list):
        return ["router_usage must be a list of calls"]
    errors = []
    for i, call in enumerate(value):
        where = "router_usage[%d]" % i
        if not isinstance(call, dict):
            errors.append(where + " must be an object")
            continue
        if not isinstance(call.get("model"), str) or not call["model"].strip():
            errors.append(where + ".model must be a non-empty string")
        for field in ("input_tokens", "output_tokens"):
            count = call.get(field)
            if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                errors.append("%s.%s must be an integer >= 0" % (where, field))
        if "purpose" in call and not isinstance(call["purpose"], str):
            errors.append(where + ".purpose must be a string")
    return errors


def _price_table(prices: Optional[Mapping[str, Tuple[float, float]]]) -> Mapping[str, Tuple[float, float]]:
    if prices is not None:
        return prices
    table = dict(ROUTERARENA_PRICES)
    table.update(PROPOSED_PROFILES)
    return table


def _call_cost(model: str, input_tokens: int, output_tokens: int, total_tokens: int,
               table: Mapping[str, Tuple[float, float]], unpriced: Counter) -> float:
    """ModelEvaluator.calculate_inference_cost: reasoning tokens reported only in total_tokens
    are billed at the output price."""
    name = routerarena_name(model)
    lookup = name[:-6] if name.endswith("_batch") else name
    price = table.get(lookup)
    if price is None:
        price = ROUTERARENA_MAX_PRICE
        unpriced[lookup] += 1
    reasoning = max(0, total_tokens - input_tokens - output_tokens)
    return (input_tokens * price[0] + output_tokens * price[1] + reasoning * price[1]) / 1_000_000


def _cost_row(row: Any) -> Tuple[str, int, int, int, Any, bool]:
    if isinstance(row, Mapping):
        model = row["model"]
        input_tokens = int(row.get("input_tokens") or 0)
        output_tokens = int(row.get("output_tokens") or 0)
        total = row.get("total_tokens")
        total_tokens = int(total) if total is not None else input_tokens + output_tokens
        answered = row.get("success", True) is True and (
            "answer" not in row or (isinstance(row["answer"], str) and row["answer"].strip() != ""))
        return model, input_tokens, output_tokens, total_tokens, row.get("router_usage"), answered
    model, input_tokens, output_tokens, router_usage = row
    input_tokens, output_tokens = int(input_tokens or 0), int(output_tokens or 0)
    return model, input_tokens, output_tokens, input_tokens + output_tokens, router_usage, True


def cost_summary(rows: Sequence[Any], prices: Optional[Mapping[str, Tuple[float, float]]] = None) -> Dict[str, Any]:
    """RouterArena's cost over one split's rows, with its parts.

    A row is ``(model, input_tokens, output_tokens, router_usage)`` or a mapping with those
    keys (optionally ``total_tokens``, ``success`` and ``answer``).  As in RouterArena, a row
    counts in the denominator always and in the numerator only when its generation is valid
    (answered, with output_tokens > 0); its cost is the answer at the model's price plus every
    ``router_usage`` call at that call's model's price.  ``prices`` replaces the table
    (default: RouterArena's pinned prices plus ``PROPOSED_PROFILES``)."""
    table = _price_table(prices)
    unpriced: Counter = Counter()
    answer_total = router_total = 0.0
    costed = 0
    for row in rows:
        model, input_tokens, output_tokens, total_tokens, router_usage, answered = _cost_row(row)
        if router_usage is not None:
            problems = validate_router_usage(router_usage)
            if problems:
                raise ValueError("; ".join(problems))
        if not answered or output_tokens <= 0:
            continue
        answer_cost = _call_cost(model, input_tokens, output_tokens, total_tokens, table, unpriced)
        router_cost = sum(
            _call_cost(c["model"], c["input_tokens"], c["output_tokens"], c["input_tokens"] + c["output_tokens"],
                       table, unpriced)
            for c in (router_usage or [])
        )
        if answer_cost + router_cost > 0:
            answer_total += answer_cost
            router_total += router_cost
            costed += 1
    n = len(rows)
    total = answer_total + router_total
    return {
        "cost_per_1k": 1000.0 * total / n if n else 0.0,
        "total_cost": total,
        "answer_cost": answer_total,
        "router_cost": router_total,
        "n_queries": n,
        "n_costed": costed,
        "unpriced_models": dict(unpriced),
    }


def cost_per_1k(rows: Sequence[Any], prices: Optional[Mapping[str, Tuple[float, float]]] = None) -> float:
    """Dollars per 1,000 queries, computed the way RouterArena's evaluator does."""
    return cost_summary(rows, prices)["cost_per_1k"]


def arena_score(cost_per_1k: float, accuracy: float, beta: float = 0.1, c_max: float = 200.0,
                c_min: float = 0.0044) -> float:
    """llm_evaluation/run.py compute_arena_score, on the board's 0-100 scale.

    ``cost_per_1k`` in dollars per 1,000 queries; ``accuracy`` as a fraction (0-1)."""
    if cost_per_1k is None or cost_per_1k <= 0:
        raise ValueError("cost must be positive")
    if accuracy is None or accuracy < 0 or accuracy > 1:
        raise ValueError("accuracy must be between 0 and 1")
    cost = max(c_min, min(cost_per_1k, c_max))
    normalized = (math.log2(c_max) - math.log2(cost)) / (math.log2(c_max) - math.log2(c_min))
    return 100.0 * ((1 + beta) * accuracy * normalized) / (beta * accuracy + normalized)


# --- Submission files: router_inference/generate_prediction_file.py + llm_inference/run.py ---

DEFAULT_ROUTER_NAME = "panda-auto"
DEFAULT_DESCRIPTION = (
    "Panda's auto router (Panda API model `auto`, balanced mode: route table v2 and task profile v2 "
    "of 2026-11-17). An EmbeddingGemma 2 embedding of the first 2,000 characters of the latest user "
    "message (one call per query, recorded in router_usage) is matched to the nearest of eleven "
    "task-type centroids built from MT-Bench's first turns and Panda's own written examples. Code, "
    "math, STEM and humanities go to DeepSeek V4.1 Flash, reasoning to GLM-5.3, and writing and the "
    "other text types to GLM-5.3 Flash; one answer call per query at Panda's model settings. No "
    "RouterArena data, template, config or answer was used to build the router. The route table was "
    "chosen on Panda's own tests, which included public MMLU-Pro, AIME, MATH and LiveCodeBench "
    "problems, datasets RouterArena also draws on."
)


def _row_identity(row: Mapping[str, Any]) -> Tuple[str, str]:
    item = row.get("item")
    if item is not None:
        return item.key["global_index"], item.messages[-1][1]
    return row["global_index"], row["prompt"]


def write_submission(out_dir: Path, rows: Sequence[Mapping[str, Any]], router_name: str = DEFAULT_ROUTER_NAME,
                     description: str = DEFAULT_DESCRIPTION,
                     expected_rows: Optional[int] = SUB10_SIZE) -> Dict[str, Any]:
    """Write RouterArena's router config and ``sub_10`` prediction file under ``out_dir``.

    Each row is a mapping with ``item`` (from ``load``; or ``global_index`` and ``prompt``),
    ``model`` (the routed model, Panda or RouterArena name), ``answer`` (text, or None when
    the call failed), ``input_tokens``, ``output_tokens`` (reasoning included), optionally
    ``total_tokens``, ``router_usage`` (calls the router made: model, input_tokens,
    output_tokens, purpose), ``provider`` and ``error``.  An empty answer or one without
    output tokens is written as a failed generation, which RouterArena scores 0.  Files go to
    ``out_dir/router_inference/config/<router_name>.json`` and
    ``out_dir/router_inference/predictions/<router_name>.json``, the paths a pull request
    adds.  Nothing is sent anywhere."""
    if not re.fullmatch(r"[A-Za-z0-9._-]+", router_name):
        raise ValueError("router name may only use letters, digits, '.', '_' and '-'")
    if expected_rows is not None and len(rows) != expected_rows:
        raise ValueError("expected %d rows, got %d" % (expected_rows, len(rows)))
    seen = set()
    predictions = []
    used: Counter = Counter()
    failed = 0
    for row in rows:
        global_index, prompt = _row_identity(row)
        if global_index in seen:
            raise ValueError("duplicate global index " + global_index)
        seen.add(global_index)
        model = routerarena_name(row["model"])
        if model not in ANSWER_MODELS:
            raise ValueError("unknown answer model %r (known: %s)" % (row["model"], ", ".join(ANSWER_MODELS)))
        answer = row.get("answer")
        if answer is not None and not isinstance(answer, str):
            raise ValueError("answer must be text or None for " + global_index)
        input_tokens = int(row.get("input_tokens") or 0)
        output_tokens = int(row.get("output_tokens") or 0)
        total = row.get("total_tokens")
        total_tokens = int(total) if total is not None else input_tokens + output_tokens
        ok = answer is not None and answer.strip() != "" and output_tokens > 0
        if not ok:
            failed += 1
        error = None
        if not ok:
            error = row.get("error") or ("no answer" if answer is None else
                                         "empty answer" if not answer.strip() else "no output tokens")
        entry: Dict[str, Any] = {
            "global index": global_index,
            "prompt": prompt,
            "prediction": model,
            "generated_result": {
                "generated_answer": answer if answer is not None else "",
                "success": ok,
                "token_usage": {"input_tokens": input_tokens, "output_tokens": output_tokens,
                                "total_tokens": total_tokens},
                "provider": row.get("provider") or "panda",
                "error": error,
            },
            "cost": None,
            "accuracy": None,
            "for_optimality": False,
        }
        router_usage = row.get("router_usage")
        if router_usage is not None:
            problems = validate_router_usage(router_usage)
            if problems:
                raise ValueError("%s: %s" % (global_index, "; ".join(problems)))
            entry["router_usage"] = [dict(call, model=routerarena_name(call["model"])) for call in router_usage]
        used[model] += 1
        predictions.append(entry)
    config = {"pipeline_params": {"router_name": router_name, "router_cls_name": "external_prediction_file",
                                  "models": list(ANSWER_MODELS), "description": description}}
    out_dir = Path(out_dir)
    config_path = out_dir / "router_inference" / "config" / (router_name + ".json")
    predictions_path = out_dir / "router_inference" / "predictions" / (router_name + ".json")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with open(predictions_path, "w", encoding="utf-8") as handle:
        json.dump(predictions, handle, ensure_ascii=False, indent=2)
    return {
        "config_path": str(config_path),
        "predictions_path": str(predictions_path),
        "n_rows": len(predictions),
        "n_failed": failed,
        "models": dict(used),
        "pr_notes": [
            "add 'z-ai/glm-5.3' to universal_names in universal_model_names.py (only its @low/@high "
            "spellings are listed; the validator rejects the config otherwise)",
            "add a price profile for google/embeddinggemma-2 to model_cost/model_profiles.yaml with a "
            "link to the published price and regenerate model_cost/model_cost.json, or every router "
            "call is charged the table's maximum ($15/$75 per 1M tokens)",
            "add router_inference/predictions/%s-robustness.json (420 routing picks on the perturbed "
            "prompts); the PR workflow requires it" % router_name,
            "the leaderboard ranks the full split (8,400 rows); a 809-row file is graded as sub_10",
        ],
    }
