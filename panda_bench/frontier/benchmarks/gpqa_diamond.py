"""GPQA Diamond, prompted and scored the way Epoch AI's benchmarking hub runs it (version 1.0.6).

Epoch runs GPQA Diamond in Inspect: one user message from the multiple-choice
chain-of-thought template, an empty system prompt, the four options shuffled, and Inspect's
``choice()`` scorer.  A reply scores only when its ``ANSWER: <letter>`` line names exactly one
of A-D.  There is no judge model.  See docs/frontier/gpqa_diamond.md for the sources, the
deviations from Epoch's protocol and the published rows.

The dataset's authors ask that its examples not be revealed online, so ``public`` and the
grades carry no question or option text: only ids, categories, a digest and letters.
"""
from __future__ import annotations

import csv
import hashlib
import io
import math
import re
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from ..core import Item, JudgeRequest, fetch_pinned, messages_digest, sha256_text


# github.com/idavidrein/gpqa at its current head (2024-09-30).  dataset.zip last changed in
# d46dc8d5 (2023-11-20); the CSV inside is byte-identical to simple-evals' public copy.
DATA_REVISION = "56686c06f5e19865c153de0fdb11be3890014df7"
DATA_FILE = "dataset.zip"
DATA_SHA256 = "461ae7329f15a3e35f8184d2dac24b990f34fdf12f366ca4062d8e6638cd08dc"
DATA_URL = f"https://raw.githubusercontent.com/idavidrein/gpqa/{DATA_REVISION}/{DATA_FILE}"
# The zip is encrypted only to keep the questions away from crawlers; the repository's
# README publishes the password.
ZIP_PASSWORD = b"deserted-untie-orchid"
CSV_MEMBER = "dataset/gpqa_diamond.csv"
CSV_SHA256 = "41d1213cd7a4998605a26c2798500652572007161b3a92817ba46b35befcd305"

LETTERS = ("A", "B", "C", "D")
OPTION_COLUMNS = ("Correct Answer", "Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3")

# The template and the answer parser below come from inspect_ai, commit
# 7a168f5ff95496bcd8bb237247f56cbf9d366437 (2026-11-08), src/inspect_ai/solver/_multiple_choice.py:
# SINGLE_ANSWER_TEMPLATE_COT and parse_answers().  Epoch's published task (gist
# tadamcz/a61515465e34a3c66f3a78673502bc3f) uses the same template text.  Epoch's changelog for
# 2026-02-20 (benchmark version 1.0.6) changed the answer format from 'ANSWER: $LETTER' to
# 'ANSWER: LETTER'; that is the only edit made here.
#
# MIT License
#
# Copyright (c) 2024 UK AI Security Institute
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
PROMPT_TEMPLATE = r"""
Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: LETTER' (without quotes) where LETTER is one of {letters}. Think step by step before answering.

{question}

{choices}
""".strip()

# parse_answers() first unwraps $X$, **X** and (X), then takes the last line-anchored
# ANSWER match, then (only if there is none) the last unanchored one.
_DOLLAR_WRAPPER = re.compile(r"\x24\s*([A-Za-z\d][A-Za-z\d ,]*?)\s*\x24")
_BOLD_WRAPPER = re.compile(r"\*\*\s*([A-Za-z\d][A-Za-z\d ,]*?)\s*\*\*")
_PAREN_WRAPPER = re.compile(r"(?i)(^|[:,])(\s*)\(([A-Za-z\d])\)(?=\s*(?:,|\.|\n|\Z))")
STRICT_PATTERN = re.compile(r"(?i)^ANSWER\s*:\s*([A-Za-z\d ,]+)\s*(?:$|\n|\.)", re.MULTILINE)
FALLBACK_PATTERN = re.compile(r"(?i)ANSWER\s*:\s*([A-Za-z\d ,]+)(?:[^\w]|\n|$|\.)")


def extract_letter(completion: str) -> Optional[str]:
    """The letter parse_answers() selects for a single-answer question, or None."""
    text = _DOLLAR_WRAPPER.sub(r"\1", completion)
    text = _BOLD_WRAPPER.sub(r"\1", text)
    text = _PAREN_WRAPPER.sub(r"\1\2\3", text)
    matches = STRICT_PATTERN.findall(text) or FALLBACK_PATTERN.findall(text)
    if not matches:
        return None
    matched = matches[-1].strip().rstrip(".").upper()
    # Exactly one token, and it must be an offered letter: "B or C", "B, C" and "BC" fail.
    tokens = [token for token in matched.split(",") if token]
    if len(tokens) == 1 and tokens[0] in LETTERS:
        return tokens[0]
    return None


def item_id(record_id: str) -> str:
    return "gpqa-" + sha256_text(record_id)[:16]


def choice_order(public_id: str) -> List[int]:
    """Which option is shown as A, B, C and D: indexes into OPTION_COLUMNS (0 is the correct
    answer), sorted by SHA-256 of "<item id>:<index>".  Fixed for every run, setup and Python."""
    return sorted(range(len(OPTION_COLUMNS)), key=lambda index: sha256_text(f"{public_id}:{index}"))


def _rows(cache_dir: Path) -> List[Dict[str, str]]:
    archive = fetch_pinned(DATA_URL, cache_dir / DATA_FILE, DATA_SHA256)
    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        raw = bundle.read(CSV_MEMBER, pwd=ZIP_PASSWORD)
    if hashlib.sha256(raw).hexdigest() != CSV_SHA256:
        raise ValueError("GPQA Diamond CSV does not match its pinned digest")
    return list(csv.DictReader(io.StringIO(raw.decode("utf-8"), newline="")))


class GPQADiamond:
    name = "gpqa-diamond"
    max_output_tokens = 16384

    def load(self, cache_dir: Path) -> List[Item]:
        items = []
        for row in _rows(cache_dir):
            public_id = item_id(row["Record ID"])
            order = choice_order(public_id)
            choices = "\n".join(f"{letter}) {row[OPTION_COLUMNS[index]]}" for letter, index in zip(LETTERS, order))
            prompt = PROMPT_TEMPLATE.format(letters=",".join(LETTERS), question=row["Question"], choices=choices)
            messages = (("user", prompt),)
            items.append(Item(
                id=public_id,
                messages=messages,
                key={"correct_letter": LETTERS[order.index(0)]},
                public={
                    "source_id": row["Record ID"],
                    "category": row["High-level domain"],
                    "subcategory": row["Subdomain"],
                    "messages_sha256": messages_digest(messages),
                },
            ))
        return items

    def judge_requests(self, item: Item, answer: str) -> List[JudgeRequest]:
        return []

    def grade(self, item: Item, answer: Optional[str], judged: Mapping[str, Optional[str]]) -> Dict[str, Any]:
        correct_letter = item.key["correct_letter"]
        # Inspect scores an empty completion as no answer (0); a failed request scores 0 too.
        failed_answer = answer is None or not answer.strip()
        letter = None if failed_answer else extract_letter(answer)
        return {
            "letter": letter,
            "correct_letter": correct_letter,
            "correct": letter == correct_letter,
            "failed_answer": failed_answer,
            "extraction_failed": not failed_answer and letter is None,
            "failed_judge_calls": 0,
        }

    def score(self, grades: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        n = len(grades)
        correct = sum(1 for grade in grades if grade["correct"])
        accuracy = correct / n if n else 0.0
        # Inspect's stderr(): sample standard deviation (ddof=1) over sqrt(n), which for 0/1
        # scores is sqrt(p(1-p)/(n-1)).  Epoch averages 16 samples per question first.
        stderr = math.sqrt(accuracy * (1.0 - accuracy) / (n - 1)) if n > 1 else 0.0
        return {
            "score": 100.0 * accuracy,
            "stderr": 100.0 * stderr,
            "n": n,
            "correct": correct,
            "extraction_failures": sum(1 for grade in grades if grade["extraction_failed"]),
            "failed_answers": sum(1 for grade in grades if grade["failed_answer"]),
            "failed_judge_calls": 0,
        }


BENCHMARK = GPQADiamond()
