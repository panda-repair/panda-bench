"""Offline GPQA Diamond checks: synthetic rows in an in-memory encrypted zip, no network."""
from __future__ import annotations

import csv
import hashlib
import io
import itertools
import json
import math
import statistics
import struct
import unittest
import zipfile
import zlib
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from panda_bench.frontier.benchmarks import gpqa_diamond as gpqa
from panda_bench.frontier.core import Item, messages_digest

# SINGLE_ANSWER_TEMPLATE_COT from inspect_ai commit 7a168f5ff95496bcd8bb237247f56cbf9d366437,
# src/inspect_ai/solver/_multiple_choice.py (MIT License, Copyright (c) 2024 UK AI Security
# Institute; the full notice is in the module under test).
INSPECT_TEMPLATE = (
    "Answer the following multiple choice question. The last line of your response should be of the "
    "following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of {letters}. Think step "
    "by step before answering.\n\n{question}\n\n{choices}"
)
COLUMNS = ["Record ID", "High-level domain", "Subdomain", "Question", "Correct Answer",
           "Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3", "Canary String"]
SECRET_MARKERS = ("Synthetic question", "RIGHT-OPTION", "WRONG-OPTION", "Second line")


def row(record_id, domain="Physics"):
    return {
        "Record ID": record_id,
        "High-level domain": domain,
        "Subdomain": "Synthetic subdomain",
        "Question": f"Synthetic question {record_id} with {{braces}}?\nSecond line. ",
        "Correct Answer": f"RIGHT-OPTION {record_id}",
        "Incorrect Answer 1": f"WRONG-OPTION one {record_id}",
        "Incorrect Answer 2": f"WRONG-OPTION two {record_id}",
        "Incorrect Answer 3": f"WRONG-OPTION three {record_id} ",
        "Canary String": "synthetic canary",
    }


def csv_bytes(rows):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _crc_step(crc, byte):
    return zlib.crc32(bytes([byte]), crc ^ 0xFFFFFFFF) ^ 0xFFFFFFFF


def encrypted_zip(member, data, password):
    """One stored member under traditional PKWARE (ZipCrypto) encryption, like dataset.zip."""
    keys = [0x12345678, 0x23456789, 0x34567890]

    def update(byte):
        keys[0] = _crc_step(keys[0], byte)
        keys[1] = ((keys[1] + (keys[0] & 0xFF)) * 134775813 + 1) & 0xFFFFFFFF
        keys[2] = _crc_step(keys[2], keys[1] >> 24)

    for byte in password:
        update(byte)
    crc = zlib.crc32(data) & 0xFFFFFFFF
    cipher = bytearray()
    for byte in bytes(range(11)) + bytes([crc >> 24]) + data:  # 12-byte header, check byte last
        stream = keys[2] | 2
        cipher.append(byte ^ (((stream * (stream ^ 1)) >> 8) & 0xFF))
        update(byte)
    name = member.encode("utf-8")
    date = (2023 - 1980) << 9 | 11 << 5 | 20
    local = struct.pack("<IHHHHHIIIHH", 0x04034B50, 20, 1, 0, 0, date, crc, len(cipher), len(data),
                        len(name), 0) + name + bytes(cipher)
    central = struct.pack("<IHHHHHHIIIHHHHHII", 0x02014B50, 20, 20, 1, 0, 0, date, crc, len(cipher),
                          len(data), len(name), 0, 0, 0, 0, 0, 0) + name
    return local + central + struct.pack("<IHHHHIIH", 0x06054B50, 0, 0, 1, 1, len(central), len(local), 0)


class GPQADiamondTests(unittest.TestCase):
    def setUp(self):
        self.bench = gpqa.BENCHMARK
        self.cache = Path("unused-in-memory-cache")

    def load(self, rows):
        data = csv_bytes(rows)
        archive = encrypted_zip(gpqa.CSV_MEMBER, data, gpqa.ZIP_PASSWORD)
        with patch.object(gpqa, "fetch_pinned", return_value=archive) as fetch, \
                patch.object(gpqa, "CSV_SHA256", hashlib.sha256(data).hexdigest()):
            items = self.bench.load(self.cache)
        fetch.assert_called_once_with(gpqa.DATA_URL, self.cache / "dataset.zip", gpqa.DATA_SHA256)
        return items

    def item(self, correct_letter="C"):
        item, = self.load([row("recSynthetic")])
        return Item(id=item.id, messages=item.messages, key={"correct_letter": correct_letter}, public=item.public)

    def test_pins_settings_and_no_judge(self):
        self.assertEqual((self.bench.name, self.bench.max_output_tokens), ("gpqa-diamond", 16384))
        self.assertEqual(gpqa.DATA_URL, "https://raw.githubusercontent.com/idavidrein/gpqa/"
                                        "56686c06f5e19865c153de0fdb11be3890014df7/dataset.zip")
        self.assertEqual(gpqa.DATA_SHA256, "461ae7329f15a3e35f8184d2dac24b990f34fdf12f366ca4062d8e6638cd08dc")
        self.assertEqual(gpqa.CSV_SHA256, "41d1213cd7a4998605a26c2798500652572007161b3a92817ba46b35befcd305")
        self.assertEqual((gpqa.CSV_MEMBER, gpqa.ZIP_PASSWORD), ("dataset/gpqa_diamond.csv", b"deserted-untie-orchid"))
        self.assertEqual(self.bench.judge_requests(self.item(), "ANSWER: C"), [])

    def test_template_is_inspects_with_only_epochs_v1_0_6_edit(self):
        self.assertEqual(gpqa.PROMPT_TEMPLATE, INSPECT_TEMPLATE.replace("'ANSWER: $LETTER'", "'ANSWER: LETTER'"))
        self.assertNotIn("$", gpqa.PROMPT_TEMPLATE)
        self.assertEqual(hashlib.sha256(gpqa.PROMPT_TEMPLATE.encode()).hexdigest(),
                         "252d1efff06346836376902f3125e8f98f95b148b99161c285b9230840c1cf66")

    def test_load_decrypts_in_memory_and_renders_the_exact_prompt(self):
        rows = [row("recSecond", "Chemistry"), row("recFirst"), row("recThird", "Biology")]
        items = self.load(rows)
        expected_ids = ["gpqa-" + hashlib.sha256(r["Record ID"].encode()).hexdigest()[:16] for r in rows]
        self.assertEqual([item.id for item in items], expected_ids)
        for source, item in zip(rows, items):
            order = sorted(range(4), key=lambda k: hashlib.sha256(f"{item.id}:{k}".encode()).hexdigest())
            options = [source[c] for c in COLUMNS[4:8]]
            prompt = (
                "Answer the following multiple choice question. The last line of your response should be of "
                "the following format: 'ANSWER: LETTER' (without quotes) where LETTER is one of A,B,C,D. "
                "Think step by step before answering.\n\n" + source["Question"] + "\n\n"
                + "\n".join(letter + ") " + options[k] for letter, k in zip("ABCD", order))
            )
            self.assertEqual(item.messages, (("user", prompt),))
            self.assertEqual(item.key, {"correct_letter": "ABCD"[order.index(0)]})
            self.assertEqual(item.public, {
                "source_id": source["Record ID"],
                "category": source["High-level domain"],
                "subcategory": "Synthetic subdomain",
                "messages_sha256": messages_digest(item.messages),
            })

    def test_fixture_is_encrypted_and_a_wrong_csv_digest_is_refused(self):
        archive = encrypted_zip(gpqa.CSV_MEMBER, csv_bytes([row("recX")]), gpqa.ZIP_PASSWORD)
        with zipfile.ZipFile(io.BytesIO(archive)) as bundle, self.assertRaises(RuntimeError):
            bundle.read(gpqa.CSV_MEMBER)
        with patch.object(gpqa, "fetch_pinned", return_value=archive), self.assertRaises(ValueError):
            self.bench.load(self.cache)

    def test_choice_order_is_a_fixed_uniform_permutation_of_the_four_options(self):
        self.assertEqual(gpqa.choice_order("gpqa-0123456789abcdef"), gpqa.choice_order("gpqa-0123456789abcdef"))
        orders = [tuple(gpqa.choice_order("gpqa-%016x" % i)) for i in range(2400)]
        self.assertEqual(set(orders), set(itertools.permutations(range(4))))
        positions = Counter(order.index(0) for order in orders)
        self.assertTrue(all(520 <= positions[p] <= 680 for p in range(4)), positions)

    def test_extract_letter_follows_inspects_parser(self):
        cases = {
            "Reasoning.\nANSWER: C": "C",
            "ANSWER: b": "B",
            "Answer: D.": "D",
            "answer:C": "C",
            "ANSWER: A\nOn reflection...\nANSWER: C": "C",  # the last answer line wins
            "The final ANSWER: D": "D",  # unanchored fallback
            "ANSWER: $B$": "B",
            "ANSWER: **C**": "C",
            "**ANSWER: B**": "B",
            "ANSWER: (C)": "C",
            "ANSWER: A,": "A",
            "ANSWER:\nB": "B",
            "ANSWER: B or C": None,  # hedges and several letters score wrong
            "ANSWER: B, C": None,
            "ANSWER: BC": None,
            "ANSWER: B and C": None,
            "ANSWER: A ,": None,
            "**ANSWER:** B": None,
            "ANSWER: E": None,
            "ANSWER: B because it fits.": None,
            "\\boxed{B}": None,
            "So the answer is (D).": None,
            "No answer line at all": None,
        }
        for completion, letter in cases.items():
            with self.subTest(completion=completion):
                self.assertEqual(gpqa.extract_letter(completion), letter)

    def test_grade_paths(self):
        item = self.item("C")
        junk = {"unused": "judges are never called"}
        cases = [
            ("Some steps.\nANSWER: C", {"letter": "C", "correct": True, "failed_answer": False, "extraction_failed": False}),
            ("ANSWER: B", {"letter": "B", "correct": False, "failed_answer": False, "extraction_failed": False}),
            ("ANSWER: B or C", {"letter": None, "correct": False, "failed_answer": False, "extraction_failed": True}),
            ("It is C.", {"letter": None, "correct": False, "failed_answer": False, "extraction_failed": True}),
            ("  \n", {"letter": None, "correct": False, "failed_answer": True, "extraction_failed": False}),
            (None, {"letter": None, "correct": False, "failed_answer": True, "extraction_failed": False}),
        ]
        for answer, expected in cases:
            with self.subTest(answer=answer):
                grade = self.bench.grade(item, answer, junk)
                self.assertEqual(grade, {**expected, "correct_letter": "C", "failed_judge_calls": 0})
                self.assertEqual(json.loads(json.dumps(grade)), grade)

    def test_score_arithmetic_by_hand(self):
        item = self.item("A")
        answers = ["ANSWER: A", "ANSWER: A", "x\nANSWER: a", "ANSWER: A or B", None]
        result = self.bench.score([self.bench.grade(item, a, {}) for a in answers])
        # 3 of 5 correct: 60.0.  Inspect's stderr = stdev([1,1,1,0,0]) / sqrt(5) = sqrt(0.6 * 0.4 / 4).
        self.assertEqual(result["score"], 60.0)
        self.assertAlmostEqual(result["stderr"], 24.494897427831781)
        self.assertAlmostEqual(result["stderr"], 100 * statistics.stdev([1, 1, 1, 0, 0]) / math.sqrt(5))
        self.assertEqual({k: result[k] for k in ("n", "correct", "extraction_failures", "failed_answers",
                                                 "failed_judge_calls")},
                         {"n": 5, "correct": 3, "extraction_failures": 1, "failed_answers": 1, "failed_judge_calls": 0})
        self.assertEqual(self.bench.score([])["score"], 0.0)
        self.assertEqual(self.bench.score([self.bench.grade(item, "ANSWER: A", {})])["stderr"], 0.0)

    def test_public_fields_and_grades_carry_no_question_or_option_text(self):
        items = self.load([row("recA"), row("recB", "Biology")])
        for item in items:
            published = [json.dumps(item.public)]
            for answer in (None, "", "ANSWER: A", "Synthetic question echoed. RIGHT-OPTION\nANSWER: B or C"):
                published.append(json.dumps(self.bench.grade(item, answer, {})))
            published.append(json.dumps(self.bench.score([self.bench.grade(item, "ANSWER: D", {})])))
            for text in published:
                for marker in SECRET_MARKERS:
                    self.assertNotIn(marker, text)


if __name__ == "__main__":
    unittest.main()
