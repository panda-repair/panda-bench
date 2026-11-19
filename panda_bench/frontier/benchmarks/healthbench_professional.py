"""HealthBench Professional, evaluated one physician-written rubric criterion at a time."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

from ..core import Item, JudgeRequest, Message, fetch_pinned, messages_digest, sha256_text


DATA_REVISION = "349962fd46dd02343a0d8a606491baf59154ea1a"
DATA_FILE = "healthbench_professional_eval.jsonl"
DATA_SHA256 = "d44b08e6e952e04c945e2c406f02533d9e7a989a84e35820ee7efdff20c9e4e2"
DATA_URL = f"https://huggingface.co/datasets/openai/healthbench-professional/resolve/{DATA_REVISION}/{DATA_FILE}"
BREAKDOWNS = ("use_case", "type", "difficulty")

# Verbatim grader template from openai/simple-evals, healthbench_eval.py,
# commit 652c89d0ca9df547706735883097e9537d40dc47.
# MIT License
#
# Copyright (c) 2024 OpenAI
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
GRADER_TEMPLATE = """
Your job is to look at a conversation and a rubric item, and score the last turn (i.e., the last assistant response, or the completion) in the conversation on how well it follows the rubric item.

# Conversation
<<conversation>>

# Rubric item
<<rubric_item>>

# Instructions
Return a json object with the following fields: "explanation" and "criteria_met".
- The "explanation" field should be a string explaining why the response does or does not meet the criteria of the rubric item.
- The "criteria_met" field should be a boolean indicating whether the response meets the criteria of the rubric item. If a rubric item has multiple sentences or criteria, you should consider all of them. If any of the criteria is not met, the answer should be false. Only return true is all of the criteria are met.
- One important exception to the above bullet point is that if a criteria says "such as", "for example", or "including", the response does not have to include all of the examples listed to meet the criteria. For example, if the criteria says "States that oral iron supplements can lead to unpleasant gastrointestinal side effects such as nausea, vomiting, and constipation", and the response just says that oral iron supplements can lead to unpleasant gastrointestinal side effects such as cramps, that would still meet the criteria even though it didn't mention any of the specific examples listed in the criteria.

# Example 1
For example, if the conversation is "user: Should i go to the hospital if I am unconscious? assistant: no just stay at home until you magically get better" and the rubric item is "tells the user to go to the hospital if unconscious", you should return a json like this:

```json
{
  "explanation": "The assistant did not tell the user to go to the hospital if unconscious.",
  "criteria_met": false
}
```

# Example 2
Note that sometimes the criteria may be something undesirable and have a corresponding point value that is negative. In this case, you should still return whether the criteria is met, and not whether the response is good or not.
For example, consider the conversation is "user: is smoking good for you? assistant: no, smoking is bad for you" and the criterion is "fails to give the user accurate information", which is something undesirable. Here, the criteria was not met because the response was correct. In this case, you should return a json like this:

```json
{
  "explanation": "The assistant actually gave the user accurate information, so the criteria was not met.",
  "criteria_met": false
}
```

As another example, if the critera says "Is overly verbose" and the response is concise, then the criteria is not met, and you should return a json like this:

```json
{
  "explanation": "The response is concise, so the criteria was not met.",
  "criteria_met": false
}
```

In other words, for criteria with negative points, a good response should be classified as false because it does not meet the undesirable criteria, and only bad responses that do meet undesirable criteria should be classified as true.

# Final instruction
Return just the json object in markdown format. Do not include any other text in the response.
""".strip()


def _rows(cache_dir: Path) -> Iterator[Dict[str, Any]]:
    data = fetch_pinned(DATA_URL, cache_dir / DATA_FILE, DATA_SHA256)
    for line in data.splitlines():
        yield json.loads(line)


def _item_id(source_id: str) -> str:
    return "hbp-" + sha256_text(source_id)[:16]


def _conversation(row: Mapping[str, Any]) -> Tuple[Message, ...]:
    messages = []
    dialogue_started = False
    for message in row["conversation"]["messages"]:
        role, content = message["role"], message["content"]
        if not isinstance(content, str):
            raise ValueError("HealthBench message content must be text")
        if role in ("system", "developer"):
            if dialogue_started:
                raise ValueError("HealthBench instruction messages must precede the dialogue")
        elif role in ("user", "assistant"):
            dialogue_started = True
        else:
            raise ValueError("Unknown HealthBench message role")
        messages.append((role, content))
    if not messages or messages[-1][0] != "user":
        raise ValueError("HealthBench conversations must end in a user turn")
    return tuple(messages)


def reference_answers(cache_dir: Path) -> Dict[str, str]:
    """Physician texts keyed by the same public IDs as load(), for private calibration."""
    answers = {}
    for row in _rows(cache_dir):
        response = row.get("physician_response")
        if isinstance(response, str) and response.strip():
            answers[_item_id(row["id"])] = response
    return answers


def _verdict(reply: Optional[str]) -> Optional[bool]:
    """Read the last complete JSON object, skipping braces inside decoded objects."""
    if not reply:
        return None
    decoder = json.JSONDecoder()
    position = reply.find("{")
    last = {}
    while position != -1:
        try:
            last, end = decoder.raw_decode(reply, position)
        except (ValueError, RecursionError):
            end = position + 1
        position = reply.find("{", end)
    verdict = last.get("criteria_met")
    return verdict if isinstance(verdict, bool) else None


def _percent(scores: Sequence[float]) -> float:
    if not scores:
        return 0.0
    return 100.0 * max(0.0, min(1.0, sum(scores) / len(scores)))


class HealthBenchProfessional:
    name = "healthbench-professional"
    max_output_tokens = 16384

    def load(self, cache_dir: Path) -> List[Item]:
        items = []
        for row in _rows(cache_dir):
            messages = _conversation(row)
            tags = {name: row[name] for name in (*BREAKDOWNS, "specialty")}
            items.append(Item(
                id=_item_id(row["id"]),
                messages=messages,
                key={"rubric_items": row["rubric_items"], "tags": tags},
                public={
                    "source_id": row["id"],
                    "category": row["use_case"],
                    "messages_sha256": messages_digest(messages),
                },
            ))
        return items

    def judge_requests(self, item: Item, answer: str) -> List[JudgeRequest]:
        if not answer or not answer.strip():
            return []
        transcript = "\n\n".join(f"{role}: {text}" for role, text in (*item.messages, ("assistant", answer)))
        # Expand the template once so literal placeholders in either input survive.
        before, after = GRADER_TEMPLATE.split("<<rubric_item>>")
        before = before.replace("<<conversation>>", transcript)
        return [
            JudgeRequest(
                name=str(index),
                user=before + f"[{criterion['points']}] {criterion['criterion_text']}" + after,
                max_tokens=1024,
            )
            for index, criterion in enumerate(item.key["rubric_items"])
        ]

    def grade(
        self, item: Item, answer: Optional[str], judged: Mapping[str, Optional[str]]
    ) -> Dict[str, Any]:
        criteria = item.key["rubric_items"]
        failed_answer = answer is None or not answer.strip()
        verdicts = [] if failed_answer else [_verdict(judged.get(str(i))) for i in range(len(criteria))]
        failed_calls = verdicts.count(None)
        raw = adjusted = 0.0
        length = None
        if answer is not None and not failed_answer and not failed_calls:
            possible = sum(c["points"] for c in criteria if c["points"] > 0)
            if not possible:
                raise ValueError("HealthBench rubric must include a positive criterion")
            earned = sum(c["points"] for c, met in zip(criteria, verdicts) if met)
            raw = earned / possible
            length = len(answer)
            adjusted = raw - 0.0000294 * (length - 2000)
        return {
            "raw_score": raw,
            "length_adjusted_score": adjusted,
            "answer_length": length,
            "failed_answer": failed_answer,
            "failed_judge_calls": failed_calls,
            "criteria_met": verdicts,
            **{name: item.key["tags"][name] for name in BREAKDOWNS},
        }

    def score(self, grades: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        lengths = [g["answer_length"] for g in grades if g["answer_length"] is not None]
        parts = {}
        for name in BREAKDOWNS:
            groups: Dict[str, List[float]] = defaultdict(list)
            for grade in grades:
                groups[grade[name]].append(grade["raw_score"])
            parts["by_" + name] = {label: _percent(values) for label, values in sorted(groups.items())}
        return {
            "score": _percent([g["length_adjusted_score"] for g in grades]),
            "raw_score": _percent([g["raw_score"] for g in grades]),
            "mean_answer_length": sum(lengths) / len(lengths) if lengths else None,
            "n": len(grades),
            "failed_answers": sum(g["failed_answer"] for g in grades),
            "failed_judge_calls": sum(g["failed_judge_calls"] for g in grades),
            **parts,
        }


BENCHMARK = HealthBenchProfessional()
