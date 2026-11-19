"""MultiChallenge (Scale AI): realistic multi-turn conversations, each judged by one rubric question.

``BENCHMARK`` is the current public release, the one Scale's board links as its open-source
dataset: Hugging Face ``ScaleAI/MultiChallenge``, 266 conversations after Scale's 2026
revision.  ``REFERENCE_BENCHMARK`` is the 2025 GitHub release that the paper scored, 273
conversations.  It exists for the grader check: ``reference_responses`` holds the paper's
stored model responses and ``REFERENCE_SCORES`` the paper's published scores for them.

The setup answers the conversation's final user turn.  One judge call per answer asks the
item's YES/NO rubric question about that answer alone, with the official judge prompt.  The
score is the unweighted mean of the four axis pass rates, times 100.  The full protocol and
every deviation are in docs/frontier/multichallenge.md.
"""
from __future__ import annotations

import io
import json
import math
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from ..core import Item, JudgeRequest, Message, fetch_pinned, messages_digest, sha256_text

# The current public release, linked from labs.scale.com/leaderboard/multichallenge.
HF_REVISION = "670fd81f270314d0f8f82b34fcfe2187fd28f699"
HF_FILE = "data/test-00000-of-00001.parquet"
HF_SHA256 = "2711646aba61f814823297714aaf757e3a1b71c66c501402231de48490650515"
HF_URL = "https://huggingface.co/datasets/ScaleAI/MultiChallenge/resolve/" + HF_REVISION + "/" + HF_FILE

# The 2025 release on GitHub: the paper's set, the official harness and the stored responses.
GITHUB_COMMIT = "5ccefcca6a39020d66c1383c4e6a809cb07afa33"
GITHUB_RAW = "https://raw.githubusercontent.com/ekwinox117/multi-challenge/" + GITHUB_COMMIT + "/"
GITHUB_FILE = "data/benchmark_questions.jsonl"
GITHUB_SHA256 = "06c72d1ffa387481910d100b146295f28ab68c4ba0032743eac5c9195edb596e"

# The axis labels as both releases spell them, in the paper's table order.
AXES = ("INSTRUCTION_RETENTION", "INFERENCE_MEMORY", "RELIABLE_VERSION_EDITING", "SELF_COHERENCE")
HF_AXIS_COUNTS = {"INSTRUCTION_RETENTION": 69, "INFERENCE_MEMORY": 109, "RELIABLE_VERSION_EDITING": 40, "SELF_COHERENCE": 48}
GITHUB_AXIS_COUNTS = {"INSTRUCTION_RETENTION": 69, "INFERENCE_MEMORY": 113, "RELIABLE_VERSION_EDITING": 41, "SELF_COHERENCE": 50}
VERDICTS = ("YES", "NO")

# JUDGE_PROMPT from src/evaluator.py at GITHUB_COMMIT, character for character; its SHA-256
# is JUDGE_PROMPT_SHA256.  The official evaluator fills it with .format(response, TARGET_QUESTION).
JUDGE_PROMPT = '''You are tasked with evaluating a model response to see if it meets a specific criteria.
The criteria will always be YES/NO evaluation.

The model response is as follows:
<MODEL_RESPONSE>
{}
</MODEL_RESPONSE>

The criteria that the model response must meet is as follows. Be VERY STRICT!:
<CRITERIA>
{}
</CRITERIA>

Print your reasoning followed by your verdict, either "YES" or "NO".'''
JUDGE_PROMPT_SHA256 = "7519dc6e1af487f8a9850239b334f16238032994b763a8b5108ad67bd0d5be7f"

# The official judge answered through OpenAI structured outputs with the repo's pydantic model
# JudgeResponse(reasoning: str, verdict: Literal["YES", "NO"]).  JudgeRequest has no response
# format field, so a system message carries the schema OpenAI's SDK derives from that model.
JUDGE_RESPONSE_SCHEMA = {
    "properties": {
        "reasoning": {"title": "Reasoning", "type": "string"},
        "verdict": {"enum": ["YES", "NO"], "title": "Verdict", "type": "string"},
    },
    "required": ["reasoning", "verdict"],
    "title": "JudgeResponse",
    "type": "object",
    "additionalProperties": False,
}
JUDGE_SYSTEM = (
    "Respond with a single JSON object, with no other text and no code fence, that conforms to "
    "this JSON schema (named JudgeResponse):\n" + json.dumps(JUDGE_RESPONSE_SCHEMA)
)
JUDGE_NAME = "verdict"
JUDGE_MAX_TOKENS = 4096  # the max_tokens the official Evaluator gives its judge

# One Markdown code fence around the whole reply, the only wrapper parse_verdict removes.
_FENCE = re.compile(r"```[A-Za-z]*[ \t]*\r?\n(.*?)\s*```", re.DOTALL)


def parse_verdict(reply: Optional[str]) -> Optional[str]:
    """``"YES"`` or ``"NO"``, read as the official harness reads its judge's reply, or None.

    The harness parses the reply with pydantic's ``JudgeResponse.model_validate_json``: one
    JSON object, ``reasoning`` a string, ``verdict`` exactly ``"YES"`` or ``"NO"``, other keys
    ignored.  Python's strict ``json.loads`` accepts and rejects the same texts here.  The one
    tolerance is a single code fence around the object, which the official judge's constrained
    decoding could not produce.  Anything else is a failed judge call."""
    if not isinstance(reply, str):
        return None
    text = reply.strip()
    fenced = _FENCE.fullmatch(text)
    if fenced:
        text = fenced.group(1)
    try:
        parsed = json.loads(text)
    except (ValueError, RecursionError):
        parsed = _last_verdict_object(text)
        if parsed is None:
            return _prose_verdict(text) or _bare_verdict(text)
    if not isinstance(parsed, dict) or not isinstance(parsed.get("reasoning"), str):
        return None
    verdict = parsed.get("verdict")
    return verdict if isinstance(verdict, str) and verdict in VERDICTS else None


_PROSE_VERDICT = re.compile(r"verdict\W{0,6}(YES|NO)\b", re.IGNORECASE)


def _prose_verdict(text: str) -> Optional[str]:
    """The verdict of a reply that follows the official prompt's own words ("Print your
    reasoning followed by your verdict, either "YES" or "NO"") without a JSON object: the
    last "Verdict: YES" or "Verdict: NO" (any case, Markdown emphasis allowed).  The stand-in
    judge ended 28 of 819 grader-check replies this way."""
    matches = _PROSE_VERDICT.findall(text)
    return matches[-1].upper() if matches else None


_BARE_VERDICT = re.compile(r"[*_#>\s]*(YES|NO)[*_.!\s]*", re.IGNORECASE)


def _bare_verdict(text: str) -> Optional[str]:
    """The verdict of a reply that ends exactly as the official prompt asks: its reasoning,
    then a last line that is only "YES" or "NO" (Markdown emphasis allowed).  The stand-in
    judge ended 75 of the main run's 2,925 MultiChallenge replies this way.  A reply that is
    only the word, with no reasoning before it, is still a failed judge call."""
    lines = [line for line in text.split("\n") if line.strip()]
    if len(lines) < 2:
        return None
    match = _BARE_VERDICT.fullmatch(lines[-1])
    return match.group(1).upper() if match else None


def _last_verdict_object(text: str) -> Optional[Dict[str, Any]]:
    """The last JSON object in ``text`` whose ``verdict`` is ``"YES"`` or ``"NO"``.

    The official judge's constrained decoding returned the object alone.  The stand-in judge
    (Claude Sonnet 5, without constrained decoding) sometimes writes its reasoning as prose
    first and the requested object last, despite the system message; in a grader check on
    the paper's stored responses this was 278 of 819 replies.  The object it ends with is its
    verdict, so it is read; a reply with no such object is still a failed judge call."""
    decoder = json.JSONDecoder()
    found: Optional[Dict[str, Any]] = None
    index = text.find("{")
    while index != -1:
        try:
            value, _ = decoder.raw_decode(text, index)
        except (ValueError, RecursionError):
            value = None
        if isinstance(value, dict) and value.get("verdict") in VERDICTS:
            found = value
        index = text.find("{", index + 1)
    return found


def _read_hf(cache_dir: Path) -> List[Dict[str, Any]]:
    import pyarrow.parquet as pq

    data = fetch_pinned(HF_URL, cache_dir / "huggingface" / HF_FILE, HF_SHA256)
    rows = []
    for raw in pq.read_table(io.BytesIO(data)).to_pylist():
        conversation = raw.get("conversation") or {}
        roles, contents = conversation.get("role"), conversation.get("content")
        if not isinstance(roles, list) or not isinstance(contents, list) or len(roles) != len(contents):
            raise ValueError("MultiChallenge conversation must pair each role with a content")
        if raw.get("num_turns") != len(roles):
            raise ValueError("MultiChallenge num_turns must count the conversation's messages")
        rows.append({
            "source_id": raw.get("question_id"),
            "axis": raw.get("axis"),
            "conversation": [{"role": role, "content": content} for role, content in zip(roles, contents)],
            "target_question": raw.get("target_question"),
            "pass_criteria": raw.get("pass_criteria"),
        })
    return rows


def _read_github(cache_dir: Path) -> List[Dict[str, Any]]:
    data = fetch_pinned(GITHUB_RAW + GITHUB_FILE, cache_dir / "github" / GITHUB_FILE, GITHUB_SHA256)
    rows = []
    for line in data.split(b"\n"):
        if not line.strip():
            continue
        raw = json.loads(line)
        if not isinstance(raw, dict):
            raise ValueError("MultiChallenge row must be a JSON object")
        rows.append({
            "source_id": raw.get("QUESTION_ID"),
            "axis": raw.get("AXIS"),
            "conversation": raw.get("CONVERSATION"),
            "target_question": raw.get("TARGET_QUESTION"),
            "pass_criteria": raw.get("PASS_CRITERIA"),
        })
    return rows


def _messages(conversation: Any) -> Tuple[Message, ...]:
    """The conversation exactly as given: optional leading system messages, then user and
    assistant turns that alternate from a user turn to the final user turn."""
    if not isinstance(conversation, list) or not conversation:
        raise ValueError("MultiChallenge conversation must be a non-empty list")
    messages = []
    for message in conversation:
        if not isinstance(message, dict) or set(message) != {"role", "content"} or not isinstance(message["content"], str):
            raise ValueError("MultiChallenge message must be exactly a role and a text content")
        messages.append((message["role"], message["content"]))
    start = 0
    while start < len(messages) and messages[start][0] == "system":
        start += 1
    turns = [role for role, _ in messages[start:]]
    if not turns or turns[-1] != "user" or turns != ["user" if i % 2 == 0 else "assistant" for i in range(len(turns))]:
        raise ValueError("MultiChallenge turns must alternate user and assistant and end in a user turn")
    return tuple(messages)


def _items(rows: Sequence[Mapping[str, Any]], id_prefix: str) -> List[Item]:
    items, seen = [], set()
    for row in rows:
        source_id, axis = row["source_id"], row["axis"]
        rubric, criteria = row["target_question"], row["pass_criteria"]
        if not isinstance(source_id, str) or not source_id or source_id in seen:
            raise ValueError("MultiChallenge question ids must be distinct non-empty strings")
        seen.add(source_id)
        if axis not in AXES:
            raise ValueError("Unknown MultiChallenge axis")
        if criteria not in VERDICTS:
            raise ValueError("MultiChallenge pass criteria must be YES or NO")
        if not isinstance(rubric, str) or not rubric.strip():
            raise ValueError("MultiChallenge rubric question must be text")
        messages = _messages(row["conversation"])
        items.append(Item(
            id=id_prefix + sha256_text(source_id)[:16],
            messages=messages,
            key={"source_id": source_id, "axis": axis, "target_question": rubric, "pass_criteria": criteria},
            public={
                "source_id": source_id,
                "category": axis,
                "messages_sha256": messages_digest(messages),
                "rubric_sha256": sha256_text(rubric),
            },
        ))
    if len({item.id for item in items}) != len(items):
        raise ValueError("MultiChallenge item ids collide")
    return items


def _failed_answer(answer: Optional[str]) -> bool:
    return answer is None or not answer.strip()


class MultiChallenge:
    """One pinned MultiChallenge release as a frontier ``Benchmark``."""

    max_output_tokens = 16384

    def __init__(self, name: str, id_prefix: str, read_rows: Callable[[Path], List[Dict[str, Any]]],
                 axis_counts: Mapping[str, int]) -> None:
        self.name = name
        self.id_prefix = id_prefix
        self.axis_counts = dict(axis_counts)
        self._read_rows = read_rows

    def load(self, cache_dir: Path) -> List[Item]:
        items = _items(self._read_rows(Path(cache_dir)), self.id_prefix)
        counts = {axis: 0 for axis in AXES}
        for item in items:
            counts[item.key["axis"]] += 1
        if counts != self.axis_counts:
            raise ValueError(f"{self.name} axis counts {counts} differ from the pinned release's {self.axis_counts}")
        return items

    def judge_requests(self, item: Item, answer: str) -> List[JudgeRequest]:
        if _failed_answer(answer):
            return []
        # str.format inserts the answer and rubric verbatim; braces inside them stay literal.
        return [JudgeRequest(
            name=JUDGE_NAME,
            user=JUDGE_PROMPT.format(answer, item.key["target_question"]),
            system=JUDGE_SYSTEM,
            max_tokens=JUDGE_MAX_TOKENS,
        )]

    def grade(self, item: Item, answer: Optional[str], judged: Mapping[str, Optional[str]]) -> Dict[str, Any]:
        axis = item.key["axis"]
        if _failed_answer(answer):
            return {"axis": axis, "verdict": None, "passed": False, "failed_answer": True, "failed_judge_calls": 0}
        verdict = parse_verdict(judged.get(JUDGE_NAME))
        return {
            "axis": axis,
            "verdict": verdict,
            "passed": verdict is not None and verdict == item.key["pass_criteria"],
            "failed_answer": False,
            "failed_judge_calls": int(verdict is None),
        }

    def score(self, grades: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        """The board's headline: each axis's pass rate, then their unweighted mean, times 100.

        A failed answer or failed judge call counts as not passed in its own axis, as the
        official harness intends, and is counted here; it never forms an axis of its own.
        ``ci95`` is a normal-approximation half-width from each axis's binomial variance."""
        tallies = {axis: {"n": 0, "passed": 0, "failed_answers": 0, "failed_judge_calls": 0} for axis in AXES}
        for grade in grades:
            tally = tallies.get(grade.get("axis"))
            if tally is None:
                raise ValueError("Grade has no MultiChallenge axis")
            failed_answer = bool(grade.get("failed_answer"))
            failed_call = bool(grade.get("failed_judge_calls"))
            tally["n"] += 1
            tally["failed_answers"] += failed_answer
            tally["failed_judge_calls"] += failed_call
            tally["passed"] += grade.get("passed") is True and not failed_answer and not failed_call
        present = [t for t in tallies.values() if t["n"]]
        rates = [t["passed"] / t["n"] for t in present]
        variance = sum(p * (1 - p) / t["n"] for p, t in zip(rates, present))
        axes ={axis: dict(t, score=100.0 * t["passed"] / t["n"] if t["n"] else None) for axis, t in tallies.items()}
        return {
            "score": 100.0 * sum(rates) / len(rates) if rates else None,
            "ci95": 100.0 * 1.96 * math.sqrt(variance) / len(rates) if rates else None,
            "axes": axes,
            "n": sum(t["n"] for t in tallies.values()),
            "passed": sum(t["passed"] for t in tallies.values()),
            "failed_answers": sum(t["failed_answers"] for t in tallies.values()),
            "failed_judge_calls": sum(t["failed_judge_calls"] for t in tallies.values()),
        }


BENCHMARK = MultiChallenge("multichallenge", "mc-", _read_hf, HF_AXIS_COUNTS)
REFERENCE_BENCHMARK = MultiChallenge("multichallenge-2025", "mc25-", _read_github, GITHUB_AXIS_COUNTS)

# The paper's stored responses at GITHUB_COMMIT: (model key = file stem, file, SHA-256).
REFERENCE_FILES = (
    ("claude-3-5-sonnet-20241022", "claude-3-5-sonnet-20241022_responses.jsonl", "d83582385768fdc9006e0882dc9f11cb6ac8cdec06eae3a25a7e2ea88a4761fb"),
    ("gemini-1.5-pro-002", "gemini-1.5-pro-002_responses.jsonl", "3be040a7bdd20672eba209de31ca27c738de7df9c5af361872db707000b3c59f"),
    ("gpt-4o-2024-08-06", "gpt-4o-2024-08-06_responses.jsonl", "12579bda05fb4dcc7330b093d93cd3eec5d89005f3d1be559fe7457048a31b96"),
    ("llama-3-2-3b-instruct", "llama-3-2-3b-instruct.jsonl", "6750f070f7afc1b8c6ccee255a70e0e14020ce78e665e09247fcb04cc7b07b62"),
    ("llama-3-3-70b-instruct", "llama-3-3-70b-instruct.jsonl", "9c65e6227ac2cf7315be55f72534ff678942f6a0b60d8cc097d74571f0134728"),
    ("llama3-1-405b-instruct-v1", "llama3-1-405b-instruct-v1_responses.jsonl", "342d6fb0007e84d5c4370ece2994207b34cdd477e1c6c68f053e10df61e3f55d"),
    ("mistral-large-latest", "mistral-large-latest_responses.jsonl", "2db7c28bbc171f27fd3ad1f3ce24966436ed9e38b2d35500a00217249706c6c7"),
    ("mixtral-8x7b-instruct", "mixtral-8x7b-instruct.jsonl", "1b30398599d3c7091c1edbde518e8f8e54552daef1896eb73efed78864a526d0"),
    ("o1-preview", "o1-preview_responses.jsonl", "94398716d955e961db3b0d1ca676d8209f3246597dcf1ddd9930b3ad5adef255"),
    ("qwen2-5_14b", "qwen2-5_14b.jsonl", "5e24d7891c9a3b9b1ceac84e1731c115a851d3c08bd146cf650a1263802b6104"),
    ("qwen2-5_72b", "qwen2-5_72b.jsonl", "7607a7cf69912343016c06aa36bd2b4d43b63866d595826ed8d08e15a94fde21"),
    ("qwen2_72b", "qwen2_72b.jsonl", "5a9c9275db9899a32bb70fe6f46ac7ec19d9ba427a2601a53dbc60048872ff70"),
)

# Table 3 percentages are exact fractions of a pre-release axis split, never published as data;
# Table 5 percentages are exact fractions of the GitHub release's split (GITHUB_AXIS_COUNTS).
PAPER_TABLE3_AXIS_N = {"INSTRUCTION_RETENTION": 70, "INFERENCE_MEMORY": 118, "RELIABLE_VERSION_EDITING": 41, "SELF_COHERENCE": 44}
PAPER = "Sirdeshmukh et al., MultiChallenge, arXiv:2501.17399v2 (6 Mar 2025), https://arxiv.org/abs/2501.17399v2"
_PAPER_ROWS = (
    # (model key, the paper's row label, table, IR, IM, RVE, SC, Average, passes out of 273)
    ("gpt-4o-2024-08-06", "GPT-4o (August 2024)", 3, 12.86, 5.93, 14.63, 18.18, 12.9, 30),
    ("llama3-1-405b-instruct-v1", "Llama 3.1 405B Instruct", 3, 14.29, 16.95, 9.76, 25.0, 16.5, 45),
    ("mistral-large-latest", "Mistral Large", 3, 18.57, 6.78, 9.76, 18.18, 13.32, 33),
    ("claude-3-5-sonnet-20241022", "Claude 3.5 Sonnet (June 2024)", 3, 61.43, 37.29, 26.83, 45.45, 42.75, 118),
    ("gemini-1.5-pro-002", "Gemini 1.5 Pro (August 27 2024)", 3, 31.43, 16.1, 19.51, 18.18, 21.3, 57),
    ("o1-preview", "o1-preview", 3, 28.57, 38.98, 39.02, 36.36, 35.73, 98),
    ("llama-3-2-3b-instruct", "Llama-3.2-3B-Instruct", 5, 15.94, 8.85, 36.59, 6.0, 16.85, 39),
    ("llama-3-3-70b-instruct", "Llama-3.3-70B-Instruct", 5, 33.33, 15.04, 24.39, 20.0, 23.19, 60),
    ("qwen2_72b", "Qwen2-72B-Instruct", 5, 27.54, 7.96, 26.83, 20.0, 20.58, 49),
    ("qwen2-5_14b", "Qwen2.5-14B-Instruct", 5, 15.94, 15.93, 24.39, 12.0, 17.07, 45),
    ("qwen2-5_72b", "Qwen2.5-72B-Instruct", 5, 21.74, 17.70, 12.20, 16.0, 16.91, 48),
    ("mixtral-8x7b-instruct", "Mixtral-8x7B-Instruct-v0.1", 5, 13.04, 7.08, 12.20, 12.0, 11.08, 28),
)
# The paper's GPT-4o-judged scores for the stored responses.  Compare a re-judge on
# REFERENCE_BENCHMARK by "score" for Table 5 rows (same axis split) and by "passed" for
# Table 3 rows (different split; same 273 conversations).
REFERENCE_SCORES: Dict[str, Dict[str, Any]] = {
    key: {
        "paper_model": label,
        "source": f"{PAPER}, Table {table}",
        "judge": "gpt-4o-2024-08-06",
        "score": average,
        "axes": dict(zip(AXES, (ir, im, rve, sc))),
        "axis_n": dict(PAPER_TABLE3_AXIS_N if table == 3 else GITHUB_AXIS_COUNTS),
        "passed": passes,
        "compare_on": "passed" if table == 3 else "score",
    }
    for key, label, table, ir, im, rve, sc, average, passes in _PAPER_ROWS
}
# Low, middle and high paper scores (28, 60 and 98 passes), no Claude or GPT-4o responses,
# and model versions that match the paper's labels.  With two, keep the first and last.
REFERENCE_RECOMMENDED = ("mixtral-8x7b-instruct", "llama-3-3-70b-instruct", "o1-preview")


def _stored_responses(data: bytes, ids: Mapping[str, str]) -> Dict[str, str]:
    found: Dict[str, str] = {}
    for line in data.split(b"\n"):
        if not line.strip():
            continue
        row = json.loads(line)
        source_id = row.get("QUESTION_ID") if isinstance(row, dict) else None
        response = row.get("RESPONSE") if isinstance(row, dict) else None
        if not isinstance(source_id, str) or source_id not in ids or ids[source_id] in found:
            raise ValueError("Stored responses must name each MultiChallenge conversation once")
        if not isinstance(response, list) or len(response) != 1 or not isinstance(response[0], str):
            raise ValueError("A stored response must be a list holding one text")
        found[ids[source_id]] = response[0]
    if len(found) != len(ids):
        raise ValueError("Stored responses miss some MultiChallenge conversations")
    return {item_id: found[item_id] for item_id in ids.values()}


def reference_responses(cache_dir: Path) -> Dict[str, Dict[str, str]]:
    """The paper's stored responses, by model key and then by the item ids of
    ``REFERENCE_BENCHMARK.load(cache_dir)``.  Judge and score them with
    ``REFERENCE_BENCHMARK``, whose rubrics are the ones the paper's scores used, and
    compare with ``REFERENCE_SCORES``.  For private calibration: never publish the texts."""
    cache_dir = Path(cache_dir)
    ids = {item.key["source_id"]: item.id for item in REFERENCE_BENCHMARK.load(cache_dir)}
    responses = {}
    for model, filename, sha256 in REFERENCE_FILES:
        path = "data/final_model_responses/" + filename
        responses[model] = _stored_responses(fetch_pinned(GITHUB_RAW + path, cache_dir / "github" / path, sha256), ids)
    return responses
