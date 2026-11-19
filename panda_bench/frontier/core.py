"""The contract between a benchmark and the frontier runner.

A benchmark turns a pinned public dataset into ``Item`` conversations, says which judge
calls an answer needs (none for an exactly graded benchmark), grades one answer from the
judges' replies, and scores a list of grades the way the benchmark's own board does.

The runner owns everything else: which setup answers (a model, a Panel or a router), rate
limits, retries, the ledger, judge batches and the release files.  A benchmark never calls
a model or a network service except to fetch its pinned dataset in ``load``.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple

# A conversation as the Panda API takes it: an optional prefix of "system" and
# "developer" messages, then "user" and "assistant" turns, the last one a user turn.
Message = Tuple[str, str]


@dataclass(frozen=True)
class Item:
    """One benchmark question.

    ``id`` is stable and public.  ``messages`` is what the setup answers.  ``key`` holds
    everything grading needs (answer key, rubric, category) and is never published, since
    some datasets ask that their content not be reposted.  ``public`` holds what a release
    may show: category, source ids and content digests.
    """

    id: str
    messages: Tuple[Message, ...]
    key: Mapping[str, Any] = field(default_factory=dict)
    public: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class JudgeRequest:
    """One judge call for one answer.  ``name`` is unique within the item (for example a
    rubric criterion's index); the runner sends ``system`` and ``user`` to the judge model
    with ``max_tokens`` and hands the reply text back to ``grade`` under the same name."""

    name: str
    user: str
    system: Optional[str] = None
    max_tokens: int = 2048


class Benchmark(Protocol):
    """What every benchmark module exposes as its module-level ``BENCHMARK``."""

    name: str  # e.g. "gpqa-diamond"; used in file names and release paths
    max_output_tokens: int  # the reply bound the runner sends (Panda allows up to 16,384)

    def load(self, cache_dir: Path) -> List[Item]:
        """The items, in a fixed order, from a pinned and hash-verified download."""

    def judge_requests(self, item: Item, answer: str) -> List[JudgeRequest]:
        """The judge calls ``answer`` needs; empty for an exactly graded benchmark."""

    def grade(self, item: Item, answer: Optional[str], judged: Mapping[str, Optional[str]]) -> Dict[str, Any]:
        """One item's grade.  ``answer`` is None when the setup gave no answer, which
        scores as the board scores a failure.  ``judged`` maps each judge request's name
        to the judge's reply text, or None when that judge call failed.  The result is
        JSON-safe and may be published: no answer key, rubric or question text."""

    def score(self, grades: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        """The board's headline metric and its parts over ``grades`` (one per item),
        as a JSON-safe dict whose ``"score"`` is the headline number on the board's
        scale (for example 0-100)."""


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def messages_digest(messages: Sequence[Message]) -> str:
    """A digest of a conversation that a release can show in place of its text."""
    return sha256_text(json.dumps([list(m) for m in messages], ensure_ascii=False, separators=(",", ":")))


def fetch_pinned(url: str, path: Path, sha256: str, max_bytes: int = 200_000_000) -> bytes:
    """``url``'s bytes, cached at ``path``, refused unless they hash to ``sha256``."""
    import urllib.request

    if path.is_file():
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() == sha256:
            return data
    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "panda-bench"})
    with urllib.request.urlopen(request, timeout=120) as response:
        data = response.read(max_bytes + 1)
    if len(data) > max_bytes or hashlib.sha256(data).hexdigest() != sha256:
        raise ValueError("pinned download does not match its digest: " + url)
    path.write_bytes(data)
    return data


def iter_jsonl(path: Path):
    """Each JSON record of a JSONL file, split on "\\n" only.

    ``str.splitlines`` would also split inside a record on U+2028, U+0085 and other line
    boundaries that ``json.dumps(..., ensure_ascii=False)`` leaves raw in answer text.  A torn
    last record (a crash mid-append) is skipped; a bad record anywhere else is an error."""
    if not path.is_file():
        return
    with open(path, "r", encoding="utf-8", newline="\n") as handle:
        lines = handle.read().split("\n")
    last = max((i for i, line in enumerate(lines) if line.strip()), default=-1)
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            yield json.loads(line)
        except ValueError:
            if index == last:
                return
            raise ValueError(f"corrupt record at line {index + 1} of {path}")
