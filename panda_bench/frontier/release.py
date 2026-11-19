"""Release files for a frontier run: what anyone may check without our keys or the datasets' text.

``write_release(run_dir, out_dir, benchmarks)`` builds ``out_dir`` from a finished run folder:

- ``manifest.json``: the design frozen in the run's ``plan.jsonl`` (setups, models, combiners,
  output limit, dataset pins, judge and its settings), every start of the run with the harness
  and product revisions it ran on, per benchmark the counts, run window and answers redone
  after a provider failure, and every file's SHA-256;
- ``<benchmark>/items.jsonl``: each item's id and its benchmark's public fields (source id,
  category, prompt digest), never a prompt, rubric, reference answer or option;
- ``<benchmark>/answers.jsonl``: per setup and item, the product status, the route or Panel
  steps, the list cost and Panda's price, the time in model calls, and the answer's SHA-256
  and length, never its text;
- ``<benchmark>/usage.jsonl``: every model call's model, role, outcome, verdict, attempt codes,
  tokens, list cost and seconds;
- ``<benchmark>/judge.jsonl``: the judge reply to every request of every final answer: its id,
  request, result, stop reason, tokens and tries, never its text (a reply can restate a rubric
  line or quote the question);
- ``<benchmark>/grades.jsonl`` and ``<benchmark>/results.json``: per-item grades and scores,
  recomputed from the answers and judge replies as the release is written;
- ``live/``: the paired sample answered through Panda's public API, projected the same way,
  with each setup's live and replay scores on the same items.

Every published field passes an allowlist, and the writer refuses any string longer than
``MAX_STRING`` characters or shaped like a credential or a local path, so a field a producer
adds later cannot carry text out by accident.  The release is built in a staging folder and
swapped in whole, so a failed write never leaves a mix of old and new files.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from . import judge as judging
from .core import iter_jsonl, sha256_text
from .run import CACHE, load_benchmark, read_rows
from .score import _status, retail_cents, score_benchmark, service_seconds

SCHEMA = "panda-bench.frontier-release/v2"
MAX_STRING = 200
ROUTE_FIELDS = ("pick", "selector", "task", "classification", "mode")
PANEL_FIELDS = ("members", "synthesizer", "answered", "draft_order", "mode", "synthesis_verdict", "stand_in")
CALL_FIELDS = ("model", "role", "outcome", "reason", "finish", "verdict", "attempts", "seconds", "wall_seconds",
               "input_tokens", "cached_input_tokens", "output_tokens", "list_cost_nanos")
JUDGE_FIELDS = ("custom_id", "setup", "item", "name", "answer_sha256", "judge", "judge_settings", "result",
                "stop_reason", "error")
LIVE_PANEL_FIELDS = ("name", "revision", "synthesizer_model")
LIVE_AGENT_FIELDS = ("index", "model", "status", "finish_reason", "charged", "cost_microdollars")
# Credentials and paths on our machines never belong in a release, in any field.
FORBIDDEN = re.compile(r"sk-ant-|\bsk-[A-Za-z0-9_-]{16}|Bearer\s|PRIVATE KEY|/Users/|/home/|/private/|\.panda-mvp")


class ReleaseError(ValueError):
    """The run cannot be released as it stands."""


def _pick(mapping: Any, fields: Sequence[str]) -> Optional[Dict[str, Any]]:
    if not isinstance(mapping, Mapping) or not mapping:
        return None
    return {field: mapping[field] for field in fields if field in mapping}


def _numbers(mapping: Any) -> Optional[Dict[str, Any]]:
    """A usage mapping's numeric entries only."""
    if not isinstance(mapping, Mapping):
        return None
    return {key: value for key, value in mapping.items() if isinstance(value, (int, float)) and not isinstance(value, bool)}


def _error_code(error: Any) -> Optional[str]:
    return error.get("code") if isinstance(error, Mapping) else None


def _check(value: Any, where: str) -> None:
    """Refuses a value that could carry text out: a long string, a credential or a local path."""
    if isinstance(value, str):
        if len(value) > MAX_STRING:
            raise ReleaseError(f"{where}: a {len(value)}-character string would be published")
        if FORBIDDEN.search(value):
            raise ReleaseError(f"{where}: a credential or local path would be published")
    elif isinstance(value, Mapping):
        for key, inner in value.items():
            _check(key, where)
            _check(inner, f"{where}.{key}")
    elif isinstance(value, (list, tuple)):
        for inner in value:
            _check(inner, where)
    elif value is not None and not isinstance(value, (bool, int, float)):
        raise ReleaseError(f"{where}: a {type(value).__name__} would be published")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            _check(row, path.name)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def _write_json(path: Path, value: Any) -> None:
    _check(value, path.name)
    path.write_text(json.dumps(value, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def frozen_design(run_dir: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """The design every start of the run declared, and the starts.  A run whose setups, pins,
    output limit or judge changed between starts, or that ran a cut-down pilot, is refused."""
    starts = list(iter_jsonl(run_dir / "plan.jsonl"))
    if not starts:
        raise ReleaseError("the run has no plan.jsonl, so its design was never frozen")
    benchmarks: Dict[str, Any] = {}
    judges = set()
    for start in starts:
        if start.get("limit") is not None:
            raise ReleaseError(f"the start at {start.get('at')} answered only {start['limit']} items per benchmark")
        judges.add((start.get("judge"), start.get("judge_settings")))
        for name, spec in start["benchmarks"].items():
            if benchmarks.setdefault(name, spec) != spec:
                raise ReleaseError(f"{name}: the design changed between starts")
    if len(judges) != 1:
        raise ReleaseError("the judge changed between starts")
    (judge, settings), = judges
    return {"judge": judge, "judge_settings": settings, "benchmarks": benchmarks}, starts


def _answer_row(setup: str, item_id: str, row: Mapping[str, Any], rows_written: int) -> Dict[str, Any]:
    text = row.get("text")
    cost = row.get("list_cost_nanos") or 0
    return {
        "setup": setup, "item": item_id, "status": _status(dict(row)),
        "route": _pick(row.get("route"), ROUTE_FIELDS), "panel": _pick(row.get("panel"), PANEL_FIELDS),
        "list_cost_nanos": cost, "retail_cents": retail_cents(cost, text is not None),
        "service_seconds": service_seconds(dict(row)), "wall_seconds": row.get("wall_seconds"),
        "answer_sha256": sha256_text(text) if text is not None else None,
        "answer_chars": len(text) if text is not None else 0,
        "rows_written": rows_written, "at": row.get("at"),
    }


def _judge_rows(run_dir: Path, name: str, wanted: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """The reply to each request of each final answer, newest try, without its text."""
    path = run_dir / "judge" / (name + ".jsonl")
    judged = judging.read_judged(path)
    tries = judging._attempts(path)
    rows = []
    for cid in sorted(wanted):
        row = judged.get(cid)
        if row is not None:
            rows.append({**(_pick(row, JUDGE_FIELDS) or {}), "usage": _numbers(row.get("usage")), "tries": tries.get(cid, 1)})
    return rows


def _benchmark(run_dir: Path, folder: Path, name: str, spec: Mapping[str, Any], allow_incomplete: bool) -> Dict[str, Any]:
    bench = load_benchmark(name)
    items = bench.load(CACHE / name)
    by_id = {item.id: item for item in items}
    setups = [setup["name"] for setup in spec["setups"]]
    path = run_dir / "answers" / (name + ".jsonl")
    written = Counter((row["setup"], row["item"]) for row in iter_jsonl(path))
    answers = read_rows(path)
    strays = sorted({setup for setup, _ in answers} - set(setups))
    if strays:
        raise ReleaseError(f"{name}: answers from setups outside the frozen design: {strays}")
    unknown = sorted({item for _, item in answers} - set(by_id))
    if unknown:
        raise ReleaseError(f"{name}: answers to {len(unknown)} items outside the pinned dataset")
    folder.mkdir(parents=True)
    _write_jsonl(folder / "items.jsonl", ({"id": item.id, **dict(item.public)} for item in items))

    answer_rows: List[Dict[str, Any]] = []
    usage_rows: List[Dict[str, Any]] = []
    wanted: Dict[str, Tuple[str, str, str]] = {}
    for (setup, item_id), row in sorted(answers.items()):
        answer_rows.append(_answer_row(setup, item_id, row, written[(setup, item_id)]))
        for call in row.get("calls", []):
            usage_rows.append({"setup": setup, "item": item_id, **(_pick(call, CALL_FIELDS) or {})})
        text = row.get("text")
        if text is not None and row.get("status") != "harness_error":
            digest = sha256_text(text)
            for request in bench.judge_requests(by_id[item_id], text):
                wanted[judging.custom_id(name, setup, item_id, request.name, digest)] = (setup, item_id, request.name)
    _write_jsonl(folder / "answers.jsonl", answer_rows)
    _write_jsonl(folder / "usage.jsonl", usage_rows)
    judge_rows = _judge_rows(run_dir, name, wanted)
    _write_jsonl(folder / "judge.jsonl", judge_rows)

    results = score_benchmark(run_dir, name, folder / "grades.jsonl")
    for row in iter_jsonl(folder / "grades.jsonl"):
        _check(row, "grades.jsonl")
    unfinished = sorted(setup for setup in setups if not results.get(setup, {}).get("complete"))
    if unfinished and not allow_incomplete:
        raise ReleaseError(f"{name}: setups without a complete result: {unfinished}")
    _write_json(folder / "results.json", results)
    stamps = [row["at"] for row in answer_rows if row.get("at")]
    return {
        "items": len(items), "answers": len(answer_rows), "calls": len(usage_rows),
        "judge_requests": len(wanted), "judge_replies": len(judge_rows),
        "rows_written": sum(written.values()),
        "redone": dict(sorted(Counter(setup for (setup, _), count in written.items() if count > 1).items())),
        "window": [min(stamps), max(stamps)] if stamps else None,
        "complete": not unfinished, "unfinished": unfinished,
    }


def _live_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    panda = row.get("panda") if isinstance(row.get("panda"), Mapping) else {}
    text = row.get("text")
    synthesis = panda.get("synthesis") if isinstance(panda.get("synthesis"), Mapping) else None
    return {
        "setup": row.get("setup"), "item": row.get("item"), "status": row.get("status"), "model": row.get("model"),
        "finish": row.get("finish"), "error_code": row.get("error_code"),
        "mode": panda.get("mode"), "route": _pick(panda.get("route"), ROUTE_FIELDS),
        "panel": _pick(panda.get("panel"), LIVE_PANEL_FIELDS),
        "agents": [{**(_pick(agent, LIVE_AGENT_FIELDS) or {}), "error_code": _error_code(agent.get("error"))}
                   for agent in panda.get("agents") or [] if isinstance(agent, Mapping)] or None,
        "synthesis": {**(_pick(synthesis, LIVE_AGENT_FIELDS) or {}), "error_code": _error_code(synthesis.get("error")),
                      "included_agents": synthesis.get("included_agents")} if synthesis else None,
        "charge_cents": row.get("charge_cents"), "usage": _numbers(row.get("usage")),
        "attempts": row.get("attempts"), "wall_seconds": row.get("wall_seconds"),
        "answer_sha256": sha256_text(text) if text is not None else None,
        "answer_chars": len(text) if text is not None else 0, "at": row.get("at"),
    }


def _paired(name: str, live_rows: Sequence[Mapping[str, Any]], live_grades: Path, replay_grades: Path) -> Dict[str, Any]:
    """Per live setup, on the sample it answered: statuses, charges, and its score next to the
    replay's score for the same setup on the same items (each graded by the same judge)."""
    bench = load_benchmark(name)
    graded: Dict[str, List[Dict[str, Any]]] = {}
    for row in iter_jsonl(live_grades):
        graded.setdefault(row["setup"].split(":", 1)[1], []).append(row)
    replay = {(row["setup"], row["item"]): row for row in iter_jsonl(replay_grades)}
    out: Dict[str, Any] = {}
    for row in live_rows:
        setup = str(row["setup"]).split(":", 1)[1]
        entry = out.setdefault(setup, {"items": 0, "statuses": Counter(), "charge_cents": 0})
        entry["items"] += 1
        entry["statuses"][str(row.get("status"))] += 1
        entry["charge_cents"] += int(row.get("charge_cents") or 0)
    for setup, entry in out.items():
        entry["statuses"] = dict(entry["statuses"])
        pairs = [(row, replay[(setup, row["item"])]) for row in graded.get(setup, []) if (setup, row["item"]) in replay]
        entry["paired_items"] = len(pairs)
        entry["live_score"] = bench.score([a for a, _ in pairs]).get("score") if pairs else None
        entry["replay_score"] = bench.score([b for _, b in pairs]).get("score") if pairs else None
    return dict(sorted(out.items()))


def _live(live_dir: Path, folder: Path, benchmarks: Sequence[str], staging: Path) -> Dict[str, Any]:
    folder.mkdir(parents=True)
    summary: Dict[str, Any] = {}
    for name in benchmarks:
        path = live_dir / "answers" / (name + ".jsonl")
        if not path.is_file():
            continue
        rows = [_live_row(row) for _, row in sorted(read_rows(path).items())]
        _write_jsonl(folder / (name + ".answers.jsonl"), rows)
        score_benchmark(live_dir, name, folder / (name + ".grades.jsonl"))
        for row in iter_jsonl(folder / (name + ".grades.jsonl")):
            _check(row, name + ".grades.jsonl")
        summary[name] = _paired(name, rows, folder / (name + ".grades.jsonl"), staging / name / "grades.jsonl")
    _write_json(folder / "results.json", summary)
    return {name: {"answers": sum(entry["items"] for entry in setups.values())} for name, setups in summary.items()}


def _digests(root: Path) -> Dict[str, Dict[str, Any]]:
    files = {}
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        data = path.read_bytes()
        files[path.relative_to(root).as_posix()] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    return files


def write_release(run_dir: Path, out_dir: Path, benchmarks: Sequence[str], notes: Optional[Mapping[str, Any]] = None,
                  live_dir: Optional[Path] = None, extra: Optional[Mapping[str, Any]] = None,
                  allow_incomplete: bool = False) -> Dict[str, Any]:
    """Build the release in a staging folder next to ``out_dir`` and swap it in whole.

    ``notes`` is the publisher's own prose, kept apart from the fields the run fixes;
    ``extra`` maps further JSON files to write at the release root (a grader check's scores,
    say), checked like everything else."""
    design, starts = frozen_design(run_dir)
    missing = [name for name in benchmarks if name not in design["benchmarks"]]
    if missing:
        raise ReleaseError(f"benchmarks the run never planned: {missing}")
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="." + out_dir.name + ".staging-", dir=out_dir.parent))
    os.chmod(staging, 0o755)  # mkdtemp's folder is private; a release is for reading
    try:
        summaries = {name: _benchmark(run_dir, staging / name, name, design["benchmarks"][name], allow_incomplete)
                     for name in benchmarks}
        live = _live(live_dir, staging / "live", benchmarks, staging) if live_dir else None
        for filename, value in (extra or {}).items():
            if "/" in filename or not filename.endswith(".json"):
                raise ReleaseError(f"extra file {filename!r} must be a .json name at the release root")
            _write_json(staging / filename, value)
        manifest = {
            "schema": SCHEMA, "release": out_dir.name,
            "judge": design["judge"], "judge_settings": design["judge_settings"],
            "benchmarks": {name: {**design["benchmarks"][name], **summaries[name]} for name in benchmarks},
            "live": live,
            "starts": [{"at": start.get("at"), "harness": start.get("harness"), "product_source": start.get("product_source"),
                        "benchmarks": sorted(start.get("benchmarks", {}))} for start in starts],
            "files": _digests(staging),
        }
        _check(manifest, "manifest.json")
        manifest["notes"] = dict(notes or {})
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        old = None
        if out_dir.exists():
            old = out_dir.with_name("." + out_dir.name + ".old-" + staging.name.rsplit("-", 1)[-1])
            os.replace(out_dir, old)
        os.replace(staging, out_dir)
        if old is not None:
            shutil.rmtree(old)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return abs(a - b) <= 1e-9 * max(1.0, abs(a), abs(b))
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


def check_release(release_dir: Path) -> List[str]:
    """What differs in a release: a file whose SHA-256 is not the manifest's, or a setup whose
    scores do not follow from its published grades.  Needs no key, dataset or network: each
    benchmark's ``score`` reads grades only.  An empty list means everything checks."""
    manifest = json.loads((release_dir / "manifest.json").read_text(encoding="utf-8"))
    problems = []
    for path, meta in manifest["files"].items():
        target = release_dir / path
        if not target.is_file():
            problems.append(f"{path}: missing")
        elif hashlib.sha256(target.read_bytes()).hexdigest() != meta["sha256"]:
            problems.append(f"{path}: SHA-256 differs from the manifest")
    for name in manifest["benchmarks"]:
        bench = load_benchmark(name)
        grades: Dict[str, List[Dict[str, Any]]] = {}
        for row in iter_jsonl(release_dir / name / "grades.jsonl"):
            grades.setdefault(row["setup"], []).append(row)
        results = json.loads((release_dir / name / "results.json").read_text(encoding="utf-8"))
        for setup, summary in sorted(results.items()):
            recomputed = bench.score(grades[setup]) if grades.get(setup) else {"score": None}
            for key, value in recomputed.items():
                if not _same(value, summary.get(key)):
                    problems.append(f"{name}/{setup}: {key} is {summary.get(key)!r}, its grades give {value!r}")
    return problems
