"""Generation: every setup answers every item of every benchmark, resumably.

Answers append to ``<run>/answers/<benchmark>.jsonl``, one line per (setup, item), as soon as
each finishes, so a stopped run resumes where it was.  Calls are paced per model by the
replay backend; ``<run>/rpm.json`` may be edited while the run goes, and is read again
every minute, to slow down or speed up a model (for example to leave room for production
traffic on Workers AI, whose per-model limit the whole account shares).
"""
from __future__ import annotations

import importlib
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

BENCHMARK_MODULES = {
    "gpqa-diamond": "panda_bench.frontier.benchmarks.gpqa_diamond",
    "simpleqa-verified": "panda_bench.frontier.benchmarks.simpleqa_verified",
    "multichallenge": "panda_bench.frontier.benchmarks.multichallenge",
    "healthbench-professional": "panda_bench.frontier.benchmarks.healthbench_professional",
    "routerarena": "panda_bench.frontier.benchmarks.routerarena",
}
CACHE = Path(os.environ.get("PANDA_BENCH_CACHE", "data/frontier")).expanduser()  # data/ is ignored

# The setups of the November 2026 frontier comparison, and the benchmarks each runs on.
OPEN_ENDED = ("healthbench-professional", "multichallenge")
ALL = ("gpqa-diamond", "simpleqa-verified", "multichallenge", "healthbench-professional")


def setups_for(benchmark: str):
    from . import replay as r

    if benchmark == "routerarena":
        return [r.auto("auto")]
    setups = [
        r.single("deepseek-v4.1-flash"),
        r.single("glm-5.3"),
        r.single("glm-5.3-flash"),
        r.single("gemini-3.8-flash"),
        r.single("gemini-4-argon"),
        r.panda_panel("panda"),
        r.panda_panel("panda-quality"),
        r.panel("panel:trio-glm", ("glm-5.3-flash", "deepseek-v4.1-flash", "gemini-3.8-flash"), "glm-5.3"),
        r.auto("auto"),
        r.auto("auto:quality"),
    ]
    if benchmark in OPEN_ENDED:
        # Self-MoA control: the same model's two drafts merged by itself, so a Panel's gain
        # can be told apart from extra samples (arXiv 2502.00674).
        setups.append(r.panel("control:deepseek-self-merge",
                              ("deepseek-v4.1-flash", "deepseek-v4.1-flash"), "deepseek-v4.1-flash"))
    return setups


DEFAULT_RPM = {
    "@cf/deepseek-ai/deepseek-v4.1-flash-1017": 10,
    "@cf/zai-org/glm-5.3": 8,
    "@cf/zai-org/glm-5.3-flash": 8,
    "@cf/google/embeddinggemma-2": 60,
    "google-vertex-ai/gemini-3.8-flash": 60,
    "google-vertex-ai/gemini-4-argon": 30,
}


def load_benchmark(name: str):
    return importlib.import_module(BENCHMARK_MODULES[name]).BENCHMARK


def read_rows(path: Path) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """The last row for each (setup, item): a retry appends, and the newest row wins."""
    from .core import iter_jsonl

    rows: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in iter_jsonl(path):
        rows[(row["setup"], row["item"])] = row
    return rows


def _transient(row: Dict[str, Any]) -> bool:
    """A failure no model caused: the harness raised, every call ended in a transport error,
    timeout or rate limit after its retries, or such an error cost a Panel one of its calls
    (a provider outage, not the Panel's design).  ``--retry-failed`` sends these again."""
    if row.get("status") == "harness_error":
        return True
    calls = [c for c in row.get("calls", []) if c.get("role") != "router_embedding"]
    if row.get("panel") and row.get("status") != "failed":
        return any(c.get("outcome") == "error" and c.get("reason") == "transport" for c in calls)
    if row.get("status") != "failed":
        return False
    return bool(calls) and all(c.get("outcome") == "error" for c in calls if c.get("outcome") != "text")


class Ledger:
    """Append-only JSONL with resume: the (setup, item) pairs already answered."""

    def __init__(self, path: Path, retry_failed: bool = False) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.done = set()
        for key, row in read_rows(path).items():
            if row.get("status") == "harness_error" or (retry_failed and _transient(row)):
                continue
            self.done.add(key)
        self._lock = threading.Lock()

    def append(self, row: Dict[str, Any]) -> None:
        line = json.dumps(row, ensure_ascii=False) + "\n"
        with self._lock:
            with self.path.open("a") as handle:
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())
            self.done.add((row["setup"], row["item"]))


def _watch_rpm(providers, path: Path, stop: threading.Event) -> None:
    while not stop.wait(60):
        try:
            for model, rpm in json.loads(path.read_text()).items():
                providers.limiter(model).rpm = float(rpm)
        except Exception:
            pass


def _git(path: str) -> Dict[str, Any]:
    import subprocess

    def run(*args):
        return subprocess.run(["git", "-C", path, *args], capture_output=True, text=True).stdout.strip()

    return {"commit": run("rev-parse", "HEAD"), "dirty_files": len([l for l in run("status", "--porcelain").splitlines() if l])}


def write_plan(run_dir: Path, benchmarks: Sequence[str], limit: Optional[int]) -> None:
    """The run's frozen design, appended each time generation starts, so a release can be
    checked against what was actually run: setups, item counts, dataset pins (each module's
    revision and digest constants), the judge settings and the code revisions."""
    import importlib

    from . import judge

    entry: Dict[str, Any] = {"at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "limit": limit,
                             "judge": judge.JUDGE_MODEL, "judge_settings": judge.JUDGE_SETTINGS,
                             "harness": _git(str(Path(__file__).resolve().parents[2])),
                             "product_source": _git(os.environ.get("PANDA_ZERO_SRC", ".")), "benchmarks": {}}
    for name in benchmarks:
        module = importlib.import_module(BENCHMARK_MODULES[name])
        pins = {k: v for k, v in vars(module).items()
                if k.isupper() and isinstance(v, str) and any(t in k for t in ("REVISION", "SHA256", "COMMIT", "URL"))}
        entry["benchmarks"][name] = {"setups": [s.describe() for s in setups_for(name)],
                                     "max_output_tokens": load_benchmark(name).max_output_tokens, "pins": pins}
    with (run_dir / "plan.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")


def _bottleneck(setup) -> str:
    """The scarcest model a setup waits on: its worker pool."""
    deepseek = "@cf/deepseek-ai/deepseek-v4.1-flash-1017"
    models = set(setup.models) | ({setup.synthesizer} if setup.synthesizer else set())
    if setup.kind == "auto":
        return "deepseek" if setup.selector == "auto" else "auto-quality"
    if deepseek in models:
        return "deepseek"
    if "@cf/zai-org/glm-5.3" in models:
        return "glm"
    if "@cf/zai-org/glm-5.3-flash" in models:
        return "glm-flash"
    return "vertex"


def generate(run_dir: Path, benchmarks: Sequence[str], limit: Optional[int], workers: int,
             only_setups: Optional[Sequence[str]] = None, max_list_usd: float = 400.0,
             retry_failed: bool = False) -> None:
    from . import replay as r

    run_dir.mkdir(parents=True, exist_ok=True)
    write_plan(run_dir, benchmarks, limit)
    rpm_path = run_dir / "rpm.json"
    if not rpm_path.is_file():
        rpm_path.write_text(json.dumps(DEFAULT_RPM, indent=1))
    providers = r.Providers(json.loads(rpm_path.read_text()))
    runner = r.Runner(providers)
    stop = threading.Event()
    threading.Thread(target=_watch_rpm, args=(providers, rpm_path, stop), daemon=True).start()

    tasks: List[Tuple[str, Any, Any, Any, Ledger]] = []
    for name in benchmarks:
        bench = load_benchmark(name)
        items = bench.load(CACHE / name)
        if limit is not None:
            items = items[:limit]
        ledger = Ledger(run_dir / "answers" / (name + ".jsonl"), retry_failed)
        for setup in setups_for(name):
            if only_setups and setup.name not in only_setups:
                continue
            for item in items:
                if (setup.name, item.id) not in ledger.done:
                    tasks.append((name, bench, setup, item, ledger))
    # Interleave benchmarks and setups so every model's queue stays busy.  An earlier
    # version ran the GLM-5.3 Flash setups first (retiring 2026-11-20); with every worker
    # then waiting on that one model's per-minute limit, the other models sat idle.  A full
    # shuffle keeps each model near its own limit, and GLM-5.3 Flash still finishes early.
    random.Random(7).shuffle(tasks)
    total = len(tasks)
    spent = [0.0]
    progress = {"done": 0, "failed": 0, "began": time.time()}
    lock = threading.Lock()
    log = (run_dir / "generate.log").open("a")

    def work(task):
        name, bench, setup, item, ledger = task
        if spent[0] > max_list_usd:
            return
        run_id = f"panda-bench.frontier.v1/{name}/{setup.name}/{item.id}"
        base = {"benchmark": name, "setup": setup.name, "item": item.id, "run_id": run_id,
                "setup_spec": setup.describe(), "max_output_tokens": bench.max_output_tokens}
        try:
            answer = runner.answer(setup, item.messages, bench.max_output_tokens, run_id)
        except Exception as error:  # a harness fault: recorded, never scored, retried on resume
            ledger.append({**base, "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                           "status": "harness_error", "text": None, "calls": [],
                           "error": type(error).__name__ + ": " + str(error)[:200]})
            with lock:
                progress["failed"] += 1
            return
        row = {**base, "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **answer.to_json()}
        ledger.append(row)
        with lock:
            spent[0] += answer.list_cost_nanos() / 1e9
            progress["done"] += 1
            progress["failed"] += answer.text is None
            if progress["done"] % 25 == 0 or progress["done"] == total:
                rate = progress["done"] / max(1.0, time.time() - progress["began"]) * 60
                log.write(json.dumps({"at": row["at"], "done": progress["done"], "total": total,
                                      "failed": progress["failed"], "per_min": round(rate, 1),
                                      "list_usd": round(spent[0], 4)}) + "\n")
                log.flush()

    log.write(json.dumps({"start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "tasks": total,
                          "benchmarks": list(benchmarks), "limit": limit}) + "\n")
    log.flush()
    # One worker pool per bottleneck: a task waiting in DeepSeek's per-minute limiter holds
    # its worker for minutes while a Gemini task takes seconds, so in one shared pool the
    # waiters crowd out everything else.  Each pool is sized from ``workers``.
    pools: Dict[str, List[Any]] = {}
    for task in tasks:
        pools.setdefault(_bottleneck(task[2]), []).append(task)
    shares = {"deepseek": 0.40, "glm": 0.15, "glm-flash": 0.10, "vertex": 0.25, "auto-quality": 0.10}
    executors = {key: ThreadPoolExecutor(max_workers=max(8, int(workers * shares.get(key, 0.1))))
                 for key in pools}
    futures = [executors[key].submit(work, task) for key, group in pools.items() for task in group]
    log.write(json.dumps({"pools": {key: len(group) for key, group in pools.items()}}) + "\n")
    log.flush()
    for future in as_completed(futures):
        error = future.exception()
        if error is not None:
            log.write(json.dumps({"error": type(error).__name__, "detail": str(error)[:300]}) + "\n")
            log.flush()
    for executor in executors.values():
        executor.shutdown(wait=True)
    stop.set()
    log.write(json.dumps({"end": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                          "done": progress["done"], "failed": progress["failed"],
                          "list_usd": round(spent[0], 4)}) + "\n")
    log.close()


def reconcile(run_dir: Path, benchmarks: Sequence[str]) -> Dict[str, int]:
    """Marks for regeneration the rows written before the replay classified calls with the
    product's own verdicts (2026-11-18 ~16:40Z, after an independent review) whose outcome
    production would have handled differently:

    - a call with text but no usage, which production cannot settle (outcome unknown);
    - a Panel whose combiner was refused in content and whose member stood in (production
      returns the refusal: "answer_refused" is not a stand-in code);
    - a Panel whose combiner was definitely refused before inference (an HTTP 4xx) and that
      failed (production's strongest member stands in: "model_refused" is a stand-in code).

    Run it with no generation running: it appends ``harness_error`` rows, which the next
    generation run answers again with the current code."""
    counts: Dict[str, int] = {}
    for name in benchmarks:
        path = run_dir / "answers" / (name + ".jsonl")
        marks = []
        for row in read_rows(path).values():
            calls = row.get("calls") or []
            if row.get("status") == "harness_error" or not calls or any("verdict" in c for c in calls):
                continue
            divergent = any(c.get("outcome") == "text" and not c.get("input_tokens") and not c.get("output_tokens")
                            and c.get("role") != "router_embedding" for c in calls)
            panel = row.get("panel") or {}
            synthesis = [c for c in calls if c.get("role") == "synthesis"]
            if synthesis:
                last = synthesis[-1]
                if panel.get("mode") == "stand_in" and last.get("outcome") == "refused":
                    divergent = True
                if panel.get("mode") == "synthesis_failed" and (last.get("attempts") or [""])[-1].startswith("http_4"):
                    divergent = True
            if divergent:
                marks.append(row)
        with path.open("a", encoding="utf-8") as handle:
            for row in marks:
                handle.write(json.dumps({"benchmark": row["benchmark"], "setup": row["setup"], "item": row["item"],
                                         "run_id": row.get("run_id"), "status": "harness_error", "text": None, "calls": [],
                                         "error": "reconcile: production would classify this outcome differently",
                                         "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}) + "\n")
        counts[name] = len(marks)
    return counts
