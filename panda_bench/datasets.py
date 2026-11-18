"""Pinned public-data retrieval. Provider libraries are never used here."""
import hashlib
import json
from .storage import ROOT, digest, read_json, read_rows, write_rows

REVISION = "26c49eb39d7d5ce2099b0bbafed5a88dcce954ec"
FILES = {
    "v2": ("e1fbdc5d46deca34e77f92991c23b0476860e26adf2bf28379a3bb3bf00d3ed6", 3695374),
    "v2-hard": ("44e2a0d20f791c51d68083c8acc8916698d4bb8d460d1dbc393b54030dbea40a", 402410),
}


def fetch():
    from pathlib import Path
    import urllib.request
    import pyarrow.parquet as pq
    data = Path(ROOT) / "data"
    data.mkdir(exist_ok=True)
    rows = {}
    for config, (expected_hash, expected_size) in FILES.items():
        path = data / (config + ".parquet")
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
            url = f"https://huggingface.co/datasets/allenai/WildBench/resolve/{REVISION}/{config}/test-00000-of-00001.parquet"
            with urllib.request.urlopen(url, timeout=60) as response:
                content = response.read(expected_size + 1)
            if len(content) != expected_size or hashlib.sha256(content).hexdigest() != expected_hash:
                raise ValueError("Dataset checksum or length mismatch: " + config)
            path.write_bytes(content)
        for row in pq.read_table(path).to_pylist():
            if len(row["conversation_input"]) != 1:
                continue
            rows[(config, "wb-" + row["session_id"])] = {
                "prompt": row["conversation_input"][0]["content"],
                "checklist": list(row.get("checklist") or []),
            }
    selected = {}
    for stage in ("stage-2", "stage-3", "stage-4"):
        manifest = read_json(ROOT / "releases" / stage / "manifest.json")
        for info in manifest["prompts"]:
            payload = rows[(info["config"], info["id"])]
            if digest(payload) != info["sha256"]:
                raise ValueError("Reconstructed prompt/checklist hash mismatch")
            selected[info["id"]] = {"id": info["id"], "config": info["config"], **payload}
    write_rows(data / "prompts.jsonl", [selected[k] for k in sorted(selected)])
    print(f"Cached {len(selected)} exact prompt/checklist records from the pinned files; data/ is ignored.")


def load_prompts(manifest, allow_fetch=False):
    path = ROOT / "data" / "prompts.jsonl"
    if not path.is_file():
        if not allow_fetch:
            raise ValueError("Dataset cache missing; run panda-bench fetch")
        fetch()
    cache = {r["id"]: r for r in read_rows(path)}
    result = {}
    for info in manifest["prompts"]:
        row = cache.get(info["id"])
        if row is None or digest({"prompt": row["prompt"], "checklist": row["checklist"]}) != info["sha256"]:
            raise ValueError("Dataset cache does not match the manifest; run panda-bench fetch")
        result[info["id"]] = row
    return result
