"""Filesystem and JSON helpers; deliberately has no networking dependencies."""
import hashlib
import json
import math
import os

class LocalPath(str):
    """Minimal local paths: pathlib imports urllib.parse on Python 3.9."""
    def __truediv__(self, name):
        return LocalPath(os.path.join(self, str(name)))

    def open(self, *args, **kwargs):
        return open(self, *args, **kwargs)

    def read_bytes(self):
        with self.open("rb") as handle:
            return handle.read()

    def is_file(self):
        return os.path.isfile(self)

    def resolve(self):
        return LocalPath(os.path.realpath(self))


ROOT = LocalPath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def read_json(path):
    with LocalPath(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def read_rows(path):
    with LocalPath(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path, value):
    from pathlib import Path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def write_rows(path, rows):
    from pathlib import Path
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text("".join(canonical(row).decode() + "\n" for row in rows), encoding="utf-8")
    os.replace(temp, path)


def finite_nonnegative(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError("Invalid " + label)
    return value


def bundle_path(name):
    path = ROOT / "releases" / name if name in ("stage-2", "stage-3", "stage-4") else LocalPath(name)
    if not (path / "manifest.json").is_file():
        raise ValueError("No manifest in " + str(path))
    return path.resolve()


def verify_files(path, manifest):
    for name, expected in manifest.get("files", {}).items():
        if os.path.basename(name) != name:
            raise ValueError("Unsafe artifact filename")
        if hashlib.sha256((path / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Artifact checksum mismatch: " + name)


def index_rows(rows):
    result = {}
    for row in rows:
        key = (row["option"], row["prompt_id"])
        if key in result:
            raise ValueError("Duplicate prompt/option row")
        result[key] = row
    return result


def update_latest(path):
    from pathlib import Path
    latest = Path(ROOT) / "runs" / "latest"
    temp = latest.with_name(".latest-tmp")
    if temp.is_symlink():
        temp.unlink()
    temp.symlink_to(Path(path).resolve(), target_is_directory=True)
    if latest.exists() and not latest.is_symlink():
        raise ValueError("runs/latest must be a symlink")
    os.replace(temp, latest)
