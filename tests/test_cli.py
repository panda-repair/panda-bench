import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from panda_bench.storage import ROOT, read_json, write_json, write_rows


class CliTests(unittest.TestCase):
    def invoke(self, arguments, guard=False):
        script = Path(sys.executable).parent / "panda-bench"
        if not script.exists():
            self.skipTest("Install editable kit to test the console entry point")
        env = dict(os.environ)
        env.pop("PANDA_API_KEY", None); env.pop("ANTHROPIC_API_KEY", None)
        if guard:
            code = """import builtins, sys
original = builtins.__import__
blocked = {'urllib', 'http', 'requests', 'httpx', 'openai', 'anthropic', 'socket', 'aiohttp'}
def safe_import(name, *args, **kwargs):
    if name.split('.')[0] in blocked: raise AssertionError('Forbidden import: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = safe_import
script, *arguments = sys.argv[1:]
sys.argv = [script, *arguments]
try:
    exec(compile(open(script).read(), script, 'exec'), {'__name__': '__main__'})
except SystemExit as exc:
    if exc.code: raise
assert not blocked.intersection({m.split('.')[0] for m in sys.modules})
"""
            command = [sys.executable, "-c", code, str(script), *arguments]
        else:
            command = [str(script), *arguments]
        return subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True)

    def test_installed_check_never_imports_network_or_provider(self):
        result = self.invoke(["check", "stage-4"], guard=True)
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.assertIn("All displayed arithmetic matches.", result.stdout)

    def test_regrade_and_run_plans_need_no_keys(self):
        result = self.invoke(["regrade", "stage-4", "--max-cost-usd", "15", "--plan"])
        self.assertEqual(result.returncode, 0, result.stderr)
        plan = json.loads(result.stdout)
        self.assertEqual(plan["request_count"], 150)
        self.assertFalse(plan["within_budget"])
        result = self.invoke(["run", "stage-4", "--models", "panel:panda,panel:panda-quality,gemini-4-argon", "--max-cost-usd", "30", "--plan"])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["request_count"], 450)

    def test_grade_plan_uses_new_bundle_and_refuses_ten_dollars(self):
        manifest = read_json(ROOT / "releases" / "stage-4" / "manifest.json")
        manifest.pop("files"); manifest["schema"] = "panda-bench.run/v1"; manifest["run_state"] = "completed"
        models = ["panel:panda", "panel:panda-quality", "gemini-4-argon"]
        manifest["judge"]["bundle_before_shuffle"] = {p["id"]: models for p in manifest["prompts"]}
        answers = [{"prompt_id": p["id"], "option": m, "text": "Synthetic answer", "status": "ok"} for p in manifest["prompts"] for m in models]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp); write_json(path / "manifest.json", manifest); write_rows(path / "answers.jsonl", answers)
            result = self.invoke(["grade", str(path), "--max-cost-usd", "10", "--plan"])
            self.assertEqual(result.returncode, 0, result.stderr)
            plan = json.loads(result.stdout)
            self.assertEqual(plan["answer_count"], 450)
            self.assertFalse(plan["within_budget"])
            self.assertIsNone(plan["historical_reference_usd"])
            result = self.invoke(["grade", str(path), "--max-cost-usd", "10"])
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("Conservative judge exposure exceeds", result.stdout)

    def test_plans_do_not_initialize_sdks_or_network(self):
        code = """import builtins, sys
original = builtins.__import__
def safe_import(name, *args, **kwargs):
    if name.split('.')[0] in {'anthropic', 'openai', 'requests', 'httpx', 'socket'}: raise AssertionError(name)
    return original(name, *args, **kwargs)
builtins.__import__ = safe_import
from panda_bench.cli import main
assert main(['regrade','stage-4','--max-cost-usd','15','--plan']) == 0
assert main(['run','stage-4','--models','panel:panda,panel:panda-quality,gemini-4-argon','--max-cost-usd','30','--plan']) == 0
"""
        env = dict(os.environ); env.pop("PANDA_API_KEY", None); env.pop("ANTHROPIC_API_KEY", None)
        result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__": unittest.main()
