import copy
import json
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from panda_bench.analysis import calculate, paired, resolved_grade
from panda_bench.storage import index_rows


class ArithmeticTests(unittest.TestCase):
    def test_fallback_and_failure_are_distinct(self):
        rows = index_rows([{"option":"base","prompt_id":"p","status":"ok","grade":8},
                           {"option":"panel","prompt_id":"p","status":"fallback","fallback_option":"base"},
                           {"option":"failed","prompt_id":"p","status":"failed","grade":10}])
        self.assertEqual(resolved_grade(rows,"panel","p"), 8)
        self.assertEqual(resolved_grade(rows,"failed","p"), 1)
        rows[("base","p")].update(status="fallback",fallback_option="panel")
        with self.assertRaises(ValueError): resolved_grade(rows,"panel","p")

    def test_bootstrap_matches_hand_constant_difference(self):
        settings={"resamples":4000,"seed":11,"lower_index":100,"upper_index":3899}
        self.assertEqual(paired([30,50,70],[20,40,60],settings), (10,10,10))
        self.assertEqual(paired([1,7,2],[2,2,3],settings), paired([1,7,2],[2,2,3],settings))

    def test_duplicate_records_fail(self):
        row={"option":"a","prompt_id":"p"}
        with self.assertRaises(ValueError): index_rows([row,row])

    def test_complete_archive_and_no_network_imports(self):
        script = "from panda_bench.cli import main; import sys; rc=main(['check','stage-4']); bad=[n for n in sys.modules if n.split('.')[0] in {'urllib','http','requests','httpx','openai','anthropic','socket'}]; assert not bad,bad; assert rc==0"
        result=subprocess.run([sys.executable,"-c",script],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)

    def test_cost_and_upper_middle_wait(self):
        m={"stage":"stage-4","prompts":[{"id":"a"},{"id":"b"}],"baseline":"x", "options":[{"id":"x"},{"id":"y"}],"bootstrap":{"resamples":4000,"seed":11,"lower_index":100,"upper_index":3899},"display":{"score_decimals":2,"median_intermediate_decimals":1}}
        grades=[{"prompt_id":i,"option":o,"status":"ok","grade":8 if o=="x" else 9} for i in ['a','b'] for o in ['x','y']]
        usage=[{"prompt_id":i,"option":o,"cost_usd":.02 if o=="x" else .01,"wall_s":1 if i=="a" else 3} for i in ['a','b'] for o in ['x','y']]
        result=calculate(m,grades,usage)
        self.assertEqual(result[1]['cost_pct'],50)
        self.assertEqual(result[1]['cost_1k'],10)
        self.assertEqual(result[1]['wait_p50_s'],3)
        self.assertEqual(result[1]['vs_argon_low'],10)
        usage[0]['cost_usd']=float('nan')
        with self.assertRaises(ValueError): calculate(m,grades,usage)


if __name__ == '__main__': unittest.main()
