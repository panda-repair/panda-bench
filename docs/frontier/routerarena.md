# RouterArena, `sub_10` split (`routerarena`)

Module: `panda_bench/frontier/benchmarks/routerarena.py` · tests: `tests/frontier/test_routerarena.py`

RouterArena ([paper](https://arxiv.org/abs/2510.00202), ICLR 2026;
[code](https://github.com/RouteWorks/RouterArena)) ranks LLM routers. For every query the
router names one model; that model's answer is graded, and the board reports accuracy, dollars
per 1,000 queries, and an "Acc-Cost Arena" score that combines the two. Commercial routers
enter by pull request with a config, a prediction file (each query's chosen model, that model's
answer and token usage, and the router's own model calls in `router_usage`) and a robustness
file. RouterArena's CI grades the full split (8,400 queries).

This module runs the `sub_10` split: 809 queries, a subset of `full`. RouterArena's evaluator
notes that `sub_10` "is generally easier than the full dataset". Every query is graded by rules;
there is no LLM judge, so the module returns no judge requests. Coding queries run
model-written code, which this module only ever runs inside a locked-down container.

## Pins

| What | Where | Revision | SHA-256 |
|---|---|---|---|
| Queries (`sub_10`, 809 rows) | HF `RouteWorks/RouterArena`, `data/sub_10-00000-of-00001.parquet` | `a4a062ce3313b56bb09c042e1bc37b61d34e3bd8` | `e7b4b7579e98df0b83202324357d0b91af78666ddbf1038a513d19edaf58efd5` |
| Full split (read only to rebuild RouterArena's LiveCodeBench matching) | same repo, `data/full-00000-of-00001.parquet` | same | `358a6bb4160320b5cb5b68d6d0a885c89b128b06d3dea267d77ed6daeebe4369` |
| LiveCodeBench problems (prompt variant and hidden tests) | HF `lighteval/code_generation_lite`, `release_v2/test-00000-of-00004.parquet` … `test-00003-of-00004.parquet` | `89e5fc5c2a8e748f50e95bc7235fab2372d49bfa` | `5939509129bd25dc49a16f276005ad006c810aba528bc153e9ec8155edd617e0`, `18e6ed06641db1bc355e4f2ff150fecbd86d5cf5fe634db974f3777911c25b66`, `24ec55689daad5da2abc86fe1fac4cd3893783d97db27ff26316b639cbe777a5`, `e0cdb54a3f965e86cbd909fb7f4c3fa3b304472878bb0c5805026045989bdbbe` |
| Derived tests for the 38 coding queries | `lcb-derived/89e5fc5c2a8e/manifest.json` in the cache; it lists each tests file with its own SHA-256 | built by the module | manifest `de299fbe80dde8e43f408c818b6ee8056602c39efb33c14a4a7809e939930803` |
| Harness (ported, not imported) | `github.com/RouteWorks/RouterArena` | `00cb25a470165b846a24bf3f2112d22dfc5758ce` (main, 2026-11-16) | per file below |
| Container for model-written code | `python:3.12-slim` | `sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1` (Python 3.12.15) | — |

Harness files ported, with their SHA-256 at the pinned commit:

| File | Ported as | SHA-256 |
|---|---|---|
| `scripts/process_datasets/prep_datasets.py` | `prompt_dataset`, `format_prompt`, `escape_format_braces`, LiveCodeBench matching | `8e65319de0cb653673b354819cac85918db2137618dab4d61a8e319cdd850595` |
| `config/eval_config/zero-shot/*.json` (41 files) | `PROMPT_TEMPLATES`, `LCB_*_TEMPLATE`, `CONFIG_METRICS` | checked file by file by `PinnedCloneTests` |
| `llm_evaluation/enhanced_extractor.py` | `extract_boxed_answer`, `extract_math_answer`, `extract_chess_move` | `cc9eb7e29f7d1d51a329893058af2ea7f17f3d140e3a14cabaa0402557a26221` |
| `llm_evaluation/metrics.py` | the `score_*` functions, `math_equal`, `meteor` | `eb149b514c071a32b5c0094b71c8a6317cd6b58f306c7217406a639bfa5eabce` |
| `llm_evaluation/metric_utils.py` | `_choice_answer_clean`, `_parse_digits`, `_numeric_equal`, `_str_to_pmatrix` | `a2dfe299cc41c5e80d6091eaefa2e3ba1ba0e0e1033b2d23adbd1e832c056c33` |
| `llm_evaluation/livecodebench_util.py` | `has_code`, `post_process_code`, `SANDBOX_HARNESS` | `335086be48376992671cf255ac07e820baf68f6e239be815fe3571c93c723dca` |
| `llm_evaluation/evaluate_models.py` | `eval_metric`, cost per call | `e3210177f9af246a12b72a8680e9417bece3d7577598b95dda04223e41722b07` |
| `llm_evaluation/eval_reasoning.py` | scorer choice | `a3f2b8f54a5cc5c885198acf25aa7c3cfcc0124c790e22e7d85def97a15eae6c` |
| `llm_evaluation/run.py` | failure handling, `cost_summary`, `arena_score` | `0c1f5e283b4b10fa17d6dc1eba35cee363e29284fabc47d62ce0b04a12599c9f` |
| `global_utils/router_usage.py` | `validate_router_usage`, router-call pricing | `7391deaba8666b8dd01b2241a6417b1aa560fd5f9b240bfc2bb3954cdf0a7a9c` |
| `model_cost/model_cost.json` | `ROUTERARENA_PRICES`, `ROUTERARENA_MAX_PRICE` | `fd9d27b76d74fcf2a7422a9e8cd1b08c2cdfd6b1cc0968887fa9262a7aedc21d` |
| `router_inference/generate_prediction_file.py` | `write_submission` (row shape) | `39a49df7a8c9c8499667cefeda604af2e1ecf6a8162e8ac7abd825e97e69bbed` |
| `llm_inference/run.py` | `write_submission` (`generated_result` shape) | `ff9da5580172e055d511ac44241696cac064e14bc34bd821ac301a76202f1b62` |
| `router_inference/check_config_prediction_files.py` | the rules `write_submission` enforces | `b60c3f9a3c976009585e84bb21d8b0330d04b80e7cc3fba7c5e0cb772059acf3` |

Why a port and not an import: `metrics.py` imports `jieba`, `fuzzywuzzy`, `rouge`, `regex`,
`sympy` and `latex2sympy2` when it loads, and this repository allows only the standard library
and `pyarrow`. Every function used for grading was ported line by line; the only part that could
not be ported is sympy's symbolic equivalence (see Math below).

`load(cache_dir)` downloads the two RouterArena parquet files (4 MB). On the first run it also
downloads the four LiveCodeBench shards (1.9 GB, about 30 s here), matches RouterArena's 385
coding questions to them exactly as `prep_datasets.py` does (exact question text, then a
100-character prefix), and writes the 38 needed problems' hidden tests (281 MB) under
`lcb-derived/`. LiveCodeBench stores those tests pickled; they are loaded with an unpickler that
refuses every global, so loading cannot run code. Later runs check the manifest and every tests
file against their digests and never touch the shards.

## Items

809 items, in the dataset's order. `Item.id` is `ra-` plus the first 16 hex characters of the
SHA-256 of RouterArena's global index (for example `ArcMMLU_98` → `ra-edfacd9d6f6d5c67`).
`Item.public` holds the global index, the source dataset name, the grading dataset and scorer,
RouterArena's category, domain and difficulty labels, and the messages digest; never question
or answer text.

| Scorer | Items | Datasets |
|---|---:|---|
| `mcq_accuracy` | 565 | MMLU-Pro 236, OpenTDB 97, PubMedQA 46, ArcMMLU 39, GeoBench 30, MedMCQA 29, Ethics 25, MusicTheoryBench 20, MMLU 19, MathQA 15, SocialiQA 7, SuperGLUE-CausalReasoning 2 |
| `meteor_score` | 70 | NarrativeQA 39, WMT19 (eight language pairs) 31 |
| `exact_match` | 62 | QANTA 58, GeoGraphyData 4 |
| `code_accuracy` | 38 | LiveCodeBench release_v2 (24 function problems, 14 stdin problems) |
| `superglue_exact_match` | 32 | SuperGLUE Wic 10, QA 7, Entailment 6, RC 6, Wsc 3 |
| `math_metric` | 24 | AsDiv 7, FinQA 7, MATH 5, AIME 3, GSM8K 2 |
| `chess_accuracy` | 13 | ChessInstruct (6 move questions, 7 multiple-choice) |
| `superglue_clozetest` | 5 | SuperGLUE ClozeTest |

By grading method: 771 items are graded by exact rules in process (all but code), 38 by running
the answer's code in the container, 0 by a judge. By difficulty label: 389 easy, 233 medium,
187 hard.

## Prompt

Each item is one user message and no system prompt: RouterArena's `prompt_formatted`, which its
inference pipeline sends as `messages=[{"role": "user", "content": prompt}]` on every provider
path except xAI's (which adds a Grok system prompt). The text is built per dataset from
`config/eval_config/zero-shot/<dataset>.json`, keeping RouterArena's quirks:

- `Context: None` when the context is empty (but SuperGLUE-Wic puts the raw context in quotes);
- options as `A. text\n` lines, so the options block ends with an extra blank line;
- braces in the question are doubled (`{1, 2}` becomes `{{1, 2}}`): `prep_datasets.py`
  escapes the values it then passes to `str.format`, which inserts them literally;
- the math templates ask for `\boxed{{}}` and give `\boxed{{42}}` as the example, with doubled
  braces, because the JSON configs double-escape them;
- SuperGLUE-RC's template has a `Provided Answer:` slot that is always empty;
- ChessInstruct rows with options use the multiple-choice template; the others the move template;
- LiveCodeBench rows use the stdin template when LiveCodeBench's own public tests are stdin
  tests, and the function template otherwise;
- a prompt over 10,000 characters is cut to its first 5,000 and last 5,000 characters joined by
  `...` (2 items).

Check: the module's 809 prompts are byte-identical to the prompts in all 37 prediction files
published in RouterArena's repository at the pin (29,926 of 29,933 rows; the 7 others are in one
older submission, `chayan`, built from an earlier dataset).

## Settings for the runner

- `max_output_tokens` = 16,384. RouterArena's own pipeline cannot call these three models (none
  is in `llm_inference/model_inference.py`'s provider map); its OpenRouter and other
  OpenAI-compatible paths set no output bound, while its Anthropic, DeepSeek-direct, Mistral,
  Azure, Perplexity and MiniMax paths use 2,048 and SiliconFlow 1,024. So this uses Panda's
  ceiling. A long reasoning answer can hit it: in the Corealis Router submission, which routes
  the same three models through OpenRouter, GLM-5.3 at high effort averaged 47,667 output
  tokens. A cut answer is graded as returned.
- Temperature and reasoning settings: RouterArena sets none on that path; leave Panda's.
- One call per query, no repeats. No judge calls.
- Record for every query: the model `auto` picked, the answer (or `None` if the call failed),
  input and output tokens (reasoning included in output), and the router's embedding call
  (model, input tokens, output tokens 0). These feed `grade`, `cost_per_1k` and
  `write_submission`.

## Grading

`grade(item, answer, judged)` returns `score` (0 or 1; a fraction for METEOR), `correct`
(`None` for METEOR), `failed_answer`, the scorer and dataset, and per-scorer flags: the parsed
letter for multiple choice, `math_check`, `code_check`, `timed_out`, `third_party_import`. It
never returns the question, the reference or extracted answer text.

The scorer comes from the global index, as `ModelEvaluator.determine_dataset_from_global_index`
does, not from the prompt template: ArcMMLU uses MMLU-Pro's config, every Ethics subset uses
`Ethics_commonsense`'s, and every ChessInstruct item, multiple-choice ones included, uses
`chess_accuracy` (which reads `\boxed{B}` as `b` and an index `1` as `b`).

| Scorer | Rule (RouterArena's, ported) |
|---|---|
| `mcq_accuracy` | Reference letter, or a 0-based index mapped to a letter. Answer: the first `\boxed{...}` (after dropping `\text{`), else "answer: X", "option X", "the answer is X" or the last standalone capital letter. Exact, case-sensitive match. |
| `math_metric` | Answer: the first balanced `\boxed{...}`, else answer patterns, else the last number; anything after the last `=`. Equal if the strings match, a choice letter matches, both parse as numbers within 1e-4 relative (also against reference ×100 and ÷100), brackets or tuples match part by part, or sympy finds them equal. |
| `exact_match` | First `\boxed{...}` (or the fallbacks), underscores to spaces, lower case, collapsed spaces; exact match. |
| `chess_accuracy` | Boxed move or the last `a1b2`-style move; lower case, separators removed. |
| `meteor_score` | RouterArena's own simplified METEOR on the boxed text: unigram F1 times one minus a fragmentation penalty, Chinese split by character; no stemming or synonyms. A word-for-word copy of a three-word reference scores 0.944, not 1. |
| `superglue_exact_match` | Boxed label mapped (A→0.0/Yes, B→1.0/No, 1→1.0…), then exact match; `yes` does not match `Yes`. |
| `superglue_clozetest` | Reference is the option text; the boxed text must equal it. Without a box the answer's first character is compared, which never matches a multi-letter option. |
| `code_accuracy` | See below. |

A missing answer (`None`) scores 0 and stays in the denominator, as RouterArena's
`router_metrics` counts a query without a valid generation. A scorer exception scores 0, as
`ModelEvaluator._evaluate_single_entry` does.

### Math

sympy's `symbolic_equal` cannot be ported to the standard library. `math_equal` here follows
every other branch of RouterArena's function and returns `None` where its verdict would come from
sympy. Those items score 0 and carry `math_check: "unresolved"`; `score()` reports
`n_math_unresolved`. On RouterArena's own re-graded rows (below), 101 of 576 math answers were
unresolved and only 1 of them was correct there, so the local score is low by at most that
count, usually far less.

### Code

1. The host takes the answer's last fenced block and post-processes it as `code_accuracy` does.
   No block: score 0, `code_check: "no_code"`, nothing runs.
2. The code, the problem's hidden tests and the harness go through stdin to
   `docker run --rm -i --network none --read-only --memory 512m --cpus 1 --pids-limit 128
   --tmpfs /tmp --cap-drop ALL --security-opt no-new-privileges --user 65534:65534 --pull never`
   on the pinned image. Nothing is mounted and nothing is written on the host.
3. Inside, `livecodebench_util.py`'s test functions run verbatim: function problems call the
   first function name in the code with parsed arguments, stdin problems `exec` the program with
   the test input as stdin and compare stripped output exactly. As in RouterArena, a forked child
   runs the tests in order after `reliability_guard` (no `os.system`, `fork`, `subprocess`, `exit`)
   and stops at the first failure; the parent waits (6 + 1) × tests + 5 seconds. Score 1 only if
   every test passed. The child cannot write to the container's stdout, and the verdict line
   carries a random nonce.
4. A container that gives no verdict scores 0 with `code_check: "sandbox_error"`; without Docker
   or the image, `code_check: "unavailable"`. Both are counted in `score()`
   (`n_code_sandbox_errors`, `n_code_ungraded`), never counted as passes. Set
   `PANDA_BENCH_NO_DOCKER=1` to skip containers entirely.

### Agreement with RouterArena's own grades

RouterArena re-graded 24 published prediction files with its current evaluator on 2026-10-29
(commit `2be9c9d`). On their `sub_10` rows with a successful generation:

| Scorer | Rows | Same score here |
|---|---:|---:|
| `mcq_accuracy` | 13,553 | 13,553 |
| `meteor_score` | 1,680 | 1,680 (exact floats) |
| `exact_match` | 1,486 | 1,486 |
| `superglue_exact_match` | 768 | 768 |
| `math_metric` | 576 | 575 (the other is one of the 101 unresolved) |
| `chess_accuracy` | 311 | 311 |
| `superglue_clozetest` | 120 | 120 |
| `code_accuracy` (a sample: 4 answers per problem, both outcomes where available) | 128 | 125 |

The code sample covers all 38 problems: 76 answers that passed on RouterArena and 52 that failed.
6 had no code block (0 on both sides) and 122 ran in the container, 119 with RouterArena's
result. Of the other 3, one brute-force program ran past the deadline here but passed on GitHub's
runner (machine speed). Two passed here and failed there: a recursion about 1,000 deep through
`lru_cache`, and a deep DFS under a raised recursion limit. Both fit Python 3.11's stricter
recursion accounting (RouterArena's CI uses 3.11, the container 3.12; not checked on 3.11 here).
The largest test sets (95 MB and 84 MB of tests) ran within the 512 MB limit, and no container
failed.

## Score

- `score` = 100 × mean of the 809 item scores, RouterArena's accuracy on the board's
  percentage scale. METEOR items add fractions; failed answers, sandbox errors, ungraded code and
  unresolved math add 0.
- Parts: `accuracy` (0–1), `n`, `points`, `n_failed_answers`, `n_failed_judge_calls` (always 0),
  `n_code_sandbox_errors`, `n_code_ungraded`, `n_code_failed_third_party_import`,
  `n_math_unresolved`, and `by_metric` / `by_dataset` tables.
- Example: scores 1, 0, 0.5, 0 (failed) over four items give `score` = 37.5.

`cost_per_1k(rows)` (and `cost_summary` with its parts) follows `llm_evaluation/run.py`:

```
cost(row)   = answer: in × p_in(model) + out × p_out(model) + max(0, total − in − out) × p_out(model)
            + Σ router_usage calls: in × p_in(call model) + out × p_out(call model)      (per 1M tokens)
cost_per_1k = 1000 × Σ cost(row over rows with a valid generation) / (number of rows)
```

A valid generation succeeded with a non-empty answer and output tokens > 0; other rows add no
cost (not even their router calls) but stay in the denominator. A model without a price is
charged the table's highest prices ($15 in, $75 out per 1M, Claude Opus 4.1's).

`arena_score(cost_per_1k, accuracy)` is RouterArena's `compute_arena_score` on the board's 0–100
scale: with C = (log2 200 − log2 c) / (log2 200 − log2 0.0044), c clamped to [0.0044, 200],
S = 100 × 1.1 × A × C / (0.1 × A + C). It reproduces the README's Divyam (77.19), GPT-5 (64.32)
and Azure (70.37) rows from their accuracy and cost.

## Prices

The three answer models are already in RouterArena's price table; only the router's embedding
model is not.

| Model | RouterArena name | RouterArena table at the pin ($/1M in / out) | Panda list price |
|---|---|---|---|
| DeepSeek V4.1 Flash | `deepseek/deepseek-v4.1-flash` | 0.21 / 0.84 | 0.30 / 1.20 |
| GLM-5.3 | `z-ai/glm-5.3` | 1.40 / 4.40 | 1.40 / 4.40 |
| GLM-5.3 Flash | `z-ai/glm-5.3-flash` | 0.15 / 0.50 | 0.15 / 0.50 |
| EmbeddingGemma 2 (router call) | `google/embeddinggemma-2` (proposed) | not listed: charged 15 / 75 | 0.0017 / 0 |

- `cost_per_1k(rows)` defaults to RouterArena's table plus a proposed EmbeddingGemma 2 profile at
  Panda's list price, which is what the board would use once a PR adds that profile.
  `cost_per_1k(rows, PANDA_LIST_PRICES)` gives the same figure at Panda's prices.
- RouterArena priced DeepSeek V4.1 Flash at $0.30/$1.20 in its 2026-10-31 snapshot (standard
  price, discounts removed) and at $0.21/$0.84 since 2026-11-16 ("as charged, discounts
  included", the median of 29 OpenRouter hosts). A PR cannot change a listed model's price; the
  board uses whatever `main` holds when it grades.
- RouterArena charges a router's own model calls ("a routing model, a classifier, voters, a
  critic…"). Panda's embedding call decides the route, so it belongs in `router_usage`: model
  `google/embeddinggemma-2`, input tokens of the embedded text (the prefix
  `task: classification | query: ` plus the first 2,000 characters of the message), output 0.
  At $0.0017 per 1M tokens it adds well under $0.001 per 1,000 queries; without a price profile
  RouterArena would charge it $15 per 1M, about $5 per 1,000 queries.

## Submission files (prepared, not submitted)

`write_submission(out_dir, rows)` writes `router_inference/config/<name>.json` and
`router_inference/predictions/<name>.json` under `out_dir`, in the shapes RouterArena's
`generate_prediction_file.py` and `llm_inference/run.py` produce and
`check_config_prediction_files.py --check-generated-result` accepts. It refuses anything but
809 unique rows (`expected_rows` overrides that), unknown answer models and malformed
`router_usage`. Nothing is sent anywhere; whether to open a pull request is the owner's decision.

```json
{"pipeline_params": {"router_name": "panda-auto", "router_cls_name": "external_prediction_file",
  "models": ["deepseek/deepseek-v4.1-flash", "z-ai/glm-5.3", "z-ai/glm-5.3-flash"],
  "description": "…"}}
```

```json
{"global index": "ArcMMLU_98", "prompt": "<the exact prompt>", "prediction": "deepseek/deepseek-v4.1-flash",
 "generated_result": {"generated_answer": "…", "success": true,
   "token_usage": {"input_tokens": 120, "output_tokens": 900, "total_tokens": 1020},
   "provider": "panda", "error": null},
 "cost": null, "accuracy": null, "for_optimality": false,
 "router_usage": [{"model": "google/embeddinggemma-2", "input_tokens": 41, "output_tokens": 0,
                   "purpose": "routing embedding"}]}
```

Check: RouterArena's own validator (`check_config_prediction_files.py <name> sub_10
--check-generated-result`, standard library only, run from a scratch copy of the pinned files)
passes a synthetic 809-row submission written by this function once `"z-ai/glm-5.3"` is added to
`universal_names`, and fails it without that line. It warns that `google/embeddinggemma-2` has no
price profile, as expected.

A row without an answer, with an empty answer or without output tokens is written with
`"success": false` and an `error`, as RouterArena's validator requires; it scores 0. The default
`description` states how the router works and what it was built from; the owner should check it
before any PR. A pull request would also need:

- `"z-ai/glm-5.3"` added to `universal_names` in `universal_model_names.py` (only its
  `@low`/`@high` spellings are listed, and the validator rejects a config model it cannot find);
- a `google/embeddinggemma-2` profile in `model_cost/model_profiles.yaml` with a link to the
  published price, and the regenerated `model_cost/model_cost.json`;
- the robustness file (below): the PR workflow requires exactly two prediction files;
- no optimality entries in a `sub_10` file: the workflow picks the split by entry count, and only
  809 entries mean `sub_10`;
- only Panda's own answers: RouterArena rejects rows copied from other submissions and any router
  trained, fitted or tuned on RouterArena data, including routing on its template text.

The board ranks the full split; a `sub_10` PR would get RouterArena's bot report, not a
leaderboard row.

## Robustness (not built)

RouterArena's robustness score is 1 − the share of queries whose pick changes when the prompt is
perturbed. It needs `router_inference/predictions/<name>-robustness.json`: 420 entries of
`{"global index", "prompt", "prediction"}` with no answers. The prompt is the robustness split's
`Question` field as is: a rewritten, already-formatted prompt (it shares no template prefix with
the original; median character similarity 0.67). All 420 global indices are in `sub_10`, so a
`sub_10` entry and a full entry are scored on the same 420 pairs. For Panda this means 420 routing
decisions by the same `auto` router: 420 embedding calls if Panda can report a pick without
answering, otherwise 420 full calls whose answers are discarded. Router-call cost is not charged
for robustness.

## A full 8,400-query entry

- 8,400 routed answers with token usage and `router_usage`, about 10.4 times the `sub_10` run's
  calls and cost, plus the 420 robustness picks.
- The 385 coding problems: the module pins tests for the 38 `sub_10` ones; a full run needs the
  derivation widened to all 385 (same shards, new manifest digest) and about ten times the
  container time.
- The full split's prompts come from the same `format_prompt`; its parquet is already pinned.
- RouterArena's CI produces the official number. A local full score is only an estimate, with
  the deviations below.
- The `sub_10` run must not be used to tune the router before a full entry: RouterArena is
  evaluation-only.

## Deviations from RouterArena's protocol

1. **Python 3.12 instead of 3.11 for code.** RouterArena's CI runs Python 3.11; the newest
   3.11 image is not on this machine and downloads are limited to the pinned data, so the
   container uses Python 3.12.15. Recursion-depth accounting and the standard library differ
   slightly between the two.
2. **Container limits.** RouterArena forks its evaluator process, so a solution may import any
   package in its environment (numpy, networkx, pandas, sympy, mpmath, sortedcontainers, regex,
   torch) and gets about 2 GB of address space on top of the evaluator's. Here the container has
   only the standard library, 512 MB, one CPU and no network. A failed solution that imports one of
   those packages is flagged `third_party_import` and counted in
   `n_code_failed_third_party_import`.
3. **Machine speed.** Deadlines are RouterArena's, but this machine is not GitHub's runner;
   a solution near the deadline can flip either way.
4. **sympy equivalence.** Not ported; reported as `n_math_unresolved` and scored 0.
5. **Harness plumbing.** A forked child and a pipe stand in for `multiprocessing.Manager`; the
   order of tests, the early stop, the deadline and the all-pass rule are unchanged.
6. **Output bound.** 16,384 tokens, as explained above.
7. **Prices.** The default cost table is RouterArena's pinned table plus a proposed
   EmbeddingGemma 2 profile at Panda's list price, which RouterArena does not yet list.
8. **Split.** The board ranks the full split; `sub_10` is a subset that RouterArena calls easier.

## License and terms

- RouterArena's code is Apache-2.0 (`LICENSE` at the pinned commit); this port keeps its
  attribution in the module.
- The dataset card declares no license. Its queries come from about 40 public datasets (MMLU-Pro,
  OpenTDB, QANTA, PubMedQA, LiveCodeBench, NarrativeQA, WMT19 and others) under their own terms.
  RouterArena calls it "an evaluation-only dataset" and rejects any submission that trains, fits
  or tunes a router component on it. This repository treats it accordingly: releases carry ids,
  labels and digests only, never question or answer text.
- `lighteval/code_generation_lite` (LiveCodeBench) is marked `cc` on its card.

## Published rows this score sits next to

RouterArena README leaderboard at commit `00cb25a` (2026-11-16), read 2026-11-18:
<https://github.com/RouteWorks/RouterArena/blob/00cb25a470165b846a24bf3f2112d22dfc5758ce/README.md>
(also <https://routeworks.github.io/leaderboard>). All rows are on the full split.

| Router | Acc-Cost Arena | Accuracy | Cost / 1K queries |
|---|---:|---:|---:|
| Divyam (#1) | 77.19 | 80.83% | $0.66 |
| BizRouter (#2) | 77.08 | 81.66% | $1.00 |
| Sqwish Router (#3) | 76.34 | 79.88% | $0.69 |
| Azure-Model-Router | 70.37 | 72.93% | $0.75 |
| GPT-5 (single model) | 64.32 | 73.96% | $10.02 |
| NotDiamond | 57.29 | 60.83% | $4.10 |

For a like-for-like `sub_10` reference, RouterArena's per-row grades in the published prediction
files give these `sub_10` accuracies (computed here, not published by RouterArena): Divyam 83.54%,
BizRouter 83.99%, Azure-Model-Router 75.78%. The Corealis Router submission routes the same three
models (with reasoning-effort variants, through OpenRouter); its file gives 77.28% on `sub_10`
and 75.88% on the full split, at $1.07 per 1,000 queries on the full split. It is not on the README
leaderboard at the pin.
