# Data sources and artifact scope

The MIT license covers this kit's code and documentation. Upstream dataset rights remain separate. The archived model outputs are provided for benchmark inspection; this does not assert ownership of any underlying material quoted by a model.

## WildBench

Source: [allenai/WildBench](https://huggingface.co/datasets/allenai/WildBench). The [pinned dataset card](https://huggingface.co/datasets/allenai/WildBench/raw/26c49eb39d7d5ce2099b0bbafed5a88dcce954ec/README.md) specifies CC BY, links [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), and describes research/educational use under Ai2 Responsible Use Guidelines. Retain attribution to the WildBench authors and follow upstream use terms. The separate upstream code license does not replace dataset terms.

The pinned revision is `26c49eb39d7d5ce2099b0bbafed5a88dcce954ec`. Both files at this revision match the original downloaded file hashes. The originally requested historical revision was not recorded; byte identity is the evidence for this pin.

| Config / split | Parquet SHA-256 | Bytes |
| --- | --- | ---: |
| v2 / test | e1fbdc5d46deca34e77f92991c23b0476860e26adf2bf28379a3bb3bf00d3ed6 | 3695374 |
| v2-hard / test | 44e2a0d20f791c51d68083c8acc8916698d4bb8d460d1dbc393b54030dbea40a | 402410 |

Reconstruction uses `session_id` (prefix `wb-`), single-turn `conversation_input[0].content`, and `list(checklist or [])`, selecting the exact source config for each released prompt. The manifests include prompt/checklist hashes but no prompt bodies. Downloaded text stays in ignored `data/`.

## Archived releases

`answers.jsonl` contains generated outputs, without source prompt fields, credentials, service IDs or raw error bodies. A whole-prompt normalized substring check and conservative private-path/credential-pattern scan were applied. Any withheld output is identified in `omissions.json`; checks use numeric grades and remain possible even when text is withheld. This screening is a limited heuristic, not a claim that outputs cannot contain quotations or personal data. The local owner requested inclusion for transparency; publication still requires their review.

`usage.jsonl` records request-level costs and the calls that contribute to them. Cached member calls appear in multiple configurations; summing all rows is not actual experiment spend. Some combiner input token counts were not retained and are null rather than invented. Error outcomes can have unreported provider charges. Per-attempt retries and invoices are not reconstructed.

Stage 1 data, competition problems and code tests are not part of these three releases.
