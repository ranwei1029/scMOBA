# scMOBA-Chat package

Code release candidate for inclusion in https://github.com/Sherryweiran/scMOBA .
Model assets: https://huggingface.co/ranweisherry/scMOBA/tree/main/chat .
This source tree includes the local deployment web app, CLI, preprocessing
code and tests. See the [M1 inference notebook](../tutorials/scmoba_chat_inference.ipynb).
This is a release candidate pending author approval of licensing.

On Linux with Python 3.11–3.13 and an NVIDIA GPU:

```bash
bash scripts/setup.sh
source .venv/bin/activate
scmoba-chat doctor --config /path/to/scMOBA/chat/config.json
scmoba-chat serve --config /path/to/scMOBA/chat/config.json --work-dir user_data --port 7860
```

Download the entire chat directory first. Open http://127.0.0.1:7860 locally,
or use SSH port forwarding to your GPU host. Upload human RNA h5ad, explicitly
select counts or already normalized log1p input and the correct gene identifiers.
Counts are normalized to 10,000 then log1p; gene ranking uses training medians.
Template tasks prepend training-matched instructions; free questions retain
same-cell conversation history. No external LLM API or uploaded-data training.

Local trusted single-user use only: no public authentication or hostile-file
sandbox. Raw uploads and outputs remain in user_data until manually removed.
Start with 24GB GPU memory; requirements vary with context length. Research only.

Authors must confirm code/weight licensing before publishing this candidate.
Upstream model licenses are separate; no blanket project license is assigned here.

## Download model assets

The model repository is private until release. Authenticate locally with a
Hugging Face account granted access; never paste a token into a script or Git.

```bash
hf auth login
hf download ranweisherry/scMOBA --include 'chat/*' --local-dir ./models
scmoba-chat doctor --config ./models/chat/config.json
```

## Synthetic installation test

```bash
python scripts/make_demo_data.py --resources models/chat/resources.json --output demo.h5ad
scmoba-chat prepare --input demo.h5ad --output demo-prepared --resources models/chat/resources.json --input-scale counts --gene-type ensembl
scmoba-chat infer --config models/chat/config.json --dataset demo-prepared --cell-id synthetic_1 --task subclass --output demo-answer.json
python scripts/reproduce_http.py --config models/chat/config.json --input demo.h5ad --input-scale counts --gene-type ensembl --output http-test
```

Synthetic answers are installation checks, not biological examples or accuracy results.
The HTTP test starts and stops its own localhost server and checks upload,
cell selection, free QA, template QA, history and CLI/API token equivalence.

## Input requirements

Use unique cell identifiers and human Ensembl IDs (optionally version-suffixed)
or exact gene symbols. Specify `--gene-column` for a var column, otherwise var_names
are used. Set `--layer counts` when counts are in that layer rather than X.
Do not submit scaled/z-scored data or log2 values as log1p. Supply broad gene
coverage, not just marker genes. Counts duplicates are summed; log1p mapped
duplicates are rejected. Existing preparation output directories are not overwritten.
Uploaded obs annotations are not automatically included in questions.
