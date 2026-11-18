# Install and run original scMOBA inference

From the repository root on Linux:

```bash
bash script/setup_inference.sh
source script/.venv/bin/activate
```

This creates a fresh Python environment using public package indexes. It
installs CUDA 12.8 PyTorch wheels and inference dependencies, and does not
need the legacy Conda training export. GPU inference still needs a compatible
NVIDIA driver and visible CUDA device.

Use the [M1 notebook](../tutorials/scmoba_inference.ipynb) or run
`python -m src.test.infer_m1 --help` from the root. Supply the matching
original checkpoint, tokenizer, Geneformer and Llama paths, together with
a local original-tokenized Hugging Face Dataset containing `input_ids` and
`Class`. Do not substitute scMOBA-Chat tokens, tokenizer or weights.

This example predicts broad cell class. It does not reproduce the separate
M1 Subclass benchmark and does not imply a validated checkpoint has already
been released on the Hub. Real M1 files are not part of this repository.
