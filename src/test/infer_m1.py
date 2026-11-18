"""Standalone original scMOBA inference on an original-tokenized HF Dataset."""
import argparse
import csv
from pathlib import Path
from types import SimpleNamespace

import torch
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model
from transformers import PreTrainedTokenizerFast

from src.model.mllm_gene import scLaGene


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('dataset', 'checkpoint', 'tokenizer', 'geneformer', 'llm', 'output'):
        p.add_argument('--' + key, required=True)
    p.add_argument('--limit', type=int, default=16)
    p.add_argument('--label-column', default='Class')
    p.add_argument('--question', default='Can you predict the cell type based on this gene expression data? Please provide your answer.')
    a = p.parse_args()
    if a.limit < 1 or Path(a.output).exists():
        p.error('Use a positive limit and a new output file.')
    if not torch.cuda.is_available():
        p.error('A visible CUDA GPU is required.')
    torch.manual_seed(42)
    ds = load_from_disk(a.dataset)
    if not {'input_ids', a.label_column}.issubset(ds.column_names) or not len(ds):
        p.error('Expected a nonempty original-tokenized dataset with input_ids and the requested label column.')
    tok_path = Path(a.tokenizer) / 'tokenizer.json'
    # The serialized backend already contains the vocabulary and added tokens.
    # Avoid interpreting version-specific tokenizer_config class names.
    tok = PreTrainedTokenizerFast(tokenizer_file=str(tok_path))
    patch_token = tok.convert_tokens_to_ids('<im_patch>')
    args = SimpleNamespace(
        bert_path=a.geneformer, llm_path=a.llm, bert_frozen=True, llm_frozen=True,
        bert_hidden_size=256, GCN_embedding_dim=2048, llm_hidden_size=2048,
        dropout=0.1, num_querys=256, encoder_num_heads=4, vocab_size=len(tok),
    )
    inputs = tok('<im_patch>' * args.num_querys + ' ' + a.question,
                 return_tensors='pt')['input_ids']
    if patch_token is None or int((inputs == patch_token).sum()) != args.num_querys:
        p.error('Tokenizer did not produce exactly 256 multimodal placeholders.')
    model = scLaGene(args)
    model.llm = get_peft_model(model.llm, LoraConfig(
        r=16, lora_alpha=32, target_modules=['q_proj', 'v_proj'],
        lora_dropout=0.1, bias='none', task_type='CAUSAL_LM'))
    model.load_state_dict(torch.load(a.checkpoint, map_location='cpu', weights_only=True), strict=True)
    model.requires_grad_(False).eval().to('cuda')
    inputs = inputs.to('cuda')
    print('Strict checkpoint load passed', flush=True)
    with Path(a.output).open('x', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['row_index', 'reference_label', 'question', 'answer'])
        for i in range(min(a.limit, len(ds))):
            row = ds[i]
            expr = torch.tensor([row['input_ids']], dtype=torch.long, device='cuda')
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                text = tok.decode(model.generate(expr, inputs)[0], skip_special_tokens=True).strip()
            if not text:
                raise RuntimeError(f'Empty generation at row {i}')
            writer.writerow([i, row[a.label_column], a.question, text])
        print(f'Wrote {min(a.limit, len(ds))} predictions', flush=True)


if __name__ == '__main__':
    main()
