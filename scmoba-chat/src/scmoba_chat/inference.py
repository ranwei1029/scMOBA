import json
import threading
from pathlib import Path
import hashlib


class Engine:
    def __init__(self, config_path, device='cuda:0'):
        import torch
        # This is text+gene inference; avoid unrelated broken audio/vision extras
        # in shared scientific Python environments.
        import transformers.utils.import_utils as optional
        optional._torchvision_available=False
        optional._librosa_available=False
        from transformers import AutoTokenizer
        from peft import PeftModel
        from ._model.model import ScLaGeneQwen
        config_path=Path(config_path).resolve()
        self.config=json.loads(config_path.read_text())
        def path(k):
            p=Path(self.config[k]); return str(p if p.is_absolute() else config_path.parent/p)
        for key in ('geneformer','qwen','checkpoint','resources'):
            if not Path(path(key)).exists(): raise FileNotFoundError(f'Missing {key}: {path(key)}')
        self.resources_path=path('resources'); self.device=device; self.lock=threading.Lock()
        if not torch.cuda.is_available() and device.startswith('cuda'): raise RuntimeError('CUDA GPU required for this configuration')
        dtype=torch.bfloat16 if device.startswith('cuda') else torch.float32
        self.tokenizer=AutoTokenizer.from_pretrained(path('qwen'),local_files_only=True,padding_side='left')
        if self.tokenizer.pad_token_id is None:self.tokenizer.pad_token=self.tokenizer.eos_token
        self.model=ScLaGeneQwen(geneformer_path=path('geneformer'),qwen_path=path('qwen'),torch_dtype=dtype)
        checkpoint=Path(path('checkpoint'))
        state=torch.load(checkpoint/'projector_trainable.bin',map_location='cpu',weights_only=True)
        prefixes=('gene_encoder.','gcn.','query_tokens','projector.')
        expected={k for k in self.model.state_dict() if k.startswith(prefixes)}
        selected={k:v for k,v in state.items() if k.startswith(prefixes)}
        if expected!=set(selected):raise RuntimeError(f'Incomplete/incompatible multimodal checkpoint: missing={sorted(expected-set(selected))}, extra={sorted(set(selected)-expected)}')
        self.model.load_state_dict(selected,strict=False)
        self.model.llm=PeftModel.from_pretrained(self.model.llm,checkpoint/'qwen_lora',local_files_only=True)
        self.model.requires_grad_(False);self.model.eval().to(device)
        self.valid_tokens=set(json.loads(Path(self.resources_path).read_text())['tokens'].values())
        self.checkpoint=str(checkpoint)

    def chat(self, input_ids, question=None, history=None, max_new_tokens=384, task='free'):
        import torch
        from .prompts import task_question
        question, instructed_question=task_question(task,question)
        if not isinstance(question,str) or not question.strip() or len(question)>16000: raise ValueError('Question must contain 1..16000 characters')
        if not input_ids or len(input_ids)>2048 or any(type(t)!=int or t not in self.valid_tokens for t in input_ids):raise ValueError('Invalid Geneformer input IDs')
        if not 1<=max_new_tokens<=1024:raise ValueError('max_new_tokens must be 1..1024')
        history=history or []
        if task!='free' and history:raise ValueError('Template prediction requires a fresh chat; use free QA for follow-up')
        if len(history)%2 or len(history)>20:raise ValueError('At most 10 complete previous turns')
        for i,m in enumerate(history):
            if m.get('role')!=('user' if i%2==0 else 'assistant') or not isinstance(m.get('content'),str):raise ValueError('Invalid history')
        content=('[Profile type: single cell]\n'+instructed_question) if not history else instructed_question
        messages=history+[{'role':'user','content':content}]
        enc=self.tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_tensors='pt',return_dict=True)
        if enc['input_ids'].shape[1]+64+max_new_tokens>8192:raise ValueError('Conversation too long; start a new chat')
        gene=torch.tensor([input_ids],dtype=torch.long,device=self.device)
        with self.lock,torch.inference_mode():
            output=self.model.generate(input_ids=enc['input_ids'].to(self.device),attention_mask=enc['attention_mask'].to(self.device),gene_input_ids=gene,gene_attention_mask=torch.ones_like(gene),has_omics=torch.tensor([True],device=self.device),max_new_tokens=max_new_tokens,do_sample=False,pad_token_id=self.tokenizer.pad_token_id,eos_token_id=self.tokenizer.eos_token_id)
        answer=self.tokenizer.decode(output[0],skip_special_tokens=True).strip()
        return {'question':question,'task':task,'model_question':content,'answer':answer,'history':messages+[{'role':'assistant','content':answer}],
                'checkpoint':self.checkpoint,'input_token_sha256':hashlib.sha256(json.dumps(input_ids).encode()).hexdigest(),
                'generated_tokens':int(output.shape[-1]),'hit_token_limit':int(output.shape[-1])>=max_new_tokens}
