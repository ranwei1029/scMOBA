"""Qwen-based scLaGene model with mixed omics/text routing."""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForCausalLM

from .gcn import GCN
from .attention import CrossMultiHeadAttention


def _text_config(config):
    return getattr(config, "text_config", config)


def _safe_affinity_adjacency(hidden_states: torch.Tensor) -> torch.Tensor:
    """Device-safe equivalent of the old aff_to_adj_batch implementation."""
    normalized = F.normalize(hidden_states.detach(), dim=-1)
    similarity = torch.bmm(normalized, normalized.transpose(1, 2))
    nodes = similarity.shape[-1]
    identity = torch.eye(nodes, device=similarity.device, dtype=similarity.dtype)
    identity = identity.unsqueeze(0).expand(similarity.shape[0], -1, -1)
    adjacency = similarity - identity
    column_sum = adjacency.sum(dim=1, keepdim=True)
    adjacency = adjacency / column_sum.clamp_min(1e-8)
    return adjacency + identity


class ScLaGeneQwen(nn.Module):
    """Geneformer -> GCN -> cross-attention -> Qwen, with a text-only bypass."""

    def __init__(
        self,
        geneformer_path: str,
        qwen_path: str,
        gcn_hidden_size: int = 2048,
        num_query_tokens: int = 64,
        projector_num_heads: int = 4,
        projector_dropout: float = 0.1,
        torch_dtype: torch.dtype = torch.bfloat16,
        attn_implementation: Optional[str] = None,
    ):
        super().__init__()
        self.gene_encoder = AutoModel.from_pretrained(
            geneformer_path, add_pooling_layer=False
        )

        load_kwargs = {"dtype": torch_dtype}
        if attn_implementation:
            load_kwargs["attn_implementation"] = attn_implementation
        # The omics modality is supplied by Geneformer/GCN/projector.  Load only
        # Qwen's causal language model; no native image encoder is needed.
        self.llm = AutoModelForCausalLM.from_pretrained(qwen_path, **load_kwargs)

        text_config = _text_config(self.llm.config)
        self.llm_hidden_size = int(text_config.hidden_size)
        self.vocab_size = int(text_config.vocab_size)
        self.num_query_tokens = int(num_query_tokens)

        gene_hidden_size = int(self.gene_encoder.config.hidden_size)
        self.gcn = GCN(
            gene_hidden_size,
            int(gcn_hidden_size),
            self.llm_hidden_size,
            float(projector_dropout),
        )
        self.query_tokens = nn.Parameter(
            torch.empty(self.num_query_tokens, self.llm_hidden_size)
        )
        nn.init.normal_(self.query_tokens, mean=0.0, std=0.02)
        self.projector = CrossMultiHeadAttention(
            self.llm_hidden_size,
            int(projector_num_heads),
            att_dropout=float(projector_dropout),
        )
        self.configure_training_stage(1)

    @property
    def pad_token_id(self) -> Optional[int]:
        return getattr(_text_config(self.llm.config), "pad_token_id", None)

    @property
    def eos_token_id(self) -> Optional[int]:
        return getattr(_text_config(self.llm.config), "eos_token_id", None)

    def projector_parameters(self):
        yield from self.gcn.parameters()
        yield self.query_tokens
        yield from self.projector.parameters()

    def configure_training_stage(self, stage: int) -> None:
        """Apply the requested two-stage freezing protocol."""
        if stage not in (1, 2):
            raise ValueError(f"stage must be 1 or 2, got {stage}")
        self.gene_encoder.requires_grad_(False)
        self.gcn.requires_grad_(True)
        self.query_tokens.requires_grad_(True)
        self.projector.requires_grad_(True)
        self.llm.requires_grad_(False)
        if stage == 2:
            for name, parameter in self.llm.named_parameters():
                if "lora_" in name or "modules_to_save" in name:
                    parameter.requires_grad = True

    def train(self, mode: bool = True):
        super().train(mode)
        # Geneformer remains frozen in both stages; keep its dropout disabled.
        self.gene_encoder.eval()
        return self

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Delegate the Trainer API to the wrapped Qwen/PEFT model."""
        kwargs = gradient_checkpointing_kwargs or {}
        self.llm.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=kwargs
        )

    def gradient_checkpointing_disable(self):
        """Delegate the Trainer API to the wrapped Qwen/PEFT model."""
        self.llm.gradient_checkpointing_disable()

    def _encode_omics(
        self,
        gene_input_ids: torch.Tensor,
        gene_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        vocab_size = int(self.gene_encoder.get_input_embeddings().num_embeddings)
        active_ids = gene_input_ids[gene_attention_mask.to(dtype=torch.bool)]
        if active_ids.numel():
            minimum = int(active_ids.min().item())
            maximum = int(active_ids.max().item())
            if minimum < 0 or maximum >= vocab_size:
                raise ValueError(
                    "Geneformer token IDs are incompatible with the loaded checkpoint: "
                    f"active range=[{minimum}, {maximum}], vocab_size={vocab_size}. "
                    "Use tokenized omics data produced with the checkpoint's exact "
                    "token dictionary; never clip or modulo gene token IDs."
                )
        outputs = self.gene_encoder(
            input_ids=gene_input_ids,
            attention_mask=gene_attention_mask,
            return_dict=True,
        )
        hidden = outputs.last_hidden_state
        gcn_dtype = next(self.gcn.parameters()).dtype
        hidden = hidden.to(dtype=gcn_dtype)
        adjacency = _safe_affinity_adjacency(hidden)
        graph_features = self.gcn(hidden, adjacency)
        queries = self.query_tokens.to(graph_features.dtype)
        queries = queries.unsqueeze(0).expand(graph_features.shape[0], -1, -1)
        return self.projector(queries, graph_features)

    def _prepare_mixed_inputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        gene_input_ids: Optional[torch.Tensor],
        gene_attention_mask: Optional[torch.Tensor],
        has_omics: Optional[torch.Tensor],
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        text_embeddings = self.llm.get_input_embeddings()(input_ids)
        batch_size = input_ids.shape[0]
        device = input_ids.device
        if has_omics is None:
            has_omics = torch.zeros(batch_size, dtype=torch.bool, device=device)
        else:
            has_omics = has_omics.to(device=device, dtype=torch.bool)

        prefix_embeddings = text_embeddings.new_zeros(
            batch_size, self.num_query_tokens, self.llm_hidden_size
        )
        prefix_attention = attention_mask.new_zeros(batch_size, self.num_query_tokens)
        omics_indices = has_omics.nonzero(as_tuple=False).flatten()

        if omics_indices.numel():
            if gene_input_ids is None or gene_attention_mask is None:
                raise ValueError("Omics samples require gene_input_ids and gene_attention_mask")
            selected_ids = gene_input_ids.index_select(0, omics_indices)
            selected_mask = gene_attention_mask.index_select(0, omics_indices)
            omics_features = self._encode_omics(selected_ids, selected_mask)
            prefix_embeddings = prefix_embeddings.index_copy(
                0, omics_indices, omics_features.to(text_embeddings.dtype)
            )
            prefix_attention[omics_indices] = 1

        result = {
            "inputs_embeds": torch.cat([prefix_embeddings, text_embeddings], dim=1),
            "attention_mask": torch.cat([prefix_attention, attention_mask], dim=1),
        }
        if labels is not None:
            ignored_prefix = labels.new_full(
                (batch_size, self.num_query_tokens), fill_value=-100
            )
            result["labels"] = torch.cat([ignored_prefix, labels], dim=1)
        return result

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        gene_input_ids: Optional[torch.Tensor] = None,
        gene_attention_mask: Optional[torch.Tensor] = None,
        has_omics: Optional[torch.Tensor] = None,
        profile_type: Optional[Sequence[str]] = None,
        group_id: Optional[Sequence[str]] = None,
        **kwargs,
    ):
        del profile_type, group_id, kwargs
        mixed = self._prepare_mixed_inputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            gene_input_ids=gene_input_ids,
            gene_attention_mask=gene_attention_mask,
            has_omics=has_omics,
        )
        return self.llm(**mixed, use_cache=False, return_dict=True)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        gene_input_ids: Optional[torch.Tensor] = None,
        gene_attention_mask: Optional[torch.Tensor] = None,
        has_omics: Optional[torch.Tensor] = None,
        **generation_kwargs,
    ) -> torch.Tensor:
        mixed = self._prepare_mixed_inputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=None,
            gene_input_ids=gene_input_ids,
            gene_attention_mask=gene_attention_mask,
            has_omics=has_omics,
        )
        return self.llm.generate(**mixed, **generation_kwargs)


# Backward-friendly name for scripts that expect scLaGene.
scLaGene = ScLaGeneQwen
