from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MidiTransformerConfig:
    vocab_size: int
    d_model: int = 512
    n_head: int = 8
    num_layers: int = 8
    dim_feedforward: int = 2048
    dropout: float = 0.1
    max_seq_len: int = 1024


class MidiTransformer(nn.Module):
    def __init__(self, config: MidiTransformerConfig):
        super().__init__()
        self.config = config
        self.tok_embed = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embed = nn.Embedding(config.max_seq_len, config.d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.n_head,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.num_layers)
        self.ln_f = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size)

    def _causal_mask(self, size: int, device: torch.device) -> torch.Tensor:
        # (T, T) with -inf above diagonal
        mask = torch.full((size, size), float("-inf"), device=device)
        mask = torch.triu(mask, diagonal=1)
        return mask

    def forward(self, input_ids: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # input_ids: (B, T)
        B, T = input_ids.shape
        device = input_ids.device
        pos = torch.arange(T, device=device).unsqueeze(0)
        x = self.tok_embed(input_ids) + self.pos_embed(pos)

        # Causal self-attention via decoder with empty memory
        causal_mask = self._causal_mask(T, device)
        x = self.decoder(
            tgt=x,
            memory=torch.zeros((B, 0, self.config.d_model), device=device),
            tgt_mask=causal_mask,
            tgt_key_padding_mask=attn_mask,
            memory_key_padding_mask=None,
        )
        x = self.ln_f(x)
        logits = self.head(x)
        return logits  # (B, T, V)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float = 0.9,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        self.eval()
        B = input_ids.size(0)
        out = input_ids
        for _ in range(max_new_tokens):
            T = out.size(1)
            if T >= self.config.max_seq_len:
                break
            logits = self.forward(out)[:, -1, :]  # (B, V)
            logits = logits / max(temperature, 1e-6)
            probs = F.softmax(logits, dim=-1)
            if top_p < 1.0:
                # nucleus sampling
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cum = torch.cumsum(sorted_probs, dim=-1)
                mask = cum > top_p
                # keep at least one token
                mask[:, 0] = False
                sorted_probs = sorted_probs.masked_fill(mask, 0.0)
                sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
                next_tokens = torch.multinomial(sorted_probs, num_samples=1)
                next_ids = sorted_idx.gather(-1, next_tokens)
            else:
                next_ids = torch.multinomial(probs, num_samples=1)

            out = torch.cat([out, next_ids], dim=1)
            if eos_token_id is not None:
                # If all sequences ended, stop early
                if ((next_ids == eos_token_id).sum().item() == B):
                    break
        return out


