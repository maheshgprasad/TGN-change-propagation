from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


class CosineTimeEncoding(nn.Module):
    """Small learned continuous-time encoder for elapsed seconds."""

    def __init__(self, dim: int = 16):
        super().__init__()
        self.log_freq = nn.Parameter(torch.linspace(-6.0, 0.0, dim))
        self.phase = nn.Parameter(torch.zeros(dim))

    def forward(self, delta_seconds: torch.Tensor) -> torch.Tensor:
        x = torch.log1p(torch.clamp(delta_seconds, min=0.0)).unsqueeze(-1)
        freq = torch.exp(self.log_freq).view(1, 1, -1)
        phase = self.phase.view(1, 1, -1)
        return torch.cos(x * freq + phase)


class TemporalAttentionScorer(nn.Module):
    """One shared scorer reused for every candidate pair."""

    def __init__(
        self,
        feature_dim: int,
        model_dim: int = 64,
        heads: int = 4,
        time_dim: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.query_proj = nn.Linear(feature_dim, model_dim)
        self.history_proj = nn.Linear(feature_dim, model_dim)
        self.time_encoder = CosineTimeEncoding(time_dim)
        self.time_proj = nn.Linear(time_dim, model_dim)
        self.same_target_proj = nn.Linear(1, model_dim, bias=False)
        self.attention = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(model_dim)
        self.head = nn.Sequential(
            nn.Linear(model_dim * 2, model_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, 1),
        )

    def forward(
        self,
        current_x: torch.Tensor,
        history_x: torch.Tensor,
        history_age: torch.Tensor,
        same_target: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> torch.Tensor:
        query = self.query_proj(current_x).unsqueeze(1)
        hist = self.history_proj(history_x)
        hist = hist + self.time_proj(self.time_encoder(history_age))
        hist = hist + self.same_target_proj(same_target.unsqueeze(-1))

        real_history = history_mask.any(dim=1)
        safe_mask = history_mask.clone()
        no_hist = ~real_history
        if no_hist.any():
            safe_mask[no_hist, 0] = True
            hist = hist.clone()
            hist[no_hist, 0, :] = 0.0

        context, _ = self.attention(
            query,
            hist,
            hist,
            key_padding_mask=~safe_mask,
            need_weights=False,
        )
        context = context.squeeze(1)
        if no_hist.any():
            context = context.clone()
            context[no_hist] = 0.0

        q = self.norm(query.squeeze(1))
        return self.head(torch.cat([q, context], dim=-1)).squeeze(-1)


@dataclass(frozen=True)
class ModelConfig:
    model_dim: int = 64
    heads: int = 4
    time_dim: int = 16
    history_k: int = 20
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 1024
    max_epochs: int = 20
    patience: int = 5
