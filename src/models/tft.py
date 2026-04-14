"""Lightweight Temporal Fusion Transformer for 3-class direction prediction."""

from __future__ import annotations

import torch
from torch import nn


class GatedLinearUnit(nn.Module):
    """GLU block used for gated skip/residual transformations."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        transformed, gate = self.projection(x).chunk(2, dim=-1)
        return transformed * torch.sigmoid(gate)


class GatedResidualNetwork(nn.Module):
    """GRN with optional static context conditioning as in TFT."""

    def __init__(
        self,
        hidden_size: int,
        context_size: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.input_layer = nn.Linear(hidden_size, hidden_size)
        self.context_layer = nn.Linear(context_size, hidden_size) if context_size else None
        self.elu = nn.ELU()
        self.hidden_layer = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.glu = GatedLinearUnit(hidden_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        residual = x
        hidden = self.input_layer(x)

        if self.context_layer is not None and context is not None:
            hidden = hidden + self.context_layer(context)

        hidden = self.elu(hidden)
        hidden = self.hidden_layer(hidden)
        hidden = self.dropout(hidden)
        hidden = self.glu(hidden)

        return self.norm(hidden + residual)


class TemporalFusionTransformerModel(nn.Module):
    """TFT-style model with VSN, static ticker embedding, LSTM, and attention."""

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        ticker_vocab_size: int,
        hidden_size: int = 128,
        attention_heads: int = 4,
        ticker_embedding_dim: int = 16,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.ticker_embedding = nn.Embedding(ticker_vocab_size, ticker_embedding_dim)

        # Variable Selection Network: produce per-feature gates at each timestep.
        self.vsn_gate = nn.Linear(num_features, num_features)
        self.input_projection = nn.Linear(num_features, hidden_size)

        self.static_context_proj = nn.Linear(ticker_embedding_dim, hidden_size)
        self.grn = GatedResidualNetwork(hidden_size=hidden_size, context_size=hidden_size, dropout=dropout)

        self.encoder_lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.post_attn_glu = GatedLinearUnit(hidden_size, hidden_size)
        self.post_attn_norm = nn.LayerNorm(hidden_size)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        ticker_id: torch.Tensor | None = None,
        return_vsn_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape

        if ticker_id is None:
            ticker_id = torch.zeros(batch_size, device=x.device, dtype=torch.long)

        # Step 1: VSN-style feature weighting.
        vsn_logits = self.vsn_gate(x)
        vsn_weights = torch.softmax(vsn_logits, dim=-1)
        selected = x * vsn_weights

        # Step 2: project to hidden space and inject static ticker context.
        hidden = self.input_projection(selected)
        static_embed = self.ticker_embedding(ticker_id)
        static_context = self.static_context_proj(static_embed).unsqueeze(1).expand(-1, seq_len, -1)
        hidden = self.grn(hidden, context=static_context)

        # Step 3: temporal encoder + self-attention with gated skip connection.
        encoded, _ = self.encoder_lstm(hidden)
        attn_out, _ = self.attention(encoded, encoded, encoded, need_weights=False)
        gated = self.post_attn_glu(attn_out)
        fused = self.post_attn_norm(encoded + gated)

        # Step 4: classify based on final fused representation.
        pooled = fused[:, -1, :]
        logits = self.classifier(pooled)

        if return_vsn_weights:
            return logits, vsn_weights.mean(dim=1)

        return logits
