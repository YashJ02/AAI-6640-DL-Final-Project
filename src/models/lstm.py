"""Stacked LSTM with additive temporal attention."""

from __future__ import annotations

import torch
from torch import nn


class TemporalAttention(nn.Module):
    """Bahdanau-style additive attention over sequence timesteps."""

    def __init__(self, hidden_size: int, attention_size: int) -> None:
        super().__init__()
        self.energy = nn.Linear(hidden_size, attention_size)
        self.score = nn.Linear(attention_size, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # hidden_states: (batch, seq_len, hidden_size)
        energy = torch.tanh(self.energy(hidden_states))
        scores = self.score(energy).squeeze(-1)
        weights = torch.softmax(scores, dim=1)

        # Weighted context vector summarizes informative timesteps.
        context = torch.bmm(weights.unsqueeze(1), hidden_states).squeeze(1)
        return context, weights


class LSTMTemporalAttentionModel(nn.Module):
    """3-layer LSTM encoder followed by temporal attention and classifier head."""

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        hidden_size: int = 256,
        num_layers: int = 3,
        dropout: float = 0.3,
        attention_size: int = 256,
        fc_hidden: int = 128,
    ) -> None:
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.attention = TemporalAttention(hidden_size=hidden_size, attention_size=attention_size)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, fc_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, num_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        ticker_id: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        del ticker_id  # Not used in LSTM baseline but kept for interface parity.

        lstm_out, _ = self.lstm(x)
        lstm_out = self.layer_norm(lstm_out)

        context, attn_weights = self.attention(lstm_out)
        logits = self.classifier(context)

        if return_attention:
            return logits, attn_weights

        return logits
