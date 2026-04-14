"""Dilated CNN + BiLSTM hybrid model for intraday direction classification."""

from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    """Conv1d + batch norm + ReLU used in each dilation branch."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
    ) -> None:
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2

        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=padding,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DilatedCNNLSTMModel(nn.Module):
    """Dilated conv feature extractor followed by bidirectional LSTM classifier."""

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        conv_channels: int = 64,
        conv_kernel_size: int = 3,
        dilations: list[int] | tuple[int, ...] = (1, 2, 4),
        lstm_hidden_size: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.3,
        fc_hidden: int = 128,
    ) -> None:
        super().__init__()

        self.conv_branches = nn.ModuleList(
            [
                ConvBlock(
                    in_channels=num_features,
                    out_channels=conv_channels,
                    kernel_size=conv_kernel_size,
                    dilation=dilation,
                )
                for dilation in dilations
            ]
        )

        merged_channels = conv_channels * len(dilations)
        self.bi_lstm = nn.LSTM(
            input_size=merged_channels,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=True,
        )

        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden_size * 2, fc_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, num_classes),
        )

    def forward(self, x: torch.Tensor, ticker_id: torch.Tensor | None = None) -> torch.Tensor:
        del ticker_id  # Not used in CNN-LSTM baseline but kept for trainer compatibility.

        # Step 1: move to channel-first shape for Conv1d branches.
        x_ch_first = x.permute(0, 2, 1)

        # Step 2: apply dilated convolutions and concatenate feature maps.
        conv_outputs = [branch(x_ch_first) for branch in self.conv_branches]
        merged = torch.cat(conv_outputs, dim=1)

        # Step 3: convert back to sequence-first representation for LSTM.
        sequence = merged.permute(0, 2, 1)
        lstm_out, _ = self.bi_lstm(sequence)

        # Use last timestep from BiLSTM output as sequence summary.
        pooled = lstm_out[:, -1, :]
        logits = self.classifier(pooled)
        return logits
