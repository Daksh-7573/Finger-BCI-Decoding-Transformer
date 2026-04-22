from __future__ import annotations

import torch
import torch.nn as nn


class EEGTransformerClassifier(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 128,
        dropout: float = 0.2,
        max_time_steps: int = 3000,
    ) -> None:
        super().__init__()

        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead}).")

        # Spatial projection over channels: (B, C, T) -> (B, d_model, T)
        self.spatial_proj = nn.Conv1d(in_channels=in_channels, out_channels=d_model, kernel_size=1)

        self.pos_embedding = nn.Parameter(torch.randn(1, max_time_steps, d_model) * 0.02)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: (batch, channels, time)
        x = self.spatial_proj(x)  # (B, d_model, T)
        x = x.transpose(1, 2)  # (B, T, d_model)

        t = x.size(1)
        if t > self.pos_embedding.size(1):
            raise ValueError(
                f"Input time length {t} exceeds max_time_steps {self.pos_embedding.size(1)}."
            )

        x = x + self.pos_embedding[:, :t, :]
        x = self.dropout(x)
        x = self.transformer(x)

        # Global average pooling across time.
        x = x.mean(dim=1)  # (B, d_model)
        logits = self.classifier(x)
        return logits
