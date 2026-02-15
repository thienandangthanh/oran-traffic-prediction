"""Custom Autoformer implementation based on Wu et al. paper.

This implements the core Autoformer components:
- Series Decomposition
- Auto-Correlation Mechanism
- Progressive Decomposition Architecture
"""

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SeriesDecomp(nn.Module):
    """Series decomposition block using moving average."""

    def __init__(self, kernel_size: int = 25):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decompose series into trend and seasonal components.

        Args:
            x: Input tensor (batch, seq_len, features)

        Returns:
            Tuple of (seasonal, trend)
        """
        # Padding
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x_padded = torch.cat([front, x, end], dim=1)

        # Moving average (trend)
        x_padded = x_padded.permute(0, 2, 1)  # (batch, features, seq_len)
        trend = self.avg(x_padded)
        trend = trend.permute(0, 2, 1)  # (batch, seq_len, features)

        # Seasonal = original - trend
        seasonal = x - trend

        return seasonal, trend


class AutoCorrelation(nn.Module):
    """Auto-Correlation mechanism for period-based dependencies."""

    def __init__(self, d_model: int, n_heads: int, factor: int = 3):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.factor = factor
        self.d_keys = d_model // n_heads
        self.d_values = d_model // n_heads

        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.out_projection = nn.Linear(d_model, d_model)

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """Apply auto-correlation mechanism.

        Args:
            queries: Query tensor (batch, seq_len, d_model)
            keys: Key tensor (batch, seq_len, d_model)
            values: Value tensor (batch, seq_len, d_model)

        Returns:
            Output tensor (batch, seq_len, d_model)
        """
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        # Project and reshape
        queries = self.query_projection(queries).view(B, L, H, self.d_keys)
        keys = self.key_projection(keys).view(B, S, H, self.d_keys)
        values = self.value_projection(values).view(B, S, H, self.d_values)

        # Transpose for attention computation
        queries = queries.transpose(1, 2)  # (B, H, L, d_keys)
        keys = keys.transpose(1, 2)  # (B, H, S, d_keys)
        values = values.transpose(1, 2)  # (B, H, S, d_values)

        # Simplified auto-correlation (using standard attention for now)
        # Full FFT-based auto-correlation can be added later
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / np.sqrt(self.d_keys)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, values)

        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(B, L, -1)
        out = self.out_projection(out)

        return out


class EncoderLayer(nn.Module):
    """Autoformer encoder layer with auto-correlation and decomposition."""

    def __init__(
        self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1, moving_avg: int = 25, factor: int = 3
    ):
        super().__init__()

        self.auto_correlation = AutoCorrelation(d_model, n_heads, factor)
        self.decomp1 = SeriesDecomp(moving_avg)
        self.decomp2 = SeriesDecomp(moving_avg)

        self.conv1 = nn.Conv1d(d_model, d_ff, 1)
        self.conv2 = nn.Conv1d(d_ff, d_model, 1)

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through encoder layer.

        Args:
            x: Input tensor (batch, seq_len, d_model)

        Returns:
            Output tensor (batch, seq_len, d_model)
        """
        # Auto-correlation
        new_x = self.auto_correlation(x, x, x)
        x = x + self.dropout(new_x)
        x, _ = self.decomp1(x)

        # Feed-forward
        y = x.transpose(1, 2)  # (batch, d_model, seq_len)
        y = self.dropout(self.activation(self.conv1(y)))
        y = self.dropout(self.conv2(y))
        y = y.transpose(1, 2)  # (batch, seq_len, d_model)

        res, _ = self.decomp2(x + y)

        return res


class DecoderLayer(nn.Module):
    """Autoformer decoder layer with auto-correlation and decomposition."""

    def __init__(
        self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1, moving_avg: int = 25, factor: int = 3
    ):
        super().__init__()

        self.self_attention = AutoCorrelation(d_model, n_heads, factor)
        self.cross_attention = AutoCorrelation(d_model, n_heads, factor)
        self.decomp1 = SeriesDecomp(moving_avg)
        self.decomp2 = SeriesDecomp(moving_avg)
        self.decomp3 = SeriesDecomp(moving_avg)

        self.conv1 = nn.Conv1d(d_model, d_ff, 1)
        self.conv2 = nn.Conv1d(d_ff, d_model, 1)

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor, cross: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through decoder layer.

        Args:
            x: Decoder input (batch, seq_len, d_model)
            cross: Encoder output (batch, seq_len, d_model)

        Returns:
            Tuple of (output, trend)
        """
        # Self auto-correlation
        x = x + self.dropout(self.self_attention(x, x, x))
        x, trend1 = self.decomp1(x)

        # Cross auto-correlation
        x = x + self.dropout(self.cross_attention(x, cross, cross))
        x, trend2 = self.decomp2(x)

        # Feed-forward
        y = x.transpose(1, 2)
        y = self.dropout(self.activation(self.conv1(y)))
        y = self.dropout(self.conv2(y))
        y = y.transpose(1, 2)

        x, trend3 = self.decomp3(x + y)

        trend = trend1 + trend2 + trend3

        return x, trend


class CustomAutoformer(nn.Module):
    """Custom Autoformer model for time series forecasting."""

    def __init__(
        self,
        input_size: int = 1,
        d_model: int = 512,
        n_heads: int = 8,
        e_layers: int = 2,
        d_layers: int = 1,
        d_ff: int = 2048,
        dropout: float = 0.1,
        moving_avg: int = 25,
        factor: int = 3,
        output_length: int = 96,
    ):
        super().__init__()

        self.input_size = input_size
        self.d_model = d_model
        self.output_length = output_length

        # Input embedding
        self.enc_embedding = nn.Linear(input_size, d_model)
        self.dec_embedding = nn.Linear(input_size, d_model)

        # Encoder
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_ff, dropout, moving_avg, factor) for _ in range(e_layers)]
        )

        # Decoder
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, n_heads, d_ff, dropout, moving_avg, factor) for _ in range(d_layers)]
        )

        # Output projection
        self.projection = nn.Linear(d_model, input_size)

        # Series decomposition for trend
        self.decomp = SeriesDecomp(moving_avg)

    def forward(self, x_enc: torch.Tensor, x_dec: torch.Tensor) -> torch.Tensor:
        """Forward pass through Autoformer.

        Args:
            x_enc: Encoder input (batch, context_length, input_size)
            x_dec: Decoder input (batch, prediction_length, input_size)

        Returns:
            Predictions (batch, prediction_length, input_size)
        """
        # Decompose encoder input
        seasonal_init, trend_init = self.decomp(x_enc)

        # Encoder embedding
        enc_out = self.enc_embedding(seasonal_init)

        # Encoder layers
        for layer in self.encoder_layers:
            enc_out = layer(enc_out)

        # Decoder embedding
        dec_out = self.dec_embedding(x_dec)

        # Initialize trend
        trend = torch.zeros_like(dec_out[:, :, : self.input_size])

        # Decoder layers
        for layer in self.decoder_layers:
            dec_out, layer_trend = layer(dec_out, enc_out)
            trend = trend + self.projection(layer_trend)

        # Final prediction = seasonal + trend
        seasonal_out = self.projection(dec_out)
        output = seasonal_out + trend

        return output
