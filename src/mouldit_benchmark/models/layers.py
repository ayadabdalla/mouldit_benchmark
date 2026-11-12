import numpy as np
import torch
import torch.nn as nn


class TemporalConvBlock(nn.Module):
    """Two dilated 1-D convolutions, each followed by LayerNorm over channels, LeakyReLU and dropout, plus a residual."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.layer_norm1 = nn.LayerNorm(out_channels)
        self.layer_norm2 = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(0.1)
        self.activation = nn.LeakyReLU(0.1)
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        """(B, C_in, T) -> (B, C_out, T)."""
        out = x
        for conv, norm in ((self.conv1, self.layer_norm1), (self.conv2, self.layer_norm2)):
            out = conv(out)
            out = norm(out.transpose(1, 2)).transpose(1, 2)
            out = self.dropout(self.activation(out))
        return out + self.residual(x)


class TemporalAttention(nn.Module):
    """
    Scaled dot-product self-attention over time steps.

    Input (B, C, T), or (B, T * hidden_dim) after a linear layer, which is read as T steps of hidden_dim
    features (seq_len = T); output (B, hidden_dim, T).
    """

    def __init__(self, hidden_dim, seq_len=None):
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.scale = np.sqrt(hidden_dim)
        self.seq_len = seq_len

    def forward(self, x):
        if x.dim() == 3:
            x = x.permute(0, 2, 1)
        else:
            x = x.view(x.shape[0], self.seq_len, x.shape[1] // self.seq_len)
        attention = torch.softmax(torch.bmm(self.query(x), self.key(x).transpose(1, 2)) / self.scale, dim=-1)
        return torch.bmm(attention, self.value(x)).permute(0, 2, 1)
