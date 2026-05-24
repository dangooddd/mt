import math

import torch
import torch.nn as nn
from mamba_ssm import Mamba
from torch import Tensor


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype

        with torch.autocast(x.device.type, enabled=False):
            x = x.to(torch.float32)
            v = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(v + self.eps)

        return self.weight * x.to(dtype)


class FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_ff, d_model, bias=False)
        self.silu = nn.SiLU()
        self.norm = RMSNorm(d_model)

    def forward(self, x: Tensor):
        x = self.norm(x)
        x = self.w3(self.silu(self.w1(x)) * self.w2(x))
        return x


class PositionalEncoding(nn.Module):
    pe: Tensor

    def __init__(self, d_model: int, max_length: int = 5000):
        super().__init__()
        position = torch.arange(max_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_length, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: Tensor):
        return x + self.pe[:, : x.size(1)].to(dtype=x.dtype)


class Attention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout, batch_first=True)
        self.x_norm = RMSNorm(d_model)
        self.y_norm = RMSNorm(d_model)

    def forward(self, x: Tensor, y: Tensor, mask: Tensor):
        x = self.x_norm(x)
        y = self.y_norm(y)
        x, _ = self.attention(x, y, y, key_padding_mask=mask)
        return x


class Transformer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float, inner_multiplier: float = 2):
        super().__init__()
        self.attention = Attention(d_model, num_heads, dropout)
        self.ffn = FFN(d_model, int(d_model * inner_multiplier))

    def forward(self, x: Tensor, y: Tensor, mask: Tensor):
        x = x + self.attention(x, y, mask)
        x = x + self.ffn(x)
        return x


class MambaBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int,
        d_conv: int,
    ):
        super().__init__()
        self.mamba = Mamba(d_model, d_state=d_state, d_conv=d_conv)
        self.norm = RMSNorm(d_model)

    def forward(self, x: Tensor):
        x = x + self.mamba(self.norm(x))
        return x
