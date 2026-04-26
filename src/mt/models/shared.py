import torch
import torch.nn as nn
from torch import BoolTensor, Tensor


class RMSNorm(nn.RMSNorm):
    def forward(self, x: Tensor):
        dtype = x.dtype
        with torch.autocast(x.device.type, enabled=False):
            return super().forward(x.float()).to(dtype)


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


class Attention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout, batch_first=True)
        self.x_norm = RMSNorm(d_model)
        self.y_norm = RMSNorm(d_model)

    def forward(self, x: Tensor, y: Tensor, mask: BoolTensor):
        x = self.x_norm(x)
        y = self.y_norm(y)
        x, _ = self.attention(x, y, y, key_padding_mask=mask)
        return x
