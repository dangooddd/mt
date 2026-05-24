import math
from typing import cast

import torch
import torch.nn as nn
from flash_attn import flash_attn_func, flash_attn_varlen_func, flash_attn_varlen_kvpacked_func
from flash_attn.bert_padding import pad_input, unpad_input
from torch import Tensor

from ..base import EncoderDecoderBilingual
from ..shared import RMSNorm


class FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
        self.silu = nn.SiLU()
        self.norm = RMSNorm(d_model)

    def forward(self, x: Tensor):
        x = self.norm(x)
        x = self.down_proj(self.silu(self.gate_proj(x)) * self.up_proj(x))
        return x


class RotaryEmbedding(nn.Module):
    inv_freq: Tensor

    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()

        if dim % 2 != 0:
            raise ValueError("RoPE head dimension must be even")

        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self, seq_length: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[Tensor, Tensor]:
        positions = torch.arange(seq_length, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(positions, self.inv_freq.to(device=device))
        cos = freqs.cos().to(dtype=dtype)[None, :, None, :]
        sin = freqs.sin().to(dtype=dtype)[None, :, None, :]
        return cos, sin

    def rotate(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)


class SelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float,
        causal: bool = False,
        theta: float = 10000.0,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout = dropout
        self.causal = causal
        self.rope = RotaryEmbedding(self.head_dim, theta=theta)

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.norm = RMSNorm(d_model)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        batch_size, seq_length, d_model = x.shape
        x = self.norm(x)

        q = self.q_proj(x).view(batch_size, seq_length, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_length, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_length, self.num_heads, self.head_dim)

        cos, sin = self.rope(seq_length, x.device, q.dtype)
        q = self.rope.rotate(q, cos, sin)
        k = self.rope.rotate(k, cos, sin)

        if mask is None:
            x = flash_attn_func(
                q,
                k,
                v,
                dropout_p=self.dropout if self.training else 0.0,
                causal=self.causal,
            )

        else:
            attention_mask = ~mask
            q, indices, cu_seqlens, max_seqlen, _ = unpad_input(q, attention_mask)
            k, _, _, _, _ = unpad_input(k, attention_mask)
            v, _, _, _, _ = unpad_input(v, attention_mask)
            x = flash_attn_varlen_func(
                q,
                k,
                v,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                dropout_p=self.dropout if self.training else 0.0,
                causal=self.causal,
            )
            x = pad_input(x, indices, batch_size, seq_length)

        x = x.contiguous().view(batch_size, seq_length, d_model)
        return self.o_proj(x)


class CrossAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout = dropout

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.kv_proj = nn.Linear(d_model, 2 * d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.x_norm = RMSNorm(d_model)
        self.y_norm = RMSNorm(d_model)

    def forward(self, x: Tensor, y: Tensor, mask: Tensor) -> Tensor:
        batch_size, x_length, d_model = x.shape
        y_length = y.size(1)

        x = self.x_norm(x)
        y = self.y_norm(y)

        q = self.q_proj(x).view(batch_size, x_length, self.num_heads, self.head_dim)
        kv = self.kv_proj(y).view(batch_size, y_length, 2, self.num_heads, self.head_dim)

        kv, _, cu_seqlens_kv, max_seqlen_kv, _ = unpad_input(kv, ~mask)
        q = q.reshape(batch_size * x_length, self.num_heads, self.head_dim)
        cu_seqlens_q = torch.arange(
            0,
            (batch_size + 1) * x_length,
            step=x_length,
            dtype=torch.int32,
            device=q.device,
        )

        x = flash_attn_varlen_kvpacked_func(
            q,
            kv,
            cu_seqlens_q,
            cu_seqlens_kv,
            x_length,
            max_seqlen_kv,
            dropout_p=self.dropout if self.training else 0.0,
            causal=False,
        )

        x = x.view(batch_size, x_length, self.num_heads, self.head_dim)
        x = x.contiguous().view(batch_size, x_length, d_model)
        return self.o_proj(x)


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float,
        inner_multiplier: float = 2.0,
        theta: float = 10000.0,
    ):
        super().__init__()

        self.attention = SelfAttention(
            d_model,
            num_heads,
            dropout,
            theta=theta,
            causal=False,
        )

        self.ffn = FFN(
            d_model,
            int(d_model * inner_multiplier),
        )

    def forward(self, x: Tensor, mask: Tensor):
        x = x + self.attention(x, mask)
        x = x + self.ffn(x)
        return x


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 8,
        dropout: float = 0.2,
        inner_multiplier: float = 2.0,
        theta: float = 10000.0,
    ):
        super().__init__()

        self.self_attn = SelfAttention(
            d_model,
            num_heads=num_heads,
            dropout=dropout,
            theta=theta,
            causal=True,
        )

        self.cross_attn = CrossAttention(
            d_model,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.ffn = FFN(
            d_model,
            int(d_model * inner_multiplier),
        )

    def forward(self, x: Tensor, y: Tensor, mask: Tensor):
        x = x + self.self_attn(x)
        x = x + self.cross_attn(x, y, mask)
        x = x + self.ffn(x)
        return x


class CompleteTransformerSeq2Seq(EncoderDecoderBilingual):
    def __init__(
        self,
        vocab_size: int,
        pad_token_id: int,
        bos_token_id: int,
        eos_token_id: int,
        encoder_layers: int = 4,
        decoder_layers: int = 4,
        d_model: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
        inner_multiplier: float = 2.0,
        theta: float = 10000.0,
        init_std_base: float = 0.02,
    ):
        super().__init__(
            vocab_size=vocab_size,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
        )

        self.embedding = nn.Embedding(vocab_size, d_model, pad_token_id)
        self.norm = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

        self.encoder = nn.ModuleList(
            [
                TransformerEncoder(
                    d_model=d_model,
                    num_heads=num_heads,
                    dropout=dropout,
                    inner_multiplier=inner_multiplier,
                    theta=theta,
                )
                for _ in range(encoder_layers)
            ]
        )

        self.decoder = nn.ModuleList(
            [
                TransformerDecoder(
                    d_model=d_model,
                    num_heads=num_heads,
                    dropout=dropout,
                    inner_multiplier=inner_multiplier,
                    theta=theta,
                )
                for _ in range(decoder_layers)
            ]
        )

        self._init_weights(init_std_base)

    def _init_weights(self, std_base: float):
        std_res_enc = std_base / math.sqrt(2 * len(self.encoder))
        std_res_dec = std_base / math.sqrt(3 * len(self.decoder))

        nn.init.normal_(self.embedding.weight, mean=0.0, std=std_base)
        nn.init.zeros_(self.embedding.weight[self.embedding.padding_idx])

        nn.init.normal_(self.head.weight, mean=0.0, std=std_base)
        nn.init.zeros_(self.head.bias)

        for layer in self.encoder:
            layer = cast(TransformerEncoder, layer)

            nn.init.normal_(layer.attention.q_proj.weight, mean=0.0, std=std_base)
            nn.init.normal_(layer.attention.k_proj.weight, mean=0.0, std=std_base)
            nn.init.normal_(layer.attention.v_proj.weight, mean=0.0, std=std_base)
            nn.init.normal_(layer.attention.o_proj.weight, mean=0.0, std=std_res_enc)

            nn.init.normal_(layer.ffn.gate_proj.weight, mean=0.0, std=std_base)
            nn.init.normal_(layer.ffn.up_proj.weight, mean=0.0, std=std_base)
            nn.init.normal_(layer.ffn.down_proj.weight, mean=0.0, std=std_res_enc)

        for layer in self.decoder:
            layer = cast(TransformerDecoder, layer)

            nn.init.normal_(layer.self_attn.q_proj.weight, mean=0.0, std=std_base)
            nn.init.normal_(layer.self_attn.k_proj.weight, mean=0.0, std=std_base)
            nn.init.normal_(layer.self_attn.v_proj.weight, mean=0.0, std=std_base)
            nn.init.normal_(layer.self_attn.o_proj.weight, mean=0.0, std=std_res_dec)

            nn.init.normal_(layer.cross_attn.q_proj.weight, mean=0.0, std=std_base)
            nn.init.normal_(layer.cross_attn.kv_proj.weight, mean=0.0, std=std_base)
            nn.init.normal_(layer.cross_attn.o_proj.weight, mean=0.0, std=std_res_dec)

            nn.init.normal_(layer.ffn.gate_proj.weight, mean=0.0, std=std_base)
            nn.init.normal_(layer.ffn.up_proj.weight, mean=0.0, std=std_base)
            nn.init.normal_(layer.ffn.down_proj.weight, mean=0.0, std=std_res_dec)

    def encode(self, input_ids: Tensor, key_padding_mask: Tensor) -> Tensor:
        y = self.embedding(input_ids)
        for layer in self.encoder:
            y = layer(y, key_padding_mask)
        return y

    def decode(
        self, output_ids: Tensor, encoder_outputs: Tensor, key_padding_mask: Tensor
    ) -> Tensor:
        x = self.embedding(output_ids)
        for layer in self.decoder:
            x = layer(x, encoder_outputs, key_padding_mask)
        x = self.head(self.norm(x))
        return x

    def forward(self, input_ids: Tensor, output_ids: Tensor, attention_mask: Tensor) -> Tensor:
        key_padding_mask = ~attention_mask.bool()
        encoder_outputs = self.encode(input_ids, key_padding_mask)
        return self.decode(output_ids, encoder_outputs, key_padding_mask)

    @torch.no_grad()
    def inference(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        max_length: int = 128,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> Tensor:
        batch_size = input_ids.size(0)
        device = input_ids.device
        key_padding_mask = ~attention_mask.bool()
        encoder_outputs = self.encode(input_ids, key_padding_mask)

        sequences = torch.full(
            (batch_size, max_length),
            self.pad_token_id,
            dtype=torch.long,
            device=device,
        )

        sequences[:, 0] = self.bos_token_id
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for t in range(1, max_length):
            logits_t = self.decode(sequences[:, :t], encoder_outputs, key_padding_mask)[:, -1, :]
            next_token = self._sample_next_token(logits_t, temperature, top_p)

            active = ~finished
            sequences[active, t] = next_token[active]
            finished |= next_token == self.eos_token_id

            if finished.all():
                break

        return sequences
