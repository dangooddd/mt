import torch
import torch.nn as nn
from torch import Tensor

from ..base import DecoderOnly
from ..shared import FFN, PositionalEncoding, RMSNorm
from .classic import SelfAttention


class TransformerDecoderOnly(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float):
        super().__init__()
        self.attention = SelfAttention(d_model, num_heads, dropout)
        self.ffn = FFN(d_model, d_model * 2)

    def forward(self, x: Tensor):
        x = x + self.attention(x)
        x = x + self.ffn(x)
        return x


class TransformerDecoder(DecoderOnly):
    def __init__(
        self,
        vocab_size: int,
        pad_token_id: int,
        bos_token_id: int,
        eos_token_id: int,
        num_layers: int = 4,
        d_model: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_length: int = 2048,
    ):
        super().__init__(
            vocab_size=vocab_size,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
        )

        self.embedding = nn.Embedding(vocab_size, d_model, pad_token_id)
        self.positional_encoding = PositionalEncoding(d_model, max_len=max_length)
        self.norm = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

        self.decoder = nn.ModuleList(
            [
                TransformerDecoderOnly(
                    d_model=d_model,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def decode(
        self,
        input_ids: Tensor,
    ) -> Tensor:
        x = self.embedding(input_ids)
        x = self.positional_encoding(x)
        for layer in self.decoder:
            x = layer(x)
        x = self.head(self.norm(x))
        return x

    def forward(self, input_ids: Tensor, attention_mask: Tensor, type_ids: Tensor) -> Tensor:
        _ = type_ids
        _ = attention_mask
        return self.decode(input_ids)

    @torch.no_grad()
    def inference(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        type_ids: Tensor,
        max_length: int = 256,
    ) -> Tensor:
        _ = type_ids
        _ = attention_mask
        batch_size = input_ids.size(0)
        device = input_ids.device

        sequences = torch.full(
            (batch_size, max_length),
            self.pad_token_id,
            dtype=torch.long,
            device=device,
        )

        sequences[:, 0] = self.bos_token_id
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for t in range(1, max_length):
            logits_t = self.decode(sequences[:, :t])[:, -1, :]
            next_token = logits_t.argmax(dim=-1)

            active = ~finished
            sequences[active, t] = next_token[active]
            finished |= next_token == self.tgt_eos_token_id

            if finished.all():
                break

        return sequences
