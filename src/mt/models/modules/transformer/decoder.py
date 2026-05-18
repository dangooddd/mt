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
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> Tensor:
        _ = type_ids
        batch_size, input_length = input_ids.shape
        device = input_ids.device

        generated = torch.full(
            (batch_size, max_length),
            self.pad_token_id,
            dtype=torch.long,
            device=device,
        )

        sequences = torch.full(
            (batch_size, input_length + max_length),
            self.pad_token_id,
            dtype=torch.long,
            device=device,
        )
        sequences[:, :input_length] = input_ids

        lengths = attention_mask.long().sum(dim=1).clamp_min(1)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for t in range(max_length):
            current_length = int(lengths.max().item())
            logits = self.decode(sequences[:, :current_length])
            last_logits = logits[torch.arange(batch_size, device=device), lengths - 1]
            next_token = self._sample_next_token(last_logits, temperature, top_p)

            active = ~finished
            positions = lengths[active]
            generated[active, t] = next_token[active]
            sequences[active, positions] = next_token[active]
            lengths[active] = lengths[active] + 1

            finished |= active & next_token.eq(self.eos_token_id)
            if finished.all():
                break

        return generated
