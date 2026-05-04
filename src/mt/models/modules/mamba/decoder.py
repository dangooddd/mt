import torch
import torch.nn as nn
from mamba_ssm import Mamba
from torch import Tensor

from ..base import DecoderOnly
from ..shared import FFN, Attention, RMSNorm


class MambaLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        dropout: float,
        d_state: int,
        d_conv: int,
        expand: int,
    ):
        super().__init__()

        self.norm = RMSNorm(hidden_size)
        self.mamba = Mamba(
            d_model=hidden_size,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dt_rank="auto",
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.norm(x)
        x = self.mamba(x)
        x = residual + self.dropout(x)
        return x


class AttentionLayer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float):
        super().__init__()
        self.attention = Attention(hidden_size, num_heads, dropout)
        self.ffn = FFN(hidden_size, hidden_size * 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, attention_mask: Tensor, type_ids: Tensor) -> Tensor:
        source_mask = attention_mask & type_ids.eq(0)
        update_mask = source_mask.unsqueeze(-1)
        key_padding_mask = ~source_mask

        attention_out = self.attention(x, x, key_padding_mask)
        updated = x + self.dropout(attention_out)
        x = torch.where(update_mask, updated, x)

        ffn_out = self.ffn(x)
        updated = x + self.dropout(ffn_out)
        x = torch.where(update_mask, updated, x)
        return x


class MambaDecoder(DecoderOnly):
    def __init__(
        self,
        vocab_size: int,
        pad_token_id: int,
        bos_token_id: int,
        eos_token_id: int,
        hidden_size: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        attention_layers: int = 2,
        dropout: float = 0.1,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__(
            vocab_size=vocab_size,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
        )

        self.embedding = nn.Sequential(
            nn.Embedding(
                vocab_size,
                hidden_size,
                padding_idx=pad_token_id,
            ),
            nn.Dropout(dropout),
        )

        self.encoder_layers = nn.ModuleList(
            [
                AttentionLayer(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(attention_layers)
            ]
        )

        self.decoder_layers = nn.ModuleList(
            [
                MambaLayer(
                    hidden_size=hidden_size,
                    dropout=dropout,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm = RMSNorm(hidden_size)
        self.output = nn.Linear(hidden_size, vocab_size, bias=False)

    def encode(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        type_ids: Tensor,
    ) -> Tensor:
        x = self.embedding(input_ids)

        for layer in self.encoder_layers:
            x = layer(x, attention_mask, type_ids)

        return x

    def decode(self, x: Tensor) -> Tensor:
        for layer in self.decoder_layers:
            x = layer(x)

        x = self.norm(x)
        x = self.output(x)
        return x

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        type_ids: Tensor,
    ) -> Tensor:
        x = self.encode(input_ids, attention_mask, type_ids)
        return self.decode(x)

    @torch.no_grad()
    def inference(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        type_ids: Tensor,
        max_length: int = 256,
        temperature: float = 0.0,
    ) -> Tensor:
        batch_size, input_length = input_ids.shape
        device = input_ids.device

        generated = torch.full(
            (batch_size, max_length),
            self.pad_token_id,
            dtype=torch.long,
            device=device,
        )

        encoded = self.encode(input_ids, attention_mask, type_ids)
        hidden = torch.zeros(
            batch_size,
            input_length + max_length,
            encoded.size(-1),
            dtype=encoded.dtype,
            device=device,
        )
        hidden[:, :input_length] = encoded

        lengths = attention_mask.long().sum(dim=1).clamp_min(1)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for t in range(max_length):
            current_length = int(lengths.max().item())
            logits = self.decode(hidden[:, :current_length])
            last_logits = logits[torch.arange(batch_size, device=device), lengths - 1]
            next_token = self._sample_next_token(last_logits, temperature)

            active = ~finished
            positions = lengths[active]
            generated[active, t] = next_token[active]

            hidden[active, positions] = self.embedding(next_token)[active]
            lengths[active] = lengths[active] + 1

            finished |= active & next_token.eq(self.eos_token_id)
            if finished.all():
                break

        return generated
