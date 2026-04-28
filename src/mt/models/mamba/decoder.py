import torch
import torch.nn as nn
from mamba_ssm import Mamba
from torch import Tensor

from ..shared import RMSNorm


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


class MambaDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_token_id: int,
        bos_token_id: int,
        eos_token_id: int,
        embedding_size: int = 512,
        hidden_size: int = 512,
        num_layers: int = 6,
        num_heads: int | None = None,
        dropout: float = 0.1,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        _ = num_heads

        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        self.embedding = nn.Sequential(
            nn.Embedding(
                vocab_size,
                embedding_size,
                padding_idx=pad_token_id,
            ),
            nn.Dropout(dropout),
        )

        self.input = (
            nn.Linear(embedding_size, hidden_size, bias=False)
            if embedding_size != hidden_size
            else nn.Identity()
        )

        self.layers = nn.ModuleList(
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

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        type_ids: Tensor | None = None,
    ) -> Tensor:
        _ = attention_mask
        _ = type_ids

        x = self.embedding(input_ids)
        x = self.input(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        return self.output(x)

    @torch.no_grad()
    def inference(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        type_ids: Tensor,
        max_length: int = 128,
    ) -> Tensor:
        _ = type_ids
        batch_size, input_length = input_ids.shape
        device = input_ids.device

        sequences = torch.full(
            (batch_size, input_length + max_length),
            self.pad_token_id,
            dtype=torch.long,
            device=device,
        )
        sequences[:, :input_length] = input_ids

        generated = torch.full(
            (batch_size, max_length),
            self.pad_token_id,
            dtype=torch.long,
            device=device,
        )

        lengths = attention_mask.long().sum(dim=1).clamp_min(1)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for t in range(max_length):
            current_length = int(lengths.max().item())
            logits = self.forward(sequences[:, :current_length])
            index = lengths.sub(1).view(-1, 1, 1).expand(-1, 1, logits.size(-1))
            next_token = logits.gather(1, index).squeeze(1).argmax(dim=-1)

            active = ~finished
            positions = lengths[active]
            sequences[active, positions] = next_token[active]
            generated[active, t] = next_token[active]
            lengths[active] = lengths[active] + 1

            finished |= active & next_token.eq(self.eos_token_id)
            if finished.all():
                break

        return generated
