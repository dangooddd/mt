import torch
import torch.nn as nn
from mamba_ssm import Mamba
from torch import BoolTensor, Tensor

from ..base import EncoderDecoder
from ..shared import FFN, Attention, RMSNorm


def reverse_padded_sequence(x: Tensor, lengths: Tensor):
    batch_size, seq_len, dim = x.shape
    idx = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
    rev_idx = lengths.unsqueeze(1) - 1 - idx
    gather_idx = torch.where(idx < lengths.unsqueeze(1), rev_idx, idx).clamp_min(0)
    return x.gather(1, gather_idx.unsqueeze(-1).expand(-1, -1, dim))


class MambaBlock(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        num_layers: int,
        dropout: float,
        d_state: int,
        d_conv: int,
        expand: int,
    ):
        super().__init__()

        self.input_proj = (
            nn.Linear(input_size, output_size, bias=False)
            if input_size != output_size
            else nn.Identity()
        )

        self.layers = nn.ModuleList(
            [
                Mamba(
                    d_model=output_size,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                )
                for _ in range(num_layers)
            ]
        )

        self.norms = nn.ModuleList([RMSNorm(output_size) for _ in range(num_layers)])
        self.norm = RMSNorm(output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor):
        x = self.input_proj(x)

        for layer, norm in zip(self.layers, self.norms):
            x = x + self.dropout(layer(norm(x)))

        x = self.norm(x)
        return x


class Head(nn.Module):
    def __init__(self, input_size: int, output_size: int, num_heads: int, dropout: float):
        super().__init__()
        self.attention = Attention(input_size, num_heads, dropout)
        self.ffn = FFN(input_size, input_size * 2)
        self.norm = RMSNorm(input_size)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x: Tensor, y: Tensor, mask: BoolTensor):
        x = x + self.attention(x, y, mask)
        x = x + self.ffn(x)
        x = self.norm(x)
        x = self.dropout(x)
        x = self.linear(x)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_token_id: int,
        num_layers: int,
        hidden_size: int,
        embedding_size: int,
        dropout: float,
        d_state: int,
        d_conv: int,
        expand: int,
    ):
        self.pad_token_id = pad_token_id
        super().__init__()

        self.embedding = nn.Sequential(
            nn.Embedding(
                vocab_size,
                embedding_size,
                padding_idx=pad_token_id,
            ),
            nn.Dropout(dropout),
        )

        self.forward_mamba = MambaBlock(
            input_size=embedding_size,
            output_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

        self.backward_mamba = MambaBlock(
            input_size=embedding_size,
            output_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

        self.norm = RMSNorm(hidden_size * 2)

    def forward(self, input_ids: Tensor, attention_mask: Tensor):
        lengths = attention_mask.long().sum(dim=1)
        embedding = self.embedding(input_ids)

        forward_outputs = self.forward_mamba(embedding)
        backward_embedding = reverse_padded_sequence(embedding, lengths)
        backward_outputs = self.backward_mamba(backward_embedding)
        backward_outputs = reverse_padded_sequence(backward_outputs, lengths)

        outputs = torch.cat([forward_outputs, backward_outputs], dim=2)
        outputs = self.norm(outputs)
        return outputs


class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_token_id: int,
        num_layers: int,
        hidden_size: int,
        embedding_size: int,
        dropout: float,
        d_state: int,
        d_conv: int,
        expand: int,
    ):
        self.pad_token_id = pad_token_id
        super().__init__()

        self.embedding = nn.Sequential(
            nn.Embedding(
                vocab_size,
                embedding_size,
                padding_idx=pad_token_id,
            ),
            nn.Dropout(dropout),
        )

        self.mamba = MambaBlock(
            input_size=embedding_size,
            output_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

    def forward(self, input_ids: Tensor):
        embedding = self.embedding(input_ids)
        outputs = self.mamba(embedding)
        return outputs


class MambaSeq2Seq(EncoderDecoder):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        src_pad_token_id: int,
        tgt_pad_token_id: int,
        tgt_bos_token_id: int,
        tgt_eos_token_id: int,
        embedding_size: int = 256,
        hidden_size: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.2,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            src_pad_token_id=src_pad_token_id,
            tgt_pad_token_id=tgt_pad_token_id,
            tgt_bos_token_id=tgt_bos_token_id,
            tgt_eos_token_id=tgt_eos_token_id,
        )

        self.encoder = Encoder(
            vocab_size=src_vocab_size,
            pad_token_id=src_pad_token_id,
            num_layers=num_layers,
            hidden_size=hidden_size,
            embedding_size=embedding_size,
            dropout=dropout,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

        self.decoder = Decoder(
            vocab_size=tgt_vocab_size,
            pad_token_id=tgt_pad_token_id,
            num_layers=num_layers,
            hidden_size=hidden_size * 2,
            embedding_size=embedding_size,
            dropout=dropout,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

        self.head = Head(
            input_size=hidden_size * 2,
            output_size=tgt_vocab_size,
            num_heads=num_heads,
            dropout=dropout,
        )

    def forward(
        self,
        input_ids: Tensor,
        output_ids: Tensor,
        attention_mask: Tensor,
    ):
        encoder_outputs = self.encoder(input_ids, attention_mask)
        decoder_outputs = self.decoder(output_ids)
        logits = self.head(decoder_outputs, encoder_outputs, ~attention_mask.bool())
        return logits

    @torch.no_grad()
    def inference(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        max_length: int = 128,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ):
        batch_size = input_ids.size(0)
        device = input_ids.device
        encoder_outputs = self.encoder(input_ids, attention_mask)

        sequences = torch.full(
            (batch_size, max_length),
            self.tgt_pad_token_id,
            device=device,
            dtype=torch.long,
        )
        sequences[:, 0] = self.tgt_bos_token_id

        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for t in range(1, max_length):
            decoder_outputs = self.decoder(sequences[:, :t])
            decoder_output = decoder_outputs[:, -1:, :]
            logits = self.head(
                decoder_output,
                encoder_outputs,
                ~attention_mask.bool(),
            )
            next_token = self._sample_next_token(logits, temperature, top_p).squeeze(1)

            mask = ~finished
            sequences[mask, t] = next_token[mask]
            finished |= next_token == self.tgt_eos_token_id

            if finished.all():
                break

        return sequences
