import torch
import torch.nn as nn
from mamba_ssm import Mamba2
from torch import Tensor

from ..base import EncoderDecoder
from ..shared import PositionalEncoding, RMSNorm, Transformer


class Mamba2Block(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int,
        d_conv: int,
        headdim: int,
    ):
        super().__init__()
        self.mamba = Mamba2(d_model, d_state=d_state, d_conv=d_conv, headdim=headdim)
        self.norm = RMSNorm(d_model)

    def forward(self, x: Tensor):
        return x + self.mamba(self.norm(x))


class Mamba2HybridBlock(nn.Module):
    def __init__(
        self,
        d_model: int = 768,
        num_heads: int = 16,
        dropout: float = 0.1,
        headdim: int = 64,
        d_state: int = 64,
        d_conv: int = 4,
    ):
        super().__init__()

        self.mamba = Mamba2Block(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            headdim=headdim,
        )

        self.transformer = Transformer(
            d_model,
            num_heads=num_heads,
            dropout=dropout,
        )

    def forward(self, x: Tensor, y: Tensor, mask: Tensor):
        x = self.mamba(x)
        x = self.transformer(x, y, mask)
        return x


class Mamba2HybridSeq2Seq(EncoderDecoder):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        src_pad_token_id: int,
        tgt_pad_token_id: int,
        tgt_bos_token_id: int,
        tgt_eos_token_id: int,
        encoder_layers: int = 4,
        decoder_layers: int = 4,
        d_model: int = 512,
        num_heads: int = 16,
        dropout: float = 0.1,
        headdim: int = 64,
        d_state: int = 64,
        d_conv: int = 4,
        max_input_length: int = 4096,
    ):
        super().__init__(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            src_pad_token_id=src_pad_token_id,
            tgt_pad_token_id=tgt_pad_token_id,
            tgt_bos_token_id=tgt_bos_token_id,
            tgt_eos_token_id=tgt_eos_token_id,
        )

        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model, src_pad_token_id)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model, tgt_pad_token_id)
        self.positional_encoding = PositionalEncoding(d_model, max_input_length)
        self.norm = RMSNorm(d_model)
        self.head = nn.Linear(d_model, tgt_vocab_size, bias=False)

        self.encoder = nn.ModuleList(
            [
                Transformer(
                    d_model=d_model,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(encoder_layers)
            ]
        )

        self.decoder = nn.ModuleList(
            [
                Mamba2HybridBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    dropout=dropout,
                    headdim=headdim,
                    d_state=d_state,
                    d_conv=d_conv,
                )
                for _ in range(decoder_layers)
            ]
        )

    def encode(
        self,
        input_ids: Tensor,
        key_padding_mask: Tensor,
    ) -> Tensor:
        y = self.encoder_embedding(input_ids)
        y = self.positional_encoding(y)
        for layer in self.encoder:
            y = layer(y, y, key_padding_mask)
        return y

    def decode(
        self,
        output_ids: Tensor,
        encoder_outputs: Tensor,
        key_padding_mask: Tensor,
    ) -> Tensor:
        x = self.decoder_embedding(output_ids)
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
        max_length: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> Tensor:
        batch_size = input_ids.size(0)
        device = input_ids.device
        key_padding_mask = ~attention_mask.bool()
        encoder_outputs = self.encode(input_ids, key_padding_mask)

        sequences = torch.full(
            (batch_size, max_length),
            self.tgt_pad_token_id,
            dtype=torch.long,
            device=device,
        )

        sequences[:, 0] = self.tgt_bos_token_id
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for t in range(1, max_length):
            logits_t = self.decode(sequences[:, :t], encoder_outputs, key_padding_mask)[:, -1, :]
            next_token = self._sample_next_token(logits_t, temperature, top_p)

            active = ~finished
            sequences[active, t] = next_token[active]
            finished |= next_token == self.tgt_eos_token_id

            if finished.all():
                break

        return sequences
