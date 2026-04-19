from typing import cast

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_token_id: int,
        num_layers: int,
        hidden_size: int,
        embedding_size: int,
        dropout: float,
    ):
        self.pad_token_id = pad_token_id
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.embedding = nn.Sequential(
            nn.Embedding(
                vocab_size,
                embedding_size,
                padding_idx=pad_token_id,
            ),
            nn.Dropout(dropout),
        )

        self.lstm = nn.LSTM(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True,
        )

        self.norm = nn.RMSNorm(hidden_size * 2)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        embedding = self.embedding(input_ids)
        length = attention_mask.sum(dim=1)
        packed = pack_padded_sequence(
            embedding,
            length,
            batch_first=True,
            enforce_sorted=False,
        )

        outputs, (hidden, cell) = self.lstm(packed)
        outputs = pad_packed_sequence(
            outputs,
            batch_first=True,
            total_length=input_ids.size(1),
        )
        outputs = self.norm(outputs)

        # view bidirectional states as undirectional
        # (num_layers * 2, batch, hidden_size) -> (num_layers, batch, 2 * hidden_size)
        hidden = cast(torch.Tensor, hidden)
        hidden = hidden.view(self.num_layers, 2, -1, self.hidden_size)
        hidden = hidden.permute(0, 2, 1, 3).flatten(2)

        cell = cast(torch.Tensor, cell)
        cell = cell.view(self.num_layers, 2, -1, self.hidden_size)
        cell = cell.permute(0, 2, 1, 3).flatten(2)

        return outputs, (hidden, cell)


class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_token_id: int,
        num_layers: int,
        hidden_size: int,
        embedding_size: int,
        dropout: float,
    ):
        self.embedding = nn.Sequential(
            nn.Embedding(
                vocab_size,
                embedding_size,
                padding_idx=pad_token_id,
            ),
            nn.Dropout(dropout),
        )

        self.lstm = nn.LSTM(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )

        self.norm = nn.RMSNorm(hidden_size)

    def forward(
        self,
        input_id: torch.Tensor,
        hidden: torch.Tensor,
        cell: torch.Tensor,
    ):
        embedding = cast(torch.Tensor, self.embedding(input_id).unsqueeze(1))
        output, (hidden, cell) = self.lstm(embedding, (hidden, cell))
        return output, (hidden, cell)


class LstmSeq2Seq(nn.Module):
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
        encoder_layers: int = 4,
        decoder_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.25,
    ):
        self.tgt_pad_token_id = tgt_pad_token_id
        self.tgt_bos_token_id = tgt_bos_token_id
        self.tgt_eos_token_id = tgt_eos_token_id

        self.encoder = Encoder(
            vocab_size=src_vocab_size,
            pad_token_id=src_pad_token_id,
            num_layers=encoder_layers,
            hidden_size=hidden_size,
            embedding_size=embedding_size,
            dropout=dropout,
        )

        self.decoder = Decoder(
            vocab_size=tgt_vocab_size,
            pad_token_id=tgt_pad_token_id,
            num_layers=decoder_layers,
            hidden_size=hidden_size * 2,
            embedding_size=embedding_size,
            dropout=dropout,
        )

        self.attention = nn.MultiheadAttention(
            hidden_size * 2,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.head = nn.Sequential(
            nn.RMSNorm(hidden_size * 4),
            nn.Linear(hidden_size * 4, tgt_vocab_size),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        output_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        encoder_outputs, (hidden, cell) = self.encoder(input_ids, attention_mask)
        output_length = output_ids.size(1)
        decoder_steps = []
        decoder_input = output_ids[:, 0]

        for t in range(1, output_length):
            decoder_output, (hidden, cell) = self.decoder(decoder_input, hidden, cell)
            decoder_steps.append(decoder_output)
            decoder_input = output_ids[:, t]

        # (batch, time, hidden_size * 2)
        decoder_outputs = torch.cat(decoder_steps, dim=1)
        context, _ = self.attention(
            decoder_outputs,
            encoder_outputs,
            encoder_outputs,
            key_padding_mask=attention_mask,
        )

        combined = torch.cat([decoder_outputs, context], dim=2)
        return self.head(combined)

    def inference(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_length: int = 128,
    ):
        batch_size = input_ids.size(0)
        device = input_ids.device
        encoder_outputs, (hidden, cell) = self.encoder(input_ids, attention_mask)

        sequences = torch.full(
            (batch_size, max_length),
            self.tgt_pad_token_id,
            device=device,
            dtype=torch.long,
        )
        sequences[:, 0] = self.tgt_bos_token_id

        decoder_input = torch.full(
            (batch_size, 1),
            self.tgt_bos_token_id,
            device=device,
            dtype=torch.long,
        )
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for t in range(1, max_length):
            decoder_output, (hidden, cell) = self.decoder(decoder_input, hidden, cell)
            context, _ = self.attention(
                decoder_output,
                encoder_outputs,
                encoder_outputs,
                key_padding_mask=attention_mask,
            )
            combined = torch.cat([decoder_output, context], dim=2)
            logits = self.head(combined)
            next_token = logits.argmax(dim=-1, keepdim=True)

            mask = ~finished
            sequences[mask, t] = next_token[mask, 0]
            decoder_input = next_token
            finished |= next_token.squeeze(1) == self.tgt_eos_token_id

            if finished.all():
                break

        return sequences
