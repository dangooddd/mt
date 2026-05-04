import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from ..base import EncoderDecoder


class LuongAttention(nn.Module):
    def __init__(self, encoder_dim: int, decoder_dim: int):
        super().__init__()
        self.key_proj = nn.Linear(encoder_dim, decoder_dim, bias=False)

    def forward(
        self,
        decoder_hidden: Tensor,  # (batch, decoder_dim)
        encoder_outputs: Tensor,  # (batch, src_len, encoder_dim)
        mask: Tensor,  # (batch, src_len)
    ):
        # project encoder outputs for scoring only:
        # (batch, src_len, encoder_dim) -> (batch, src_len, decoder_dim)
        keys = self.key_proj(encoder_outputs)  # (batch, src_len, decoder_dim)

        # compute scores: (batch, src_len)
        scores = torch.bmm(
            keys,
            decoder_hidden.unsqueeze(2),  # (batch, decoder_dim, 1)
        ).squeeze(2)

        # apply mask
        neg_inf = torch.finfo(scores.dtype).min
        scores = scores.masked_fill(~mask, neg_inf)

        # attention weights
        attn_weights = torch.softmax(scores, dim=1)  # (batch, src_len)

        # context vector over original encoder outputs
        context = torch.bmm(
            attn_weights.unsqueeze(1),  # (batch, 1, src_len)
            encoder_outputs,  # (batch, src_len, encoder_dim)
        ).squeeze(1)  # (batch, encoder_dim)

        return context, attn_weights


class LuongSeq2Seq(EncoderDecoder):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        embedding_dim: int = 1000,
        hidden_dim: int = 1000,
        num_layers: int = 4,
        dropout: float = 0.3,
        src_pad_token_id: int = 0,
        tgt_pad_token_id: int = 0,
        tgt_bos_token_id: int = 1,
        tgt_eos_token_id: int = 2,
    ):
        super().__init__(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            src_pad_token_id=src_pad_token_id,
            tgt_pad_token_id=tgt_pad_token_id,
            tgt_bos_token_id=tgt_bos_token_id,
            tgt_eos_token_id=tgt_eos_token_id,
        )
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.encoder_emb = nn.Sequential(
            nn.Embedding(
                src_vocab_size,
                embedding_dim,
                padding_idx=src_pad_token_id,
            ),
            nn.Dropout(dropout),
        )

        self.decoder_emb = nn.Sequential(
            nn.Embedding(
                tgt_vocab_size,
                embedding_dim,
                padding_idx=tgt_pad_token_id,
            ),
            nn.Dropout(dropout),
        )

        self.encoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True,
        )
        self.encoder_norm = nn.LayerNorm(hidden_dim * 2)

        self.hidden_proj = nn.Linear(2 * hidden_dim, hidden_dim)
        self.cell_proj = nn.Linear(2 * hidden_dim, hidden_dim)

        self.decoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )
        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.attention = LuongAttention(2 * hidden_dim, hidden_dim)

        self.output = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, tgt_vocab_size),
        )

        self.init_weights()

    def init_weights(self):
        for name, p in self.named_parameters():
            if name == "encoder_emb.0.weight":
                nn.init.normal_(p, mean=0, std=0.01)
                p.data[self.src_pad_token_id].zero_()
            elif name == "decoder_emb.0.weight":
                nn.init.normal_(p, mean=0, std=0.01)
                p.data[self.tgt_pad_token_id].zero_()
            elif p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, input_ids: Tensor, attention_mask: Tensor):
        lengths = attention_mask.sum(dim=1)
        emb = self.encoder_emb(input_ids)  # (batch, src_len, embedding_dim)

        packed = pack_padded_sequence(
            emb,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        encoder_outputs, (hidden, cell) = self.encoder(packed)
        encoder_outputs, _ = pad_packed_sequence(
            encoder_outputs,
            batch_first=True,
            total_length=input_ids.size(1),
        )
        encoder_outputs = self.encoder_norm(encoder_outputs)

        # project bidirectional states to unidirectional for decoder
        # (num_layers * 2, batch, hidden_dim) -> (num_layers, batch, 2 * hidden_dim)
        hidden = hidden.view(self.num_layers, 2, -1, self.hidden_dim)
        hidden = hidden.permute(0, 2, 1, 3).flatten(2)
        cell = cell.view(self.num_layers, 2, -1, self.hidden_dim)
        cell = cell.permute(0, 2, 1, 3).flatten(2)

        hidden = self.hidden_proj(hidden)
        cell = self.cell_proj(cell)

        return encoder_outputs, (hidden, cell)

    def decode_step(
        self,
        input_id: Tensor,
        hidden: Tensor,
        cell: Tensor,
        encoder_outputs: Tensor,
        attention_mask: Tensor,
    ):
        # embedding
        emb = self.decoder_emb(input_id).unsqueeze(1)  # (batch, 1, embedding_dim)

        # decoder
        decoder_output, (new_hidden, new_cell) = self.decoder(emb, (hidden, cell))
        decoder_output = self.decoder_norm(decoder_output)
        decoder_hidden = decoder_output.squeeze(1)  # (batch, hidden_dim)

        # attention
        context, _ = self.attention(decoder_hidden, encoder_outputs, attention_mask)

        # combine context and decoder hidden
        combined = torch.cat([decoder_hidden, context], dim=1)  # (batch, hidden_dim * 2)

        logits = self.output(combined)
        return logits, new_hidden, new_cell

    def forward(self, input_ids: Tensor, output_ids: Tensor, attention_mask: Tensor) -> Tensor:
        encoder_outputs, (hidden, cell) = self.encode(input_ids, attention_mask)

        _, output_length = output_ids.size()
        preds = []

        for t in range(output_length):
            input_id = output_ids[:, t]
            logits, hidden, cell = self.decode_step(
                input_id,
                hidden,
                cell,
                encoder_outputs,
                attention_mask,
            )
            preds.append(logits)

        # stack along time dimension: (batch, tgt_len, tgt_vocab_size)
        return torch.stack(preds, dim=1)

    @torch.no_grad()
    def inference(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        max_length: int = 100,
        temperature: float = 0.0,
    ) -> Tensor:
        batch_size = input_ids.size(0)
        device = input_ids.device

        encoder_outputs, (hidden, cell) = self.encode(input_ids, attention_mask)

        sequences = torch.full(
            (batch_size, max_length),
            self.tgt_pad_token_id,
            dtype=torch.long,
            device=device,
        )
        sequences[:, 0] = self.tgt_bos_token_id

        decoder_input = torch.full(
            (batch_size, 1),
            self.tgt_bos_token_id,
            dtype=torch.long,
            device=device,
        )
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for t in range(1, max_length):
            logits, hidden, cell = self.decode_step(
                decoder_input.squeeze(1), hidden, cell, encoder_outputs, attention_mask
            )

            next_token = self._sample_next_token(logits, temperature).unsqueeze(1)  # (batch, 1)

            mask = ~finished
            sequences[mask, t] = next_token[mask, 0]
            decoder_input = next_token

            finished |= next_token.squeeze(1) == self.tgt_eos_token_id

            if finished.all():
                break

        return sequences
