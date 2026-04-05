from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LuongAttention(nn.Module):
    """Luong attention with learnable matrix."""

    def __init__(self, encoder_dim: int, decoder_dim: int):
        super().__init__()
        self.linear = nn.Linear(encoder_dim, decoder_dim, bias=False)

    def forward(
        self,
        decoder_hidden: Tensor,  # (batch, decoder_dim)
        encoder_outputs: Tensor,  # (batch, src_len, encoder_dim)
        mask: Tensor,  # (batch, src_len)
    ) -> Tuple[Tensor, Tensor]:
        # Project encoder outputs: (batch, src_len, encoder_dim) -> (batch, src_len, decoder_dim)
        encoder_proj = self.linear(encoder_outputs)  # (batch, src_len, decoder_dim)

        # Compute scores: (batch, src_len)
        scores = torch.bmm(
            encoder_proj,
            decoder_hidden.unsqueeze(2),  # (batch, decoder_dim, 1)
        ).squeeze(2)

        # Apply mask
        neg_inf = torch.finfo(scores.dtype).min
        scores = scores.masked_fill(~mask, neg_inf)

        # Attention weights
        attn_weights = torch.softmax(scores, dim=1)  # (batch, src_len)

        # Context vector
        context = torch.bmm(
            attn_weights.unsqueeze(1),  # (batch, 1, src_len)
            encoder_proj,  # (batch, src_len, decoder_dim)
        ).squeeze(1)  # (batch, decoder_dim)

        return context, attn_weights


class LuongSeq2Seq(nn.Module):
    """Seq2Seq model with BiLSTM encoder, Luong attention, and LayerNorm."""

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
    ):
        super().__init__()
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.src_pad_token_id = src_pad_token_id
        self.tgt_pad_token_id = tgt_pad_token_id

        self.src_embedding = nn.Sequential(
            nn.Embedding(
                src_vocab_size,
                embedding_dim,
                padding_idx=src_pad_token_id,
            ),
            nn.Dropout(dropout),
        )

        self.tgt_embedding = nn.Sequential(
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
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, tgt_vocab_size),
        )

        self.init_weights()

    def init_weights(self):
        for name, p in self.named_parameters():
            if "embedding" in name and ("src" in name or "tgt" in name):
                nn.init.normal_(p, mean=0, std=0.01)
                if "src" in name:
                    p.data[self.src_pad_token_id].zero_()
                else:
                    p.data[self.tgt_pad_token_id].zero_()
            elif p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(
        self,
        src: Tensor,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor], Tensor]:
        """Encode source sequence.
        Returns:
            encoder_outputs: (batch, src_len, 2 * hidden_dim)  # bidirectional outputs
            (hidden, cell): tuple of tensors each (num_layers, batch, hidden_dim)
        """
        src_mask = src != self.src_pad_token_id
        src_lengths = src_mask.sum(dim=1)

        src_emb = self.src_embedding(src)  # (batch, src_len, embedding_dim)

        packed = pack_padded_sequence(
            src_emb,
            src_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        encoder_outputs, (hidden, cell) = self.encoder(packed)
        encoder_outputs, _ = pad_packed_sequence(encoder_outputs, batch_first=True)
        encoder_outputs = self.encoder_norm(encoder_outputs)

        # Project bidirectional states to unidirectional for decoder
        # (num_layers * 2, batch, hidden_dim) -> (num_layers, batch, 2 * hidden_dim)
        hidden = hidden.view(self.num_layers, 2, -1, self.hidden_dim)
        hidden = hidden.permute(0, 2, 1, 3).flatten(2)
        cell = cell.view(self.num_layers, 2, -1, self.hidden_dim)
        cell = cell.permute(0, 2, 1, 3).flatten(2)

        hidden = self.hidden_proj(hidden)
        cell = self.cell_proj(cell)

        return encoder_outputs, (hidden, cell), src_mask

    def decode_step(
        self,
        input_token: Tensor,  # (batch,)
        hidden: Tensor,
        cell: Tensor,
        encoder_outputs: Tensor,
        src_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Single decoder step.
        Returns:
            logits: (batch, tgt_vocab_size)
            new_hidden: (num_layers, batch, hidden_dim)
            new_cell: (num_layers, batch, hidden_dim)
        """
        # Embedding
        emb = self.tgt_embedding(input_token).unsqueeze(1)  # (batch, 1, embedding_dim)

        # Decoder LSTM
        decoder_output, (new_hidden, new_cell) = self.decoder(emb, (hidden, cell))
        decoder_output = self.decoder_norm(decoder_output)
        decoder_hidden = decoder_output.squeeze(1)  # (batch, hidden_dim)

        # Attention
        context, _ = self.attention(decoder_hidden, encoder_outputs, src_mask)

        # Combine context and decoder hidden
        combined = torch.cat(
            [decoder_hidden, context], dim=1
        )  # (batch, hidden_dim * 2)

        logits = self.output(combined)

        return logits, new_hidden, new_cell

    def forward(self, src: Tensor, tgt: Tensor) -> Tensor:
        encoder_outputs, (hidden, cell), src_mask = self.encode(src)

        batch_size, tgt_len = tgt.size()
        outputs = []
        input_token = tgt[:, 0]  # (batch,)

        for t in range(1, tgt_len):
            logits, hidden, cell = self.decode_step(
                input_token, hidden, cell, encoder_outputs, src_mask
            )
            outputs.append(logits)
            input_token = tgt[:, t]

        # Stack along time dimension: (batch, tgt_len-1, tgt_vocab_size)
        return torch.stack(outputs, dim=1)

    def inference(
        self,
        src: Tensor,
        max_len: int = 100,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
    ) -> Tensor:
        batch_size = src.size(0)
        device = src.device

        encoder_outputs, (hidden, cell), src_mask = self.encode(src)

        sequences = torch.full(
            (batch_size, max_len),
            self.tgt_pad_token_id,
            dtype=torch.long,
            device=device,
        )
        sequences[:, 0] = bos_token_id

        decoder_input = torch.full(
            (batch_size, 1),
            bos_token_id,
            dtype=torch.long,
            device=device,
        )
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for t in range(1, max_len):
            logits, hidden, cell = self.decode_step(
                decoder_input.squeeze(1), hidden, cell, encoder_outputs, src_mask
            )

            next_token = logits.argmax(dim=-1, keepdim=True)  # (batch, 1)

            mask = ~finished
            sequences[mask, t] = next_token[mask, 0]
            decoder_input = next_token

            finished |= next_token.squeeze(1) == eos_token_id

            if finished.all():
                break

        return sequences
