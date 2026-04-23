import torch
import torch.nn as nn
from torch import BoolTensor, Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class RMSNorm(nn.RMSNorm):
    def forward(self, x: Tensor):
        dtype = x.dtype
        with torch.autocast(x.device.type, enabled=False):
            return super().forward(x.float()).to(dtype)


class FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_ff, d_model, bias=False)
        self.silu = nn.SiLU()
        self.norm = RMSNorm(d_model)

    def forward(self, x: torch.Tensor):
        x = self.norm(x)
        x = self.w3(self.silu(self.w1(x)) * self.w2(x))
        return x


class Attention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout, batch_first=True)
        self.x_norm = RMSNorm(d_model)
        self.y_norm = RMSNorm(d_model)

    def forward(self, x: Tensor, y: Tensor, mask: BoolTensor):
        x = self.x_norm(x)
        y = self.y_norm(y)
        x, _ = self.attention(x, y, y, key_padding_mask=mask)
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
    ):
        self.pad_token_id = pad_token_id
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        super().__init__()

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

    def forward(self, input_ids: Tensor, attention_mask: Tensor):
        embedding = self.embedding(input_ids)
        length = attention_mask.sum(dim=1).cpu()
        packed = pack_padded_sequence(
            embedding,
            length,
            batch_first=True,
            enforce_sorted=False,
        )

        outputs, (hidden, cell) = self.lstm(packed)
        outputs, _ = pad_packed_sequence(
            outputs,
            batch_first=True,
            total_length=input_ids.size(1),
        )

        # view bidirectional states as undirectional
        # (num_layers * 2, batch, hidden_size) -> (num_layers, batch, 2 * hidden_size)
        hidden = hidden.view(self.num_layers, 2, -1, self.hidden_size)
        hidden = hidden.permute(0, 2, 1, 3).flatten(2)
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
        super().__init__()

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

    def forward(self, input_id: Tensor, hidden: Tensor, cell: Tensor):
        embedding = self.embedding(input_id).unsqueeze(1)
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
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.2,
    ):
        self.tgt_pad_token_id = tgt_pad_token_id
        self.tgt_bos_token_id = tgt_bos_token_id
        self.tgt_eos_token_id = tgt_eos_token_id
        super().__init__()

        self.encoder = Encoder(
            vocab_size=src_vocab_size,
            pad_token_id=src_pad_token_id,
            num_layers=num_layers,
            hidden_size=hidden_size,
            embedding_size=embedding_size,
            dropout=dropout,
        )

        self.decoder = Decoder(
            vocab_size=tgt_vocab_size,
            pad_token_id=tgt_pad_token_id,
            num_layers=num_layers,
            hidden_size=hidden_size * 2,
            embedding_size=embedding_size,
            dropout=dropout,
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
        logits = self.head(decoder_outputs, encoder_outputs, ~attention_mask.bool())
        return logits

    def inference(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
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
            (batch_size,),
            self.tgt_bos_token_id,
            device=device,
            dtype=torch.long,
        )
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for t in range(1, max_length):
            decoder_output, (hidden, cell) = self.decoder(decoder_input, hidden, cell)
            logits = self.head(
                decoder_output,
                encoder_outputs,
                ~attention_mask.bool(),
            )
            next_token = logits.argmax(dim=2).squeeze(1)

            mask = ~finished
            sequences[mask, t] = next_token[mask]
            decoder_input = next_token
            finished |= next_token == self.tgt_eos_token_id

            if finished.all():
                break

        return sequences
