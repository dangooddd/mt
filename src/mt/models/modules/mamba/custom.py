import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..base import EncoderDecoder


def reverse_padded_sequence(x: Tensor, lengths: Tensor) -> Tensor:
    batch_size, seq_len, dim = x.shape
    idx = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
    rev_idx = lengths.unsqueeze(1) - 1 - idx
    gather_idx = torch.where(idx < lengths.unsqueeze(1), rev_idx, idx).clamp_min(0)
    return x.gather(1, gather_idx.unsqueeze(-1).expand(-1, -1, dim))


def apply_padding_mask(x: Tensor, mask: Tensor | None) -> Tensor:
    if mask is None:
        return x
    return x * mask.unsqueeze(-1).to(x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class FeedForward(nn.Module):
    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class MambaMixer(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: int | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = expand * d_model
        self.dt_rank = math.ceil(d_model / 16) if dt_rank is None else dt_rank

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            bias=True,
        )

        self.x_proj = nn.Linear(
            self.d_inner,
            self.dt_rank + 2 * d_state,
            bias=False,
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        self.A_log = nn.Parameter(torch.randn(self.d_inner, d_state))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.xavier_uniform_(self.x_proj.weight)
        nn.init.xavier_uniform_(self.dt_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

        conv_std = self.d_conv**-0.5
        nn.init.uniform_(self.conv1d.weight, -conv_std, conv_std)
        if self.conv1d.bias is not None:
            nn.init.zeros_(self.conv1d.bias)

        nn.init.normal_(self.A_log, mean=0.0, std=0.02)
        nn.init.ones_(self.D)

        with torch.no_grad():
            self.dt_proj.bias.fill_(-2.0)

    def _causal_depthwise_conv(self, x: Tensor) -> Tensor:
        x = x.transpose(1, 2)
        length = x.size(-1)
        x = self.conv1d(x)[..., :length]
        x = x.transpose(1, 2)
        return F.silu(x)

    def _selective_scan(self, u: Tensor, delta: Tensor, b: Tensor, c: Tensor) -> Tensor:
        batch_size, seq_len, d_inner = u.shape
        dtype = u.dtype

        u_f = u.float()
        delta_f = delta.float()
        b_f = b.float()
        c_f = c.float()

        a = -torch.exp(self.A_log.float())
        d = self.D.float()

        state = u.new_zeros(batch_size, d_inner, self.d_state, dtype=torch.float32)
        ys = []

        for t in range(seq_len):
            u_t = u_f[:, t]
            delta_t = delta_f[:, t]
            b_t = b_f[:, t]
            c_t = c_f[:, t]

            delta_a = torch.exp(delta_t.unsqueeze(-1) * a.unsqueeze(0))
            delta_b_u = delta_t.unsqueeze(-1) * u_t.unsqueeze(-1) * b_t.unsqueeze(1)

            state = delta_a * state + delta_b_u
            y_t = (state * c_t.unsqueeze(1)).sum(dim=-1) + d.unsqueeze(0) * u_t
            ys.append(y_t)

        y = torch.stack(ys, dim=1)
        return y.to(dtype)

    def forward(self, x: Tensor) -> Tensor:
        xz = self.in_proj(x)
        x_ssm, z = xz.chunk(2, dim=-1)

        x_ssm = self._causal_depthwise_conv(x_ssm)

        ssm_params = self.x_proj(x_ssm)
        delta_raw, b, c = torch.split(
            ssm_params,
            [self.dt_rank, self.d_state, self.d_state],
            dim=-1,
        )

        delta = F.softplus(self.dt_proj(delta_raw))
        y = self._selective_scan(x_ssm, delta, b, c)

        y = y * F.silu(z)
        return self.out_proj(y)


class MambaBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.mixer = MambaMixer(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dt_rank=dt_rank,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.dropout(self.mixer(self.norm(x)))


class MambaEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int,
        d_conv: int,
        expand: int,
        dt_rank: int | None,
        dropout: float,
    ) -> None:
        super().__init__()
        self.mamba = MambaBlock(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dt_rank=dt_rank,
            dropout=dropout,
        )
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = FeedForward(d_model, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, padding_mask: Tensor | None = None) -> Tensor:
        x = self.mamba(x)
        x = apply_padding_mask(x, padding_mask)

        x = x + self.dropout(self.ffn(self.ffn_norm(x)))
        x = apply_padding_mask(x, padding_mask)
        return x


class MambaDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        encoder_dim: int,
        num_heads: int,
        d_state: int,
        d_conv: int,
        expand: int,
        dt_rank: int | None,
        dropout: float,
    ) -> None:
        super().__init__()
        self.self_mamba = MambaBlock(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dt_rank=dt_rank,
            dropout=dropout,
        )

        self.cross_norm = RMSNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            kdim=encoder_dim,
            vdim=encoder_dim,
            batch_first=True,
        )

        self.ffn_norm = RMSNorm(d_model)
        self.ffn = FeedForward(d_model, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        encoder_outputs: Tensor,
        encoder_mask: Tensor,
        decoder_padding_mask: Tensor | None = None,
    ) -> Tensor:
        x = self.self_mamba(x)
        x = apply_padding_mask(x, decoder_padding_mask)

        q = self.cross_norm(x)
        attn_out, _ = self.cross_attn(
            q,
            encoder_outputs,
            encoder_outputs,
            key_padding_mask=~encoder_mask.bool(),
            need_weights=False,
        )
        x = x + self.dropout(attn_out)
        x = apply_padding_mask(x, decoder_padding_mask)

        x = x + self.dropout(self.ffn(self.ffn_norm(x)))
        x = apply_padding_mask(x, decoder_padding_mask)
        return x


class MambaEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_layers: int,
        d_state: int,
        d_conv: int,
        expand: int,
        dt_rank: int | None,
        dropout: float,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                MambaEncoderLayer(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dt_rank=dt_rank,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = RMSNorm(d_model)

    def forward(self, x: Tensor, padding_mask: Tensor | None = None) -> Tensor:
        for layer in self.layers:
            x = layer(x, padding_mask)
        x = self.norm(x)
        x = apply_padding_mask(x, padding_mask)
        return x


class MambaDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        encoder_dim: int,
        num_layers: int,
        num_heads: int,
        d_state: int,
        d_conv: int,
        expand: int,
        dt_rank: int | None,
        dropout: float,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                MambaDecoderLayer(
                    d_model=d_model,
                    encoder_dim=encoder_dim,
                    num_heads=num_heads,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dt_rank=dt_rank,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = RMSNorm(d_model)

    def forward(
        self,
        x: Tensor,
        encoder_outputs: Tensor,
        encoder_mask: Tensor,
        decoder_padding_mask: Tensor | None = None,
    ) -> Tensor:
        for layer in self.layers:
            x = layer(
                x,
                encoder_outputs=encoder_outputs,
                encoder_mask=encoder_mask,
                decoder_padding_mask=decoder_padding_mask,
            )
        x = self.norm(x)
        x = apply_padding_mask(x, decoder_padding_mask)
        return x


class MambaSeq2Seq(EncoderDecoder):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        embedding_dim: int = 512,
        hidden_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: int | None = None,
        dropout: float = 0.1,
        src_pad_token_id: int = 0,
        tgt_pad_token_id: int = 0,
        tgt_bos_token_id: int = 1,
        tgt_eos_token_id: int = 2,
    ) -> None:
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

        self.encoder_in = nn.Linear(embedding_dim, hidden_dim, bias=False)
        self.decoder_in = nn.Linear(embedding_dim, hidden_dim, bias=False)

        self.encoder_fwd = MambaEncoder(
            d_model=hidden_dim,
            num_layers=num_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dt_rank=dt_rank,
            dropout=dropout,
        )
        self.encoder_bwd = MambaEncoder(
            d_model=hidden_dim,
            num_layers=num_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dt_rank=dt_rank,
            dropout=dropout,
        )
        self.encoder_norm = RMSNorm(2 * hidden_dim)

        self.decoder = MambaDecoder(
            d_model=hidden_dim,
            encoder_dim=2 * hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dt_rank=dt_rank,
            dropout=dropout,
        )

        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, tgt_vocab_size),
        )

    def encode(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        src_mask = attention_mask.bool()
        lengths = src_mask.long().sum(dim=1)

        x = self.encoder_emb(input_ids)
        x = self.encoder_in(x)
        x = apply_padding_mask(x, src_mask)

        x_fwd = self.encoder_fwd(x, src_mask)

        x_rev = reverse_padded_sequence(x, lengths)
        x_bwd = self.encoder_bwd(x_rev, src_mask)
        x_bwd = reverse_padded_sequence(x_bwd, lengths)

        encoder_outputs = torch.cat([x_fwd, x_bwd], dim=-1)
        encoder_outputs = self.encoder_norm(encoder_outputs)
        encoder_outputs = apply_padding_mask(encoder_outputs, src_mask)
        return encoder_outputs

    def decode(
        self,
        output_ids: Tensor,
        encoder_outputs: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        src_mask = attention_mask.bool()
        tgt_mask = output_ids.ne(self.tgt_pad_token_id)

        x = self.decoder_emb(output_ids)
        x = self.decoder_in(x)
        x = apply_padding_mask(x, tgt_mask)

        x = self.decoder(
            x,
            encoder_outputs=encoder_outputs,
            encoder_mask=src_mask,
            decoder_padding_mask=tgt_mask,
        )
        x = apply_padding_mask(x, tgt_mask)

        return self.output(x)

    def decode_step(
        self,
        output_ids: Tensor,
        encoder_outputs: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        logits = self.decode(output_ids, encoder_outputs, attention_mask)
        return logits[:, -1]

    def forward(
        self,
        input_ids: Tensor,
        output_ids: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        encoder_outputs = self.encode(input_ids, attention_mask)

        logits = self.decode(
            output_ids,
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
        )

        # (batch, tgt_len, tgt_vocab_size)
        return logits

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

        encoder_outputs = self.encode(input_ids, attention_mask)

        sequences = torch.full(
            (batch_size, max_length),
            self.tgt_pad_token_id,
            dtype=torch.long,
            device=device,
        )
        sequences[:, 0] = self.tgt_bos_token_id

        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for t in range(1, max_length):
            logits_t = self.decode_step(
                sequences[:, :t],
                encoder_outputs=encoder_outputs,
                attention_mask=attention_mask,
            )
            next_token = self._sample_next_token(logits_t, temperature)

            active = ~finished
            sequences[active, t] = next_token[active]

            finished |= next_token == self.tgt_eos_token_id
            if finished.all():
                break

        return sequences
