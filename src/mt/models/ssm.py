import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _next_power_of_two(n: int) -> int:
    return 1 << (n - 1).bit_length()


def causal_fft_conv1d(u: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    dtype = u.dtype
    _, _, length = u.shape
    n_fft = _next_power_of_two(2 * length - 1)

    u_f = torch.fft.rfft(u.float(), n=n_fft, dim=-1)
    k_f = torch.fft.rfft(k.float(), n=n_fft, dim=-1)

    y = torch.fft.irfft(u_f * k_f.unsqueeze(0), n=n_fft, dim=-1)
    return y[..., :length].to(dtype)


def reverse_padded_sequence(x: Tensor, lengths: Tensor) -> Tensor:
    batch_size, seq_len, dim = x.shape
    idx = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
    rev_idx = lengths.unsqueeze(1) - 1 - idx
    gather_idx = torch.where(idx < lengths.unsqueeze(1), rev_idx, idx).clamp_min(0)
    return x.gather(1, gather_idx.unsqueeze(-1).expand(-1, -1, dim))


def masked_mean(x: Tensor, mask: Tensor) -> Tensor:
    mask_f = mask.unsqueeze(-1).to(x.dtype)
    denom = mask_f.sum(dim=1).clamp_min(1.0)
    return (x * mask_f).sum(dim=1) / denom


class DiagonalSSMKernel(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_state: int,
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_state = n_state

        log_dt = torch.rand(d_model) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        self.log_dt = nn.Parameter(log_dt)

        self.log_A_real = nn.Parameter(torch.randn(d_model, n_state) - 1.0)

        a_im_init = math.pi * torch.arange(n_state, dtype=torch.float32)
        a_im_init = a_im_init.unsqueeze(0).repeat(d_model, 1)
        self.A_im = nn.Parameter(a_im_init)

        scale = n_state**-0.5
        self.B_re = nn.Parameter(torch.randn(d_model, n_state) * scale)
        self.B_im = nn.Parameter(torch.randn(d_model, n_state) * scale)
        self.C_re = nn.Parameter(torch.randn(d_model, n_state) * scale)
        self.C_im = nn.Parameter(torch.randn(d_model, n_state) * scale)

        self.D = nn.Parameter(torch.zeros(d_model))

    def forward(self, length: int) -> tuple[Tensor, Tensor]:
        dt = torch.exp(self.log_dt).unsqueeze(-1)
        a = torch.complex(-torch.exp(self.log_A_real), self.A_im)
        b = torch.complex(self.B_re, self.B_im)
        c = torch.complex(self.C_re, self.C_im)

        a_dt = dt * a
        b_bar = torch.expm1(a_dt) / a * b

        t = torch.arange(length, device=a.device, dtype=dt.dtype)
        vander = torch.exp(a_dt.unsqueeze(-1) * t)

        k = torch.sum((c * b_bar).unsqueeze(-1) * vander, dim=1).real
        return k, self.D


class S4D(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_state: int,
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
    ) -> None:
        super().__init__()
        self.kernel = DiagonalSSMKernel(
            d_model=d_model,
            n_state=n_state,
            dt_min=dt_min,
            dt_max=dt_max,
        )

    def forward(self, x: Tensor) -> Tensor:
        u = x.transpose(1, 2)
        k, d = self.kernel(u.size(-1))
        y = causal_fft_conv1d(u, k)
        y = y + d.view(1, -1, 1) * u
        return y.transpose(1, 2)


class S4DBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_state: int,
        dropout: float = 0.0,
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, 2 * d_model)
        self.ssm = S4D(
            d_model=d_model,
            n_state=n_state,
            dt_min=dt_min,
            dt_max=dt_max,
        )
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        residual = x

        z = self.in_proj(self.norm(x))
        u, gate = z.chunk(2, dim=-1)

        y = self.ssm(u)
        y = y * torch.sigmoid(gate)
        y = F.gelu(y)
        y = self.out_proj(y)
        y = self.dropout(y)

        return residual + y


class S4DStack(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_state: int,
        num_layers: int,
        dropout: float = 0.0,
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                S4DBlock(
                    d_model=d_model,
                    n_state=n_state,
                    dropout=dropout,
                    dt_min=dt_min,
                    dt_max=dt_max,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


class S4Seq2Seq(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        embedding_dim: int = 512,
        hidden_dim: int = 512,
        n_state: int = 64,
        num_layers: int = 6,
        dropout: float = 0.1,
        src_pad_token_id: int = 0,
        tgt_pad_token_id: int = 0,
        tgt_bos_token_id: int = 1,
        tgt_eos_token_id: int = 2,
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.src_pad_token_id = src_pad_token_id
        self.tgt_pad_token_id = tgt_pad_token_id
        self.tgt_bos_token_id = tgt_bos_token_id
        self.tgt_eos_token_id = tgt_eos_token_id

        self.encoder_emb = nn.Embedding(
            src_vocab_size,
            embedding_dim,
            padding_idx=src_pad_token_id,
        )
        self.decoder_emb = nn.Embedding(
            tgt_vocab_size,
            embedding_dim,
            padding_idx=tgt_pad_token_id,
        )

        self.encoder_dropout = nn.Dropout(dropout)
        self.decoder_dropout = nn.Dropout(dropout)

        self.encoder_in = nn.Linear(embedding_dim, hidden_dim)
        self.decoder_in = nn.Linear(embedding_dim, hidden_dim)

        self.encoder_forward = S4DStack(
            d_model=hidden_dim,
            n_state=n_state,
            num_layers=num_layers,
            dropout=dropout,
            dt_min=dt_min,
            dt_max=dt_max,
        )
        self.encoder_backward = S4DStack(
            d_model=hidden_dim,
            n_state=n_state,
            num_layers=num_layers,
            dropout=dropout,
            dt_min=dt_min,
            dt_max=dt_max,
        )
        self.encoder_norm = nn.LayerNorm(2 * hidden_dim)

        self.decoder = S4DStack(
            d_model=hidden_dim,
            n_state=n_state,
            num_layers=num_layers,
            dropout=dropout,
            dt_min=dt_min,
            dt_max=dt_max,
        )
        self.decoder_context_proj = nn.Linear(2 * hidden_dim, hidden_dim)

        self.output = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, tgt_vocab_size),
        )

    def encode(self, input_ids: Tensor, attention_mask: Tensor) -> tuple[Tensor, Tensor]:
        lengths = attention_mask.long().sum(dim=1)

        x = self.encoder_emb(input_ids)
        x = self.encoder_dropout(x)
        x = self.encoder_in(x)

        x_fwd = self.encoder_forward(x)

        x_rev = reverse_padded_sequence(x, lengths)
        x_bwd = self.encoder_backward(x_rev)
        x_bwd = reverse_padded_sequence(x_bwd, lengths)

        encoder_outputs = torch.cat([x_fwd, x_bwd], dim=-1)
        encoder_outputs = self.encoder_norm(encoder_outputs)
        encoder_outputs = encoder_outputs * attention_mask.unsqueeze(-1).to(encoder_outputs.dtype)

        context = masked_mean(encoder_outputs, attention_mask)
        return encoder_outputs, context

    def decode(self, output_ids: Tensor, context: Tensor) -> Tensor:
        x = self.decoder_emb(output_ids)
        x = self.decoder_dropout(x)
        x = self.decoder_in(x)

        x = x + self.decoder_context_proj(context).unsqueeze(1)
        x = self.decoder(x)

        context_expanded = context.unsqueeze(1).expand(-1, x.size(1), -1)
        logits = self.output(torch.cat([x, context_expanded], dim=-1))
        return logits

    def decode_step(self, output_ids: Tensor, context: Tensor) -> Tensor:
        logits = self.decode(output_ids, context)
        return logits[:, -1]

    def forward(self, input_ids: Tensor, output_ids: Tensor, attention_mask: Tensor) -> Tensor:
        _, context = self.encode(input_ids, attention_mask)

        decoder_input_ids = output_ids[:, :-1]
        logits = self.decode(decoder_input_ids, context)

        # (batch, tgt_len - 1, tgt_vocab_size)
        return logits

    @torch.no_grad()
    def inference(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        max_length: int = 100,
    ) -> Tensor:
        batch_size = input_ids.size(0)
        device = input_ids.device

        _, context = self.encode(input_ids, attention_mask)

        sequences = torch.full(
            (batch_size, max_length),
            self.tgt_pad_token_id,
            dtype=torch.long,
            device=device,
        )
        sequences[:, 0] = self.tgt_bos_token_id

        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for t in range(1, max_length):
            logits_t = self.decode_step(sequences[:, :t], context)
            next_token = logits_t.argmax(dim=-1)

            active = ~finished
            sequences[active, t] = next_token[active]

            finished |= next_token == self.tgt_eos_token_id
            if finished.all():
                break

        return sequences
