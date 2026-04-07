import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SSMBlock(nn.Module):
    """
    x_t[h, n] = A[h, n] * x_{t-1}[h, n] + B[h, n] * u_t[h]
    y_t[h] = sum_n C[h, n] * x_t[h, n] + D[h] * u_t[h]
    """

    def __init__(self, model_dim: int, state_dim: int, dropout: float = 0.1):
        super().__init__()
        self.model_dim = model_dim
        self.state_dim = state_dim

        self.in_proj = nn.Linear(model_dim, model_dim)

        self.A_log = nn.Parameter(torch.zeros(model_dim, state_dim))
        self.B = nn.Parameter(torch.randn(model_dim, state_dim) * 0.02)
        self.C = nn.Parameter(torch.randn(model_dim, state_dim) * 0.02)
        self.D = nn.Parameter(torch.ones(model_dim))

        self.out_proj = nn.Linear(model_dim, model_dim)
        self.norm = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)

    @property
    def A(self) -> Tensor:
        # stable diagonal decay
        return torch.exp(-torch.exp(self.A_log))

    def convolution_kernel(self, length: int, *, device: torch.device) -> Tensor:
        A = self.A
        B = self.B
        C = self.C
        powers = torch.arange(length, device=device)

        # for broadcasting
        # A.unsqueeze: [H, N] -> [1, H, N]
        # powers.view: [S] -> [S, 1, 1]
        A_powers = A.unsqueeze(0) ** powers.view(length, 1, 1)

        kernel = (A_powers * B.unsqueeze(0) * C.unsqueeze(0)).sum(dim=-1)
        return kernel.transpose(0, 1).contiguous()

    def forward(self, u: Tensor, mask: Tensor | None = None) -> Tensor:
        """Convolutional mode for training.

        Args:
            u: (batch, seq_len, model_dim)
            mask: (batch, seq_len) with True for valid positions
        Returns:
            (batch, seq_len, model_dim)
        """
        if mask is not None:
            u = u * mask.unsqueeze(-1).to(u.dtype)

        u = self.in_proj(u)
        _, seq_len, model_dim = u.shape
        x = u.transpose(1, 2)

        kernel = self.convolution_kernel(seq_len, device=u.device)
        weight = kernel.flip(-1).unsqueeze(1)

        y = F.conv1d(x, weight, groups=model_dim, padding=seq_len - 1)[..., :seq_len]
        y = y + self.D.view(1, -1, 1) * x
        y = y.transpose(1, 2)
        y = self.out_proj(y)
        y = self.norm(self.dropout(y) + u)

        if mask is not None:
            y = y * mask.unsqueeze(-1).to(y.dtype)

        return y

    def init_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        return torch.zeros(batch_size, self.model_dim, self.state_dim, device=device, dtype=dtype)

    def step(self, u: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
        """Recurrent mode for autoregressive decoding.

        Args:
            u: (batch, model_dim)
            state: (batch, model_dim, state_dim)
        Returns:
            y: (batch, model_dim)
            new_state: (batch, model_dim, state_dim)
        """
        u = self.in_proj(u)

        A = self.A.unsqueeze(0)
        B = self.B.unsqueeze(0)
        C = self.C.unsqueeze(0)
        D = self.D.unsqueeze(0)

        new_state = A * state + B * u.unsqueeze(-1)
        y = (C * new_state).sum(dim=-1) + D * u
        y = self.out_proj(y)
        y = self.norm(self.dropout(y) + u)
        return y, new_state


class SSMSeq2Seq(nn.Module):
    """Seq2Seq model with SSM encoder/decoder and no attention."""

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        embedding_dim: int = 512,
        hidden_dim: int = 512,
        num_layers: int = 4,
        dropout: float = 0.2,
        src_pad_token_id: int = 0,
        tgt_pad_token_id: int = 0,
        state_dim: int = 64,
    ):
        super().__init__()
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.state_dim = state_dim
        self.src_pad_token_id = src_pad_token_id
        self.tgt_pad_token_id = tgt_pad_token_id

        self.src_embedding = nn.Sequential(
            nn.Embedding(src_vocab_size, embedding_dim, padding_idx=src_pad_token_id),
            nn.Dropout(dropout),
        )
        self.tgt_embedding = nn.Sequential(
            nn.Embedding(tgt_vocab_size, embedding_dim, padding_idx=tgt_pad_token_id),
            nn.Dropout(dropout),
        )

        self.src_input_proj = nn.Linear(embedding_dim, hidden_dim)
        self.tgt_input_proj = nn.Linear(embedding_dim, hidden_dim)
        self.context_proj = nn.Linear(hidden_dim, hidden_dim)

        self.encoder = nn.ModuleList(
            [SSMBlock(hidden_dim, state_dim, dropout=dropout) for _ in range(num_layers)]
        )
        self.decoder = nn.ModuleList(
            [SSMBlock(hidden_dim, state_dim, dropout=dropout) for _ in range(num_layers)]
        )

        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, tgt_vocab_size),
        )

        self.init_weights()

    def init_weights(self) -> None:
        for name, p in self.named_parameters():
            if "embedding" in name and p.dim() > 1:
                nn.init.normal_(p, mean=0.0, std=0.01)
                if "src_embedding.0.weight" in name:
                    p.data[self.src_pad_token_id].zero_()
                elif "tgt_embedding.0.weight" in name:
                    p.data[self.tgt_pad_token_id].zero_()
            elif p.dim() > 1:
                nn.init.xavier_uniform_(p)
            elif "norm" not in name:
                nn.init.zeros_(p)

    def encode(self, src: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        src_mask = src != self.src_pad_token_id
        src_lengths = src_mask.sum(dim=1)

        if torch.any(src_lengths == 0):
            raise ValueError("Source batch contains an empty sequence after padding removal")

        x = self.src_embedding(src)
        x = self.src_input_proj(x)

        for layer in self.encoder:
            x = layer(x, src_mask)

        last_indices = (src_lengths - 1).clamp_min(0)
        batch_indices = torch.arange(src.size(0), device=src.device)
        context = x[batch_indices, last_indices]
        context = self.context_proj(context)
        return x, context, src_mask

    def decode(self, tgt_input: Tensor, context: Tensor) -> Tensor:
        tgt_mask = tgt_input != self.tgt_pad_token_id

        x = self.tgt_embedding(tgt_input)
        x = self.tgt_input_proj(x)
        x = x + context.unsqueeze(1)

        for layer in self.decoder:
            x = layer(x, tgt_mask)

        return self.output(x)

    def decode_step(
        self,
        input_token: Tensor,
        context: Tensor,
        states: list[Tensor | None],
    ) -> tuple[Tensor, list[Tensor]]:
        x = self.tgt_embedding(input_token)
        x = self.tgt_input_proj(x)
        x = x + context

        new_states: list[Tensor] = []
        for i, layer in enumerate(self.decoder):
            state = states[i]
            if state is None:
                state = layer.init_state(x.size(0), x.device, x.dtype)
            x, state = layer.step(x, state)
            new_states.append(state)

        logits = self.output(x)
        return logits, new_states

    def forward(self, src: Tensor, tgt: Tensor) -> Tensor:
        _, context, _ = self.encode(src)
        return self.decode(tgt[:, :-1], context)

    @torch.no_grad()
    def inference(
        self,
        src: Tensor,
        max_len: int = 100,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
    ) -> Tensor:
        batch_size = src.size(0)
        device = src.device

        _, context, _ = self.encode(src)
        states: list[Tensor | None] = [None for _ in range(self.num_layers)]

        sequences = torch.full(
            (batch_size, max_len),
            self.tgt_pad_token_id,
            dtype=torch.long,
            device=device,
        )
        sequences[:, 0] = bos_token_id

        input_token = torch.full((batch_size,), bos_token_id, dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for t in range(1, max_len):
            logits, states = self.decode_step(input_token, context, states)
            next_token = logits.argmax(dim=-1)

            active = ~finished
            sequences[active, t] = next_token[active]
            input_token = next_token

            finished |= next_token == eos_token_id
            if finished.all():
                break

        return sequences
