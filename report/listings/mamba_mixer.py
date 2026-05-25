class MambaMixer(nn.Module):
    def _selective_scan(self, u, delta, b, c):
        batch_size, seq_len, d_inner = u.shape
        a = -torch.exp(self.A_log.float())
        d = self.D.float()
        state = u.new_zeros(batch_size, d_inner, self.d_state, dtype=torch.float32)
        ys = []

        for t in range(seq_len):
            u_t = u.float()[:, t]
            delta_t = delta.float()[:, t]
            b_t = b.float()[:, t]
            c_t = c.float()[:, t]

            delta_a = torch.exp(delta_t.unsqueeze(-1) * a.unsqueeze(0))
            delta_b_u = delta_t.unsqueeze(-1) * u_t.unsqueeze(-1) * b_t.unsqueeze(1)
            state = delta_a * state + delta_b_u
            y_t = (state * c_t.unsqueeze(1)).sum(dim=-1) + d.unsqueeze(0) * u_t
            ys.append(y_t)

        return torch.stack(ys, dim=1).to(u.dtype)

    def forward(self, x):
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
