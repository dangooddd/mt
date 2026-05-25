class LuongAttention(nn.Module):
    def __init__(self, encoder_dim: int, decoder_dim: int):
        super().__init__()
        self.key_proj = nn.Linear(encoder_dim, decoder_dim, bias=False)

    def forward(self, decoder_hidden, encoder_outputs, mask):
        keys = self.key_proj(encoder_outputs)
        scores = torch.bmm(keys, decoder_hidden.unsqueeze(2)).squeeze(2)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        attn_weights = torch.softmax(scores, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, attn_weights


class LuongSeq2Seq(nn.Module):
    def forward(self, input_ids, output_ids, attention_mask):
        encoder_outputs, (hidden, cell) = self.encode(input_ids, attention_mask)
        preds = []

        for t in range(output_ids.size(1)):
            input_id = output_ids[:, t]
            logits, hidden, cell = self.decode_step(
                input_id,
                hidden,
                cell,
                encoder_outputs,
                attention_mask,
            )
            preds.append(logits)

        return torch.stack(preds, dim=1)
