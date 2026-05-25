class MambaLayer(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.norm(x)
        x = self.mamba(x)
        return residual + self.dropout(x)


class MambaDecoder(DecoderOnly):
    def encode(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        type_ids: Tensor,
    ) -> Tensor:
        x = self.embedding(input_ids)
        for layer in self.encoder_layers:
            x = layer(x, attention_mask, type_ids)
        return x

    def decode(self, x: Tensor) -> Tensor:
        for layer in self.decoder_layers:
            x = layer(x)
        x = self.norm(x)
        return self.output(x)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        type_ids: Tensor,
    ) -> Tensor:
        x = self.encode(input_ids, attention_mask, type_ids)
        return self.decode(x)
