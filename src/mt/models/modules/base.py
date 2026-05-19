from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch import Tensor


def apply_inference_params(
    logits: Tensor,
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> Tensor:
    if temperature < 0.0:
        raise ValueError("temperature must be non-negative")
    if top_p <= 0.0 or top_p > 1.0:
        raise ValueError("top_p must be in (0, 1]")

    logits = logits.float()
    if temperature > 0.0:
        logits = logits / temperature

    if top_p == 1.0:
        return logits

    sorted_logits, sorted_indices = torch.sort(logits, descending=False, dim=-1)
    cumulative_probabilities = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
    sorted_indices_to_remove = cumulative_probabilities <= (1 - top_p)
    sorted_indices_to_remove[..., -1:] = False

    indices_to_remove = torch.zeros_like(sorted_indices_to_remove).scatter(
        -1,
        sorted_indices,
        sorted_indices_to_remove,
    )
    return logits.masked_fill(indices_to_remove, float("-inf"))


class TemperatureGenerationMixin:
    @staticmethod
    def _sample_next_token(logits: Tensor, temperature: float = 0.0, top_p: float = 1.0) -> Tensor:
        if temperature == 0.0:
            return logits.argmax(dim=-1)

        logits = apply_inference_params(logits, temperature, top_p)
        probabilities = torch.softmax(logits, dim=-1)
        sampled = torch.multinomial(
            probabilities.reshape(-1, probabilities.size(-1)),
            num_samples=1,
        )
        return sampled.reshape(probabilities.shape[:-1])


class EncoderDecoder(TemperatureGenerationMixin, ABC, nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        src_pad_token_id: int,
        tgt_pad_token_id: int,
        tgt_bos_token_id: int,
        tgt_eos_token_id: int,
    ):
        super().__init__()
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.src_pad_token_id = src_pad_token_id
        self.tgt_pad_token_id = tgt_pad_token_id
        self.tgt_bos_token_id = tgt_bos_token_id
        self.tgt_eos_token_id = tgt_eos_token_id

    @abstractmethod
    def forward(
        self,
        input_ids: Tensor,
        output_ids: Tensor,
        attention_mask: Tensor,
    ): ...

    @abstractmethod
    def inference(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        max_length: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ): ...


class EncoderDecoderBilingual(TemperatureGenerationMixin, ABC, nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_token_id: int,
        bos_token_id: int,
        eos_token_id: int,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

    @abstractmethod
    def forward(
        self,
        input_ids: Tensor,
        output_ids: Tensor,
        attention_mask: Tensor,
    ): ...

    @abstractmethod
    def inference(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        max_length: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ): ...


class DecoderOnly(TemperatureGenerationMixin, ABC, nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_token_id: int,
        bos_token_id: int,
        eos_token_id: int,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

    @abstractmethod
    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        type_ids: Tensor,
    ): ...

    @abstractmethod
    def inference(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        type_ids: Tensor,
        max_length: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ): ...
