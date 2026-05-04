from abc import ABC, abstractmethod

import torch.nn as nn
from torch import Tensor


class EncoderDecoder(ABC, nn.Module):
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
    ): ...


class DecoderOnly(ABC, nn.Module):
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
    ): ...
