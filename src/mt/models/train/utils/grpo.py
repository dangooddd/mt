from typing import Any, cast

import torch
from sacrebleu.metrics import CHRF
from torch import Tensor

from mt.models.modules import DecoderOnly, EncoderDecoder
from mt.tokenizers import BaseTokenizer, BilingualBaseTokenizer


def completion_mask(token_ids: Tensor, eos_token_id: int) -> Tensor:
    eos_mask = token_ids.eq(eos_token_id)
    eos_count = eos_mask.cumsum(dim=1)
    return eos_count.eq(0) | (eos_mask & eos_count.eq(1))


def compute_chrf_rewards(
    predictions: list[str],
    references: list[str],
    metric: CHRF,
) -> list[float]:
    return [
        metric.sentence_score(prediction or "", [reference or ""]).score
        for prediction, reference in zip(predictions, references, strict=True)
    ]


def compute_advantages(
    rewards: list[float],
    batch_size: int,
    num_generations: int,
    device: torch.device,
    eps: float,
) -> tuple[Tensor, Tensor]:
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device).view(
        batch_size,
        num_generations,
    )
    mean = rewards_tensor.mean(dim=1, keepdim=True)
    std = rewards_tensor.std(dim=1, keepdim=True, unbiased=False)
    advantages = (rewards_tensor - mean) / (std + eps)
    return advantages.reshape(-1), rewards_tensor


def grpo_loss(token_log_probs: Tensor, mask: Tensor, advantages: Tensor) -> Tensor:
    weights = advantages.detach().view(-1, 1).to(dtype=token_log_probs.dtype)
    mask = mask.to(dtype=token_log_probs.dtype)
    return -((token_log_probs * weights * mask).sum() / mask.sum().clamp_min(1.0))


def encoder_decoder_token_log_probs(
    model: EncoderDecoder,
    src_ids: Tensor,
    src_mask: Tensor,
    generated_ids: Tensor,
) -> tuple[Tensor, Tensor]:
    decoder_input = generated_ids[:, :-1]
    targets = generated_ids[:, 1:]

    logits = model.forward(src_ids, decoder_input, src_mask)
    log_probs = logits.float().log_softmax(dim=-1)
    token_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    mask = completion_mask(targets, model.tgt_eos_token_id)
    return token_log_probs, mask


def decoder_only_token_log_probs(
    model: DecoderOnly,
    input_ids: Tensor,
    attention_mask: Tensor,
    type_ids: Tensor,
    generated_ids: Tensor,
) -> tuple[Tensor, Tensor]:
    batch_size, generated_length = generated_ids.shape
    prompt_lengths = attention_mask.long().sum(dim=1).clamp_min(1)
    max_prompt_length = int(prompt_lengths.max().item())
    total_length = max_prompt_length + generated_length
    device = input_ids.device

    full_ids = torch.full(
        (batch_size, total_length),
        model.pad_token_id,
        dtype=torch.long,
        device=device,
    )
    full_mask = torch.zeros((batch_size, total_length), dtype=torch.bool, device=device)
    full_type_ids = torch.zeros((batch_size, total_length), dtype=torch.long, device=device)
    loss_mask = torch.zeros((batch_size, total_length - 1), dtype=torch.bool, device=device)
    generated_mask = completion_mask(generated_ids, model.eos_token_id)

    for i in range(batch_size):
        prompt_length = int(prompt_lengths[i].item())
        generated_start = prompt_length
        generated_end = generated_start + generated_length
        loss_start = prompt_length - 1
        loss_end = loss_start + generated_length

        full_ids[i, :prompt_length] = input_ids[i, :prompt_length]
        full_ids[i, generated_start:generated_end] = generated_ids[i]
        full_mask[i, :generated_end] = True
        full_type_ids[i, :prompt_length] = type_ids[i, :prompt_length]
        full_type_ids[i, generated_start:generated_end] = 1
        loss_mask[i, loss_start:loss_end] = generated_mask[i]

    logits = model.forward(full_ids[:, :-1], full_mask[:, :-1], full_type_ids[:, :-1])
    targets = full_ids[:, 1:]
    log_probs = logits.float().log_softmax(dim=-1)
    token_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return token_log_probs, loss_mask


def compute_encoder_decoder_grpo_loss(
    model: EncoderDecoder,
    batch: dict[str, Any],
    device: torch.device,
    tgt_tokenizer: BaseTokenizer,
    metric: CHRF,
    num_generations: int,
    max_length: int,
    temperature: float,
    advantage_eps: float,
) -> tuple[Tensor, dict[str, Any]]:
    src_ids = batch["src_ids"].to(device=device)
    src_mask = batch["src_mask"].to(device=device)
    references = cast(list[str], batch["targets"])
    batch_size = src_ids.size(0)

    src_ids = src_ids.repeat_interleave(num_generations, dim=0)
    src_mask = src_mask.repeat_interleave(num_generations, dim=0)
    references = [reference for reference in references for _ in range(num_generations)]

    model.eval()
    with torch.no_grad():
        generated_ids = model.inference(src_ids, src_mask, max_length, temperature=temperature)

    predictions = tgt_tokenizer.decode_batch(generated_ids.cpu().tolist())
    rewards = compute_chrf_rewards(predictions, references, metric)
    advantages, rewards_tensor = compute_advantages(
        rewards,
        batch_size=batch_size,
        num_generations=num_generations,
        device=device,
        eps=advantage_eps,
    )

    model.train()
    token_log_probs, mask = encoder_decoder_token_log_probs(model, src_ids, src_mask, generated_ids)
    loss = grpo_loss(token_log_probs, mask, advantages)

    return loss, {
        "reward": float(rewards_tensor.mean().detach().cpu()),
        "reward_std": float(rewards_tensor.std(unbiased=False).detach().cpu()),
        "completion_length": float(mask.float().sum(dim=1).mean().detach().cpu()),
    }


def compute_decoder_only_grpo_loss(
    model: DecoderOnly,
    batch: dict[str, Any],
    device: torch.device,
    tokenizer: BilingualBaseTokenizer,
    metric: CHRF,
    num_generations: int,
    max_length: int,
    temperature: float,
    advantage_eps: float,
) -> tuple[Tensor, dict[str, Any]]:
    input_ids = batch["inference_input_ids"].to(device=device)
    attention_mask = batch["inference_attention_mask"].to(device=device)
    type_ids = batch["inference_type_ids"].to(device=device)
    references = cast(list[str], batch["targets"])
    batch_size = input_ids.size(0)

    input_ids = input_ids.repeat_interleave(num_generations, dim=0)
    attention_mask = attention_mask.repeat_interleave(num_generations, dim=0)
    type_ids = type_ids.repeat_interleave(num_generations, dim=0)
    references = [reference for reference in references for _ in range(num_generations)]

    model.eval()
    with torch.no_grad():
        generated_ids = model.inference(
            input_ids,
            attention_mask,
            type_ids,
            max_length,
            temperature=temperature,
        )

    predictions = tokenizer.decode_batch(generated_ids.cpu().tolist())
    rewards = compute_chrf_rewards(predictions, references, metric)
    advantages, rewards_tensor = compute_advantages(
        rewards,
        batch_size=batch_size,
        num_generations=num_generations,
        device=device,
        eps=advantage_eps,
    )

    model.train()
    token_log_probs, mask = decoder_only_token_log_probs(
        model,
        input_ids,
        attention_mask,
        type_ids,
        generated_ids,
    )
    loss = grpo_loss(token_log_probs, mask, advantages)

    return loss, {
        "reward": float(rewards_tensor.mean().detach().cpu()),
        "reward_std": float(rewards_tensor.std(unbiased=False).detach().cpu()),
        "completion_length": float(mask.float().sum(dim=1).mean().detach().cpu()),
    }
