from typing import Any, cast

import torch
from torch import Tensor

from mt.models.modules import DecoderOnly, EncoderDecoder, EncoderDecoderBilingual
from mt.tokenizers import BaseTokenizer, DecoderBaseTokenizer

from .pretrain import compute_decoder_loss
from .pretrain import compute_bilingual_loss, compute_loss as compute_pretrain_loss
from .rewards import RewardScorer


def completion_mask(token_ids: Tensor, eos_token_id: int) -> Tensor:
    eos_mask = token_ids.eq(eos_token_id)
    eos_count = eos_mask.cumsum(dim=1)
    return eos_count.eq(0) | (eos_mask & eos_count.eq(1))


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


def dapo_loss(
    per_token_logps: Tensor,
    old_per_token_logps: Tensor,
    ref_per_token_logps: Tensor,
    mask: Tensor,
    advantages: Tensor,
    clip_eps_low: float,
    clip_eps_high: float,
    kl_beta: float,
) -> tuple[Tensor, dict[str, Any]]:
    advantages = advantages.detach().view(-1, 1).to(dtype=per_token_logps.dtype)
    mask = mask.to(dtype=per_token_logps.dtype)

    per_token_ratio = torch.exp(per_token_logps - old_per_token_logps)
    per_token_clipped_ratio = per_token_ratio.clamp(1.0 - clip_eps_low, 1.0 + clip_eps_high)
    per_token_policy_loss = -torch.min(
        per_token_ratio * advantages,
        per_token_clipped_ratio * advantages,
    )
    kl_delta = torch.clamp(ref_per_token_logps - per_token_logps, min=-20.0, max=20.0)
    per_token_kl = torch.exp(kl_delta) - kl_delta - 1.0

    token_count = mask.sum().clamp_min(1.0)
    policy_loss = (per_token_policy_loss * mask).sum() / token_count
    kl = (per_token_kl * mask).sum() / token_count
    loss = policy_loss + kl_beta * kl

    clip_fraction = (
        ((per_token_ratio - per_token_clipped_ratio).abs() > 0) * mask
    ).sum() / token_count
    ratio = (per_token_ratio * mask).sum() / token_count

    return loss, {
        "policy_loss": float(policy_loss.detach().cpu()),
        "kl": float(kl.detach().cpu()),
        "clip_fraction": float(clip_fraction.detach().cpu()),
        "ratio": float(ratio.detach().cpu()),
    }


def encoder_decoder_per_token_logps(
    model: EncoderDecoder,
    src_ids: Tensor,
    src_mask: Tensor,
    generated_ids: Tensor,
    temperature: float,
    top_p: float,
) -> tuple[Tensor, Tensor]:
    decoder_input = generated_ids[:, :-1]
    targets = generated_ids[:, 1:]

    logits = model.forward(src_ids, decoder_input, src_mask).float()
    if temperature > 0.0:
        logits = logits / temperature
    logps = logits.log_softmax(dim=-1)
    per_token_logps = logps.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    eos_token_id = model.eos_token_id if isinstance(model, EncoderDecoderBilingual) else model.tgt_eos_token_id
    mask = completion_mask(targets, eos_token_id)
    return per_token_logps, mask


def decoder_only_per_token_logps(
    model: DecoderOnly,
    input_ids: Tensor,
    attention_mask: Tensor,
    type_ids: Tensor,
    generated_ids: Tensor,
    temperature: float,
    top_p: float,
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

    logits = model.forward(full_ids[:, :-1], full_mask[:, :-1], full_type_ids[:, :-1]).float()
    if temperature > 0.0:
        logits = logits / temperature
    targets = full_ids[:, 1:]
    logps = logits.log_softmax(dim=-1)
    per_token_logps = logps.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return per_token_logps, loss_mask


def compute_encoder_decoder_dapo_loss(
    model: EncoderDecoder,
    batch: dict[str, Any],
    device: torch.device,
    old_policy: EncoderDecoder,
    reference_policy: EncoderDecoder,
    tgt_tokenizer: BaseTokenizer,
    reward_scorer: RewardScorer,
    num_generations: int,
    max_length: int,
    temperature: float,
    top_p: float,
    advantage_eps: float,
    clip_eps_low: float,
    clip_eps_high: float,
    kl_beta: float,
    sft_mu: float,
) -> tuple[Tensor, dict[str, Any]]:
    sft_src_ids = batch["src_ids"].to(device=device)
    sft_src_mask = batch["src_mask"].to(device=device)
    sources = cast(list[str], batch["sources"])
    references = cast(list[str], batch["targets"])
    batch_size = sft_src_ids.size(0)

    src_ids = sft_src_ids.repeat_interleave(num_generations, dim=0)
    src_mask = sft_src_mask.repeat_interleave(num_generations, dim=0)
    sources = [source for source in sources for _ in range(num_generations)]
    references = [reference for reference in references for _ in range(num_generations)]

    old_policy.eval()
    reference_policy.eval()

    with torch.no_grad():
        generated_ids = old_policy.inference(
            src_ids,
            src_mask,
            max_length,
            temperature=temperature,
            top_p=top_p,
        )
        old_per_token_logps, mask = encoder_decoder_per_token_logps(
            old_policy,
            src_ids,
            src_mask,
            generated_ids,
            temperature,
            top_p,
        )
        ref_per_token_logps, _ = encoder_decoder_per_token_logps(
            reference_policy,
            src_ids,
            src_mask,
            generated_ids,
            temperature,
            top_p,
        )

    predictions = tgt_tokenizer.decode_batch(generated_ids.cpu().tolist())
    rewards = reward_scorer(predictions, references, sources)
    advantages, rewards_tensor = compute_advantages(
        rewards,
        batch_size=batch_size,
        num_generations=num_generations,
        device=device,
        eps=advantage_eps,
    )

    per_token_logps, _ = encoder_decoder_per_token_logps(
        model,
        src_ids,
        src_mask,
        generated_ids,
        temperature,
        top_p,
    )

    dapo_loss_value, metrics = dapo_loss(
        per_token_logps=per_token_logps,
        old_per_token_logps=old_per_token_logps,
        ref_per_token_logps=ref_per_token_logps,
        mask=mask,
        advantages=advantages,
        clip_eps_low=clip_eps_low,
        clip_eps_high=clip_eps_high,
        kl_beta=kl_beta,
    )

    if sft_mu > 0.0:
        if isinstance(model, EncoderDecoderBilingual):
            sft_loss, _ = compute_bilingual_loss(model, batch, device=device)
        else:
            sft_loss, _ = compute_pretrain_loss(model, batch, device=device)
        loss = (1 - sft_mu) * dapo_loss_value + sft_mu * sft_loss
    else:
        sft_loss = dapo_loss_value.detach().new_zeros(())
        loss = dapo_loss_value

    metrics.update(
        {
            "dapo_loss": float(dapo_loss_value.detach().cpu()),
            "sft_loss": float(sft_loss.detach().cpu()),
            "reward": float(rewards_tensor.mean().detach().cpu()),
            "reward_std": float(rewards_tensor.std(unbiased=False).detach().cpu()),
            "completion_length": float(mask.float().sum(dim=1).mean().detach().cpu()),
        }
    )

    return loss, metrics


def compute_decoder_only_dapo_loss(
    model: DecoderOnly,
    batch: dict[str, Any],
    device: torch.device,
    old_policy: DecoderOnly,
    reference_policy: DecoderOnly,
    tokenizer: DecoderBaseTokenizer,
    reward_scorer: RewardScorer,
    num_generations: int,
    max_length: int,
    temperature: float,
    top_p: float,
    advantage_eps: float,
    clip_eps_low: float,
    clip_eps_high: float,
    kl_beta: float,
    sft_mu: float,
) -> tuple[Tensor, dict[str, Any]]:
    input_ids = batch["inference_input_ids"].to(device=device)
    attention_mask = batch["inference_attention_mask"].to(device=device)
    type_ids = batch["inference_type_ids"].to(device=device)
    sources = cast(list[str], batch["sources"])
    references = cast(list[str], batch["targets"])
    batch_size = input_ids.size(0)

    input_ids = input_ids.repeat_interleave(num_generations, dim=0)
    attention_mask = attention_mask.repeat_interleave(num_generations, dim=0)
    type_ids = type_ids.repeat_interleave(num_generations, dim=0)
    sources = [source for source in sources for _ in range(num_generations)]
    references = [reference for reference in references for _ in range(num_generations)]

    old_policy.eval()
    reference_policy.eval()

    with torch.no_grad():
        generated_ids = old_policy.inference(
            input_ids,
            attention_mask,
            type_ids,
            max_length,
            temperature=temperature,
            top_p=top_p,
        )
        old_per_token_logps, mask = decoder_only_per_token_logps(
            old_policy,
            input_ids,
            attention_mask,
            type_ids,
            generated_ids,
            temperature,
            top_p,
        )
        ref_per_token_logps, _ = decoder_only_per_token_logps(
            reference_policy,
            input_ids,
            attention_mask,
            type_ids,
            generated_ids,
            temperature,
            top_p,
        )

    predictions = tokenizer.decode_batch(generated_ids.cpu().tolist())
    rewards = reward_scorer(predictions, references, sources)
    advantages, rewards_tensor = compute_advantages(
        rewards,
        batch_size=batch_size,
        num_generations=num_generations,
        device=device,
        eps=advantage_eps,
    )

    per_token_logps, _ = decoder_only_per_token_logps(
        model,
        input_ids,
        attention_mask,
        type_ids,
        generated_ids,
        temperature,
        top_p,
    )

    dapo_loss_value, metrics = dapo_loss(
        per_token_logps=per_token_logps,
        old_per_token_logps=old_per_token_logps,
        ref_per_token_logps=ref_per_token_logps,
        mask=mask,
        advantages=advantages,
        clip_eps_low=clip_eps_low,
        clip_eps_high=clip_eps_high,
        kl_beta=kl_beta,
    )

    if sft_mu > 0.0:
        sft_loss, _ = compute_decoder_loss(model, batch, device=device)
        loss = (1 - sft_mu) * dapo_loss_value + sft_mu * sft_loss
    else:
        sft_loss = dapo_loss_value.detach().new_zeros(())
        loss = dapo_loss_value

    metrics.update(
        {
            "dapo_loss": float(dapo_loss_value.detach().cpu()),
            "sft_loss": float(sft_loss.detach().cpu()),
            "reward": float(rewards_tensor.mean().detach().cpu()),
            "reward_std": float(rewards_tensor.std(unbiased=False).detach().cpu()),
            "completion_length": float(mask.float().sum(dim=1).mean().detach().cpu()),
        }
    )

    return loss, metrics
