from collections.abc import Callable

import torch
from sacrebleu.metrics import BLEU, CHRF

from mt.dataset.score import suppress_output

DEFAULT_COMET_MODEL_NAME = "Unbabel/wmt22-comet-da"

RewardScorer = Callable[[list[str], list[str], list[str]], list[float]]


def create_sacrebleu_reward_scorer(reward: str) -> RewardScorer:
    metric = CHRF() if reward == "chrf" else BLEU(effective_order=True)

    def score(
        predictions: list[str],
        references: list[str],
        sources: list[str],
    ) -> list[float]:
        _ = sources
        return [
            float(metric.sentence_score(prediction or "", [reference or ""]).score)
            for prediction, reference in zip(predictions, references, strict=True)
        ]

    return score


def create_comet_reward_scorer(
    model_name: str = DEFAULT_COMET_MODEL_NAME,
    batch_size: int = 100,
) -> RewardScorer:
    with suppress_output():
        from comet import download_model, load_from_checkpoint

        scorer = load_from_checkpoint(download_model(model_name))
        scorer.eval()

    @torch.inference_mode()
    def score(
        predictions: list[str],
        references: list[str],
        sources: list[str],
    ) -> list[float]:
        samples = [
            {"src": source or "", "mt": prediction or "", "ref": reference or ""}
            for source, prediction, reference in zip(sources, predictions, references, strict=True)
        ]
        use_cuda = torch.cuda.is_available()

        with suppress_output():
            result = scorer.predict(
                samples,
                batch_size=batch_size,
                gpus=1 if use_cuda else 0,
                progress_bar=False,
                accelerator="auto" if use_cuda else "cpu",
                num_workers=0,
            )

        return [float(score) for score in result.scores]  # type: ignore

    return score


def create_reward_scorer(
    reward: str,
    comet_model_name: str = DEFAULT_COMET_MODEL_NAME,
    comet_batch_size: int = 100,
) -> RewardScorer:
    if reward in {"chrf", "bleu"}:
        return create_sacrebleu_reward_scorer(reward)

    if reward == "comet":
        return create_comet_reward_scorer(
            model_name=comet_model_name,
            batch_size=comet_batch_size,
        )

    raise ValueError("reward must be one of: chrf, bleu, comet")
