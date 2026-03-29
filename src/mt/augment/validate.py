from argparse import ArgumentParser
from typing import cast

import numpy as np
import transformers
from datasets import Dataset, load_from_disk
from sentence_transformers import SentenceTransformer, util


def add_similarity_scores(batch, model):
    ru = [x if x is not None else "" for x in batch["ru"]]
    en = [x if x is not None else "" for x in batch["en"]]
    aug_ru = [x if x is not None else "" for x in batch["aug_ru"]]
    aug_en = [x if x is not None else "" for x in batch["aug_en"]]

    aug_valid = [
        False if len(x) == 0 and len(y) == 0 else True
        for x, y in zip(batch["aug_ru"], batch["aug_en"])
    ]

    ru_emb = model.encode(ru, convert_to_tensor=True, normalize_embeddings=True)
    en_emb = model.encode(en, convert_to_tensor=True, normalize_embeddings=True)
    aug_ru_emb = model.encode(aug_ru, convert_to_tensor=True, normalize_embeddings=True)
    aug_en_emb = model.encode(aug_en, convert_to_tensor=True, normalize_embeddings=True)

    scores = util.pairwise_cos_sim(ru_emb, en_emb).cpu().numpy()
    aug_scores = util.pairwise_cos_sim(aug_ru_emb, aug_en_emb).cpu().numpy()
    aug_delta = aug_scores - scores

    return {
        "score": scores.tolist(),
        "aug_score": aug_scores.tolist(),
        "aug_delta": aug_delta.tolist(),
        "aug_valid": aug_valid,
    }


def main():
    parser = ArgumentParser("Validate augmented dataset using Sentance Transformer")
    parser.add_argument("--model", type=str)
    parser.add_argument("--dataset-path", type=str)
    parser.add_argument("--stat-error", default=0.0, type=float)
    args = parser.parse_args()

    transformers.logging.set_verbosity_error()
    dataset = cast(Dataset, load_from_disk(args.dataset_path))
    model = SentenceTransformer(args.model)

    scored = dataset.map(
        add_similarity_scores,
        fn_kwargs={"model": model},
        batched=True,
        batch_size=64,
        load_from_cache_file=False,
    )

    share_valid = np.mean(scored["aug_valid"])
    mean_score = np.mean(scored["score"])

    mean_aug_len_ru = np.mean(
        [len(x) for x in scored["aug_ru"]],
        where=scored["aug_valid"],
    )

    mean_aug_len_en = np.mean(
        [len(x) for x in scored["aug_en"]],
        where=scored["aug_valid"],
    )

    mean_aug_score = np.mean(
        scored["aug_score"],
        where=scored["aug_valid"],
    )

    mean_aug_delta = np.mean(
        scored["aug_delta"],
        where=scored["aug_valid"],
    )

    share_improved = np.mean(
        np.array(scored["aug_delta"]) > args.stat_error,
        where=scored["aug_valid"],
    )

    share_worsened = np.mean(
        np.array(scored["aug_delta"]) < -args.stat_error,
        where=scored["aug_valid"],
    )

    print("Valid share:", float(share_valid))
    print("Improved share:", float(share_improved))
    print("Worsened share:", float(share_worsened))
    print("Base mean score:", float(mean_score))
    print("Augmented mean ru length:", float(mean_aug_len_ru))
    print("Augmented mean en length:", float(mean_aug_len_en))
    print("Augmented mean score:", float(mean_aug_score))
    print("Augmented mean delta:", float(mean_aug_delta))


if __name__ == "__main__":
    main()
