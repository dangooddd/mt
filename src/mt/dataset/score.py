from argparse import ArgumentParser
from pathlib import Path

import transformers
from datasets import load_from_disk
from sentence_transformers import SentenceTransformer, util


def add_similarity_scores(batch, model, feature_name: str, lhs: str, rhs: str):
    ru = [x if x is not None else "" for x in batch[lhs]]
    en = [x if x is not None else "" for x in batch[rhs]]
    ru_emb = model.encode(ru, convert_to_tensor=True, normalize_embeddings=True)
    en_emb = model.encode(en, convert_to_tensor=True, normalize_embeddings=True)
    scores = util.pairwise_cos_sim(ru_emb, en_emb).cpu().numpy()
    return {feature_name: scores.tolist()}


def main():
    parser = ArgumentParser("Validate augmented dataset using Sentance Transformer")
    parser.add_argument("--model", type=str, default="paraphrase-multilingual-MiniLM-L12-v2")
    parser.add_argument("--dataset-path", type=str)
    parser.add_argument("--feature-name", type=str, default="score")
    parser.add_argument("--lhs", type=str, default="ru")
    parser.add_argument("--rhs", type=str, default="en")
    args = parser.parse_args()

    transformers.logging.set_verbosity_error()
    dataset = load_from_disk(args.dataset_path)
    model = SentenceTransformer(args.model)

    scored = dataset.map(
        add_similarity_scores,
        fn_kwargs={
            "model": model,
            "feature_name": args.feature_name,
            "lhs": args.lhs,
            "rhs": args.rhs,
        },
        batched=True,
        batch_size=512,
        load_from_cache_file=False,
    )

    scored.save_to_disk(Path(args.dataset_path).with_suffix(".scored"))


if __name__ == "__main__":
    main()
