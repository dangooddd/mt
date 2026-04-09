import csv
import json
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, cast

import torch
import transformers
from datasets import Dataset, DatasetDict, load_from_disk
from sacrebleu.metrics import BLEU, CHRF
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from tqdm import tqdm

from mt.dataset.score import add_similarity_scores
from mt.tokenizers import BaseTokenizer, get_tokenizer

from .luong import LuongSeq2Seq
from .mamba import MambaSeq2Seq
from .ssm import S4Seq2Seq
from .train import CollateFn, Models, compute_predictions

MODEL_CLASSES = {
    "luong": LuongSeq2Seq,
    "mamba": MambaSeq2Seq,
    "ssm": S4Seq2Seq,
}

SRC_COLUMN = "ru"
TGT_COLUMN = "en"
BATCH_SIZE = 64
SIMILARITY_BATCH_SIZE = 512
SIMILARITY_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


def load_split(dataset_path: str, split: str) -> Dataset:
    dataset = load_from_disk(dataset_path)

    if isinstance(dataset, DatasetDict):
        if split not in dataset:
            raise ValueError(f"Split '{split}' not found. Available splits: {list(dataset.keys())}")
        return dataset[split]

    return dataset


def resolve_tokenizer(config: dict[str, Any], config_key: str, file_path: Path) -> BaseTokenizer:
    tokenizer_type = config.get(config_key, "unigram")
    if not isinstance(tokenizer_type, str):
        raise ValueError(f"{config_key} must be a string tokenizer type")
    return get_tokenizer(model=tokenizer_type, file=file_path)


def instantiate_model(
    model_dir: Path,
) -> tuple[Models, BaseTokenizer, BaseTokenizer, dict[str, Any]]:
    config_path = model_dir / "config.json"
    model_path = model_dir / "model.pt"
    ru_tokenizer_path = model_dir / "ru_tokenizer.json"
    en_tokenizer_path = model_dir / "en_tokenizer.json"

    for path in [config_path, model_path, ru_tokenizer_path, en_tokenizer_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    config = cast(dict[str, Any], json.loads(config_path.read_text()))

    class_name = config.get("class") or config.get("name") or config.get("model")
    if class_name not in MODEL_CLASSES:
        raise ValueError(f"Unknown model class '{class_name}'. Supported: {sorted(MODEL_CLASSES)}")

    model_args = dict(config.get("args") or config.get("model_args") or {})
    src_tokenizer = resolve_tokenizer(config, "src_tokenizer", ru_tokenizer_path)
    tgt_tokenizer = resolve_tokenizer(config, "tgt_tokenizer", en_tokenizer_path)

    model_args["src_vocab_size"] = src_tokenizer.get_vocab_size()
    model_args["tgt_vocab_size"] = tgt_tokenizer.get_vocab_size()
    model_args["src_pad_token_id"] = src_tokenizer.pad_token_id
    model_args["tgt_pad_token_id"] = tgt_tokenizer.pad_token_id
    model_args["tgt_bos_token_id"] = tgt_tokenizer.bos_token_id
    model_args["tgt_eos_token_id"] = tgt_tokenizer.eos_token_id

    model = MODEL_CLASSES[class_name](**model_args)

    checkpoint = torch.load(model_path, map_location="cpu")
    if (
        isinstance(checkpoint, dict)
        and "model" in checkpoint
        and isinstance(checkpoint["model"], dict)
    ):
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    return model, src_tokenizer, tgt_tokenizer, config


@torch.inference_mode()
def generate_predictions(
    dataset: Dataset,
    model: Models,
    src_tokenizer: BaseTokenizer,
    tgt_tokenizer: BaseTokenizer,
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> list[str]:
    collate_fn = CollateFn(
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        src_column=SRC_COLUMN,
        tgt_column=TGT_COLUMN,
        max_src_length=max_length,
        max_tgt_length=max_length,
    )

    loader = DataLoader(
        cast(TorchDataset, dataset),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    model.to(device)
    model.eval()

    predictions: list[str] = []
    autocast_enabled = device.type == "cuda"

    for batch in tqdm(loader, desc="Inference"):
        with torch.autocast(device_type=device.type, enabled=autocast_enabled):
            batch_predictions, _ = compute_predictions(
                model=model,
                batch=batch,
                device=device,
                tgt_tokenizer=tgt_tokenizer,
                max_length=max_length,
            )

        predictions.extend(batch_predictions)

    return predictions


def add_scores(
    dataset: Dataset, model: SentenceTransformer, feature_name: str, lhs: str, rhs: str
) -> Dataset:
    return dataset.map(
        add_similarity_scores,
        fn_kwargs={
            "model": model,
            "feature_name": feature_name,
            "lhs": lhs,
            "rhs": rhs,
        },
        batched=True,
        batch_size=SIMILARITY_BATCH_SIZE,
        load_from_cache_file=False,
    )


def save_metrics(path: Path, bleu: float, chrf: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["metric", "value"])
        writer.writerow(["bleu", bleu])
        writer.writerow(["chrf", chrf])


def main() -> None:
    parser = ArgumentParser("Evaluate mt.models checkpoint on a dataset split")
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Path to model dir with config.json, model.pt, ru_tokenizer.json, en_tokenizer.json",
    )
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    args = parser.parse_args()
    transformers.logging.set_verbosity_error()

    model_dir = Path(args.model_dir)
    dataset = load_split(args.dataset_path, args.split)
    model, src_tokenizer, tgt_tokenizer, runtime_config = instantiate_model(model_dir)

    max_length = int(runtime_config.get("max_length", 256))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    predictions = generate_predictions(
        dataset=dataset,
        model=model,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        batch_size=BATCH_SIZE,
        max_length=max_length,
        device=device,
    )

    similarity_model = SentenceTransformer(SIMILARITY_MODEL_NAME, device=str(device))

    references = cast(list[str | None], dataset[TGT_COLUMN])
    references_text = [text or "" for text in references]

    scored = dataset.add_column("prediction", predictions)
    scored = add_scores(scored, similarity_model, "prediction_ru_score", SRC_COLUMN, "prediction")
    scored = add_scores(scored, similarity_model, "prediction_en_score", TGT_COLUMN, "prediction")

    bleu = BLEU().corpus_score(predictions, [references_text]).score
    chrf = CHRF().corpus_score(predictions, [references_text]).score

    output_path = Path(args.output_path)
    scored.save_to_disk(str(output_path))
    save_metrics(output_path / "metrics.csv", bleu=bleu, chrf=chrf)

    print(f"Saved scored split to: {output_path}")
    print(f"Saved metrics to: {output_path / 'metrics.csv'}")
    print(f"BLEU={bleu:.4f} CHRF={chrf:.4f}")


if __name__ == "__main__":
    main()
