from argparse import ArgumentParser
from pathlib import Path
from typing import cast

import torch
import transformers
from datasets import DatasetDict, load_from_disk
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from tqdm import tqdm
from transformers import pipeline

MODEL_NAME = "google/translategemma-12b-it"
BATCH_SIZE = 512
MAX_NEW_TOKENS = 256


def build_messages(text: str, src: str, tgt: str) -> list[dict[str, object]]:
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "source_lang_code": src,
                    "target_lang_code": tgt,
                    "text": text,
                }
            ],
        }
    ]


def main() -> None:
    parser = ArgumentParser("Generate TranslateGemma translations for a dataset split")
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--src", type=str, default="ru")
    parser.add_argument("--tgt", type=str, default="en")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--model-name", type=str, default=MODEL_NAME)
    args = parser.parse_args()
    transformers.logging.set_verbosity_error()

    dataset = load_from_disk(args.dataset_path)
    assert isinstance(dataset, DatasetDict)
    split = dataset[args.split]
    dtype = torch.bfloat16

    pipe = pipeline(
        "image-text-to-text",
        model=args.model_name,
        dtype=dtype,
        device=0 if torch.cuda.is_available() else -1,
    )

    loader = DataLoader(
        cast(TorchDataset, split),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda items: [item.get(args.src) for item in items],
    )

    translations: list[str] = []
    for batch in tqdm(loader, desc="TranslateGemma"):
        messages = [build_messages(text or "", args.src, args.tgt) for text in batch]
        outputs = pipe(
            text=messages,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            cache_implementation="static",
        )  # type: ignore
        translations.extend(
            output[0]["generated_text"][-1]["content"].strip() for output in outputs
        )

    column = f"distill_{args.tgt}"
    if column in split.column_names:
        split = split.remove_columns(column)
    split = split.add_column(column, translations)
    dataset[args.split] = split
    dataset.save_to_disk(str(Path(args.output_path)))


if __name__ == "__main__":
    main()
