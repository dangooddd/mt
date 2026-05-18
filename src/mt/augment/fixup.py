import asyncio
import json
import logging
import os
from argparse import ArgumentParser
from functools import lru_cache
from pathlib import Path
from string import Template
from typing import Callable, TypedDict

import datasets.config
from datasets import Dataset, load_from_disk
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from ..logging import setup_logger

PROMPT = """
# Task
You are a bilingual data-cleaning and alignment assistant.
You are given two versions of the same content: in russian and in english languages.
Your task is to produce cleaned, corrected, and semantically aligned output for both languages.

Rules:
1. If the texts contain typos, spelling mistakes, grammar errors, OCR noise, malformed fragments, encoding issues, or other dirty data, fix them.
2. If the language is wrong, translate it into the correct language.
3. Longest out of two texts is ground truth - infer missed text for second text.
4. Remove irrelevant garbage, duplicated fragments, and artifacts unless they are necessary for meaning.
5. Preserve the original meaning as much as possible, but prefer clarity, correctness, and cross-language consistency over literal wording.

## English version
$en

## Russian version
$ru

## Output
Return **valid** JSON object only:
1. No markdown, no explanations, no extra text.
2. The first character of the response must be {.
3. The last character of the response must be }.
4. All string values must be valid JSON strings, escape quotes and newlines correctly.
5. Two fields: "aug_en" and "aug_ru" **only**.

## Schema
{
  "aug_en": "string",
  "aug_ru": "string"
}
"""

setup_logger("augment")
logger = logging.getLogger("augment")


class AugmentResult(TypedDict):
    aug_en: str
    aug_ru: str
    aug: bool


@lru_cache(maxsize=1)
def get_client() -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ["OPENAI_BASE_URL"],
        timeout=500,
        max_retries=1,
    )


def checkpointed_map(
    transform: Callable,
    dataset: Dataset,
    fn_kwargs: dict[str, object],
    save_every_n: int,
    save_dir: Path,
):
    save_dir.mkdir(parents=True, exist_ok=True)
    dataset.add_column

    for start in range(0, len(dataset), save_every_n):
        end = min(len(dataset), start + save_every_n)
        chunk = dataset.select(range(start, end))
        chunk_save = save_dir / f"chunk-{start}-{end}"

        if chunk_save.exists():
            print(f"Skip chunk-{start}-{end} because {str(chunk_save)} exists")
            continue

        augmented = chunk.map(
            transform,
            fn_kwargs=fn_kwargs,
            desc=f"Processing chunk-{start}-{end}",
        )

        augmented.save_to_disk(str(chunk_save))


async def augment(
    example,
    model: str,
    threshold: float,
) -> AugmentResult:
    if example.get("aug", False):
        return {
            "aug_ru": example.get("aug_ru") or example["ru"],
            "aug_en": example.get("aug_en") or example["en"],
            "aug": True,
        }

    if example["score"] >= threshold and example["ru_detected"] and example["en_detected"]:
        return {
            "aug_ru": "",
            "aug_en": "",
            "aug": False,
        }

    client = get_client()

    message: ChatCompletionMessageParam = {
        "role": "user",
        "content": Template(PROMPT).substitute(
            en=example["en"],
            ru=example["ru"],
        ),
    }

    try:
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model=model,
                messages=[message],
            ),
            timeout=550,
        )

        parsed = json.loads(response.choices[0].message.content or "{}")

        if (not isinstance(parsed["aug_en"], str)) or (not isinstance(parsed["aug_ru"], str)):
            raise ValueError("Model did not return correct structured output")

        if len(parsed["aug_en"]) * len(parsed["aug_ru"]) == 0:
            raise ValueError("Model did not return correct structured output")

        parsed["aug"] = True
        return parsed

    except Exception as e:
        logger.error(f"Error occured during augmentation call ({type(e).__name__}): {e}")
        return {"aug_ru": "", "aug_en": "", "aug": False}


def main():
    parser = ArgumentParser("Augment translation dataset using LLM")
    parser.add_argument("--model", type=str)
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--n-calls", default=128, type=int)
    parser.add_argument("--save-every-n", default=1024, type=int)
    parser.add_argument("--save-dir", type=Path)
    parser.add_argument("--split", default="train", type=str)
    parser.add_argument("--threshold", default=0.7, type=float)
    args = parser.parse_args()

    datasets.config.MAX_NUM_RUNNING_ASYNC_MAP_FUNCTIONS_IN_PARALLEL = args.n_calls
    dataset = load_from_disk(args.dataset_path)[args.split].flatten()

    checkpointed_map(
        augment,
        fn_kwargs={"model": args.model, "threshold": args.threshold},
        dataset=dataset,
        save_dir=args.save_dir,
        save_every_n=args.save_every_n,
    )


if __name__ == "__main__":
    main()
