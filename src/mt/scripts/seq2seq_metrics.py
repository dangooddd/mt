from functools import partial
from typing import Optional, cast

import sacrebleu
import torch
import typer
from datasets import Dataset, DatasetDict, load_from_disk
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..datasets import get_dataset_preset, limit_samples, load_dataset_cached
from ..datasets.dataloader import Seq2SeqDataset, collate_fn
from ..models import LuongSeq2Seq, Seq2SeqLightningModule
from ..tokenizers import Seq2SeqTokenizer


def main(
    ckpt_path: str,
    src_tokenizer_path: str = "data/tokenizers/seq2seq_en.json",
    tgt_tokenizer_path: str = "data/tokenizers/seq2seq_ru.json",
    dataset_preset: str = "opus-100",
    from_disk: bool = False,
    src_field: str = "en",
    tgt_field: str = "ru",
    max_gen_length: int = 512,
    max_samples: Optional[int] = 512,
    embedding_dim: int = 512,
    hidden_dim: int = 512,
    num_layers: int = 4,
    dropout: float = 0.3,
    batch_size: int = 32,
    split: str = "validation",
    seed: int = 42,
    num_workers: int = 0,
    pin_memory: bool = True,
    show_examples: int = 5,
):
    torch.set_float32_matmul_precision("medium")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    src_tokenizer = Seq2SeqTokenizer.from_file(src_tokenizer_path)
    tgt_tokenizer = Seq2SeqTokenizer.from_file(tgt_tokenizer_path)
    src_vocab_size = src_tokenizer.tokenizer.get_vocab_size()
    tgt_vocab_size = tgt_tokenizer.tokenizer.get_vocab_size()

    model = LuongSeq2Seq(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        src_pad_token_id=src_tokenizer.pad_token_id,
        tgt_pad_token_id=tgt_tokenizer.pad_token_id,
    )

    model = Seq2SeqLightningModule.load_from_checkpoint(
        ckpt_path,
        model=model,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        max_gen_length=max_gen_length,
    ).model

    if from_disk:
        dataset = cast(DatasetDict, load_from_disk(dataset_preset))[split]
    else:
        preset = get_dataset_preset(dataset_preset)
        preset["split"] = split
        dataset = load_dataset_cached(preset)[split]

    if max_samples is not None:
        dataset = limit_samples(dataset, max_samples)

    seq2seq_dataset = Seq2SeqDataset(
        dataset=dataset,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        src_field=src_field,
        tgt_field=tgt_field,
        max_length=None,
    )

    data_loader = DataLoader(
        seq2seq_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        collate_fn=partial(
            collate_fn,
            src_pad_id=src_tokenizer.pad_token_id,
            tgt_pad_id=tgt_tokenizer.pad_token_id,
        ),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    predictions = []
    targets = []
    sources = []

    with torch.no_grad():
        for src, tgt in tqdm(data_loader, desc="Batches"):
            src = src.to(device)
            tgt = tgt.to(device)

            pred = model.inference(
                src,
                max_len=max_gen_length,
                bos_token_id=tgt_tokenizer.bos_token_id,
                eos_token_id=tgt_tokenizer.eos_token_id,
            )

            for pred_ids, tgt_ids, src_ids in zip(
                pred.cpu().tolist(),
                tgt.cpu().tolist(),
                src.cpu().tolist(),
            ):
                pred_text = tgt_tokenizer.decode(pred_ids, skip_special_tokens=True)
                tgt_text = tgt_tokenizer.decode(tgt_ids, skip_special_tokens=True)
                src_text = src_tokenizer.decode(src_ids, skip_special_tokens=True)

                predictions.append(pred_text)
                targets.append(tgt_text)
                sources.append(src_text)

    bleu = sacrebleu.corpus_bleu(predictions, [targets], tokenize="13a")
    chrf = sacrebleu.corpus_chrf(predictions, [targets])

    print("=" * 60)
    print(f"Examples num: {len(predictions)}")
    print(f"BLEU score: {bleu.score}")
    print(f"chrF score: {chrf.score}")
    print()

    if show_examples > 0:
        print("Examples")
        print("=" * 60)
        for i in range(min(show_examples, len(predictions))):
            print(f"Example {i + 1}:")
            print(f"  Source: {sources[i]}")
            print(f"  Target: {targets[i]}")
            print(f"  Prediction: {predictions[i]}")
            print()


if __name__ == "__main__":
    typer.run(main)
