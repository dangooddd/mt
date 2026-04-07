from typing import cast

import torch.optim as optim
from datasets import load_from_disk
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset

from mt.models.luong import LuongSeq2Seq
from mt.models.train import (
    attach_running_average,
    attach_tensorboard_loss_logging,
    build_collate_fn,
    compute_loss,
    create_trainer,
)
from mt.tokenizers import UnigramTokenizer

MAX_LR = 0.001
MIN_LR = 0.000001
BATCH_SIZE = 100
EPOCH_STEPS = 1000
WARMUP_STEPS = 50000
STEPS = 5000000

dataset = load_from_disk("data/datasets/opus-100-final")
tokenizer_ru = UnigramTokenizer.from_file("data/tokenizers/ru-unigram-24000.json")
tokenizer_en = UnigramTokenizer.from_file("data/tokenizers/en-unigram-24000.json")

model = LuongSeq2Seq(
    src_vocab_size=tokenizer_ru.get_vocab_size(),
    tgt_vocab_size=tokenizer_en.get_vocab_size(),
    src_pad_token_id=tokenizer_ru.pad_token_id,
    tgt_pad_token_id=tokenizer_en.pad_token_id,
    tgt_bos_token_id=tokenizer_en.bos_token_id,
    tgt_eos_token_id=tokenizer_en.eos_token_id,
)

optimizer = optim.AdamW(model.parameters(), lr=MAX_LR)
criterion = CrossEntropyLoss(ignore_index=tokenizer_en.pad_token_id, label_smoothing=0.1)

warmup_scheduler = optim.lr_scheduler.LinearLR(
    optimizer,
    start_factor=0.1,
    end_factor=1.0,
    total_iters=WARMUP_STEPS,
)

cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=(STEPS - WARMUP_STEPS),
    eta_min=MIN_LR,
)

scheduler = optim.lr_scheduler.SequentialLR(
    optimizer,
    schedulers=[warmup_scheduler, cosine_scheduler],
    milestones=[WARMUP_STEPS],
)

trainer = create_trainer(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    compute_loss=compute_loss,
)
attach_running_average(trainer)
attach_tensorboard_loss_logging(trainer, "data/runs")

collate_fn = build_collate_fn(
    src_tokenizer=tokenizer_ru,
    tgt_tokenizer=tokenizer_en,
    max_src_length=1024,
    max_tgt_length=1024,
    src_column="ru",
    tgt_column="en",
)

train_loader = DataLoader(
    cast(Dataset, dataset["train"]),
    batch_size=BATCH_SIZE,
    collate_fn=collate_fn,
    num_workers=8,
    shuffle=True,
)

trainer.run(
    train_loader,
    max_epochs=STEPS // EPOCH_STEPS,
    epoch_length=EPOCH_STEPS,
)
