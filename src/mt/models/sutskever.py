import random

import sacrebleu
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from lightning.pytorch.loggers import TensorBoardLogger
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from ..tokenizers import Seq2SeqTokenizer


class SutskeverSeq2Seq(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        embedding_dim: int = 1000,
        hidden_dim: int = 1000,
        num_layers: int = 4,
        dropout: float = 0.3,
        src_pad_token_id: int = 0,
        tgt_pad_token_id: int = 0,
        reverse_input: bool = True,
    ):
        super().__init__()
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.reverse_input = reverse_input
        self.src_pad_token_id = src_pad_token_id
        self.tgt_pad_token_id = tgt_pad_token_id

        self.src_embedding = nn.Embedding(
            src_vocab_size, embedding_dim, padding_idx=src_pad_token_id
        )
        self.tgt_embedding = nn.Embedding(
            tgt_vocab_size, embedding_dim, padding_idx=tgt_pad_token_id
        )

        self.encoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )

        self.decoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )

        self.output_layer = nn.Linear(hidden_dim, tgt_vocab_size)

        self.init_weights()

    def forward(self, src: Tensor, tgt: Tensor) -> Tensor:
        if self.reverse_input:
            src = torch.flip(src, dims=[1])

        src_lengths = (src != self.src_pad_token_id).sum(dim=1)
        src_embedded = pack_padded_sequence(
            self.src_embedding(src),
            src_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, (hidden, cell) = self.encoder(src_embedded)

        tgt_lengths = (tgt[:, :-1] != self.tgt_pad_token_id).sum(dim=1)
        tgt_embedded = pack_padded_sequence(
            self.tgt_embedding(tgt[:, :-1]),
            tgt_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        decoder_output, _ = self.decoder(tgt_embedded, (hidden, cell))
        decoder_output, _ = pad_packed_sequence(
            decoder_output,
            batch_first=True,
        )

        output = self.output_layer(decoder_output)
        return output

    def inference(
        self,
        src: torch.Tensor,
        max_len: int = 100,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
    ) -> torch.Tensor:
        batch_size = src.size(0)
        device = src.device

        if self.reverse_input:
            src = torch.flip(src, dims=[1])

        src_lengths = (src != self.src_pad_token_id).sum(dim=1)
        src_embedded = pack_padded_sequence(
            self.src_embedding(src),
            src_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, (hidden, cell) = self.encoder(src_embedded)

        sequences = torch.full(
            (batch_size, max_len),
            self.tgt_pad_token_id,
            dtype=torch.long,
            device=device,
        )
        sequences[:, 0] = bos_token_id

        decoder_input = torch.full(
            (batch_size, 1),
            bos_token_id,
            dtype=torch.long,
            device=device,
        )

        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for t in range(1, max_len):
            tgt_embedded = self.tgt_embedding(decoder_input)
            decoder_output, (hidden, cell) = self.decoder(tgt_embedded, (hidden, cell))
            output = self.output_layer(decoder_output[:, -1:, :])

            # Get prediction
            next_token = output.argmax(dim=-1)

            # Update predictions and decoder input
            mask = ~finished
            sequences[mask, t] = next_token[mask, 0]
            decoder_input = next_token

            # Update finished sequences
            new_finished = (next_token == eos_token_id).squeeze(-1)
            finished = finished | new_finished

            if finished.all():
                break

        return sequences

    def init_weights(self):
        for name, p in self.named_parameters():
            if "src_embedding" in name:
                nn.init.normal_(p, mean=0, std=0.01)
                p.data[self.src_pad_token_id].zero_()
            elif "tgt_embedding" in name:
                nn.init.normal_(p, mean=0, std=0.01)
                p.data[self.tgt_pad_token_id].zero_()
            elif p.dim() > 1:
                nn.init.xavier_uniform_(p)


class Seq2SeqLightningModule(LightningModule):
    def __init__(
        self,
        model,
        src_tokenizer: Seq2SeqTokenizer,
        tgt_tokenizer: Seq2SeqTokenizer,
        learning_rate: float = 0.001,
        weight_decay: float = 0.001,
        warmup_interval: float = 0.05,
        gradient_accumulation_steps: int = 1,
        clip_grad_norm: float = 1.0,
        compute_bleu: bool = True,
        max_gen_length: int = 100,
        seed: int = 42,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model", "src_tokenizer", "tgt_tokenizer"])

        self.model = model
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer

        random.seed(self.hparams.seed)  # type: ignore
        self.val_predictions: list[list[int]] = []
        self.val_references: list[list[int]] = []

    def forward(self, src: Tensor, tgt: Tensor) -> Tensor:
        return self.model(src, tgt)

    def training_step(self, batch, batch_idx):
        src, tgt = batch
        output = self.model(src, tgt)
        target = tgt[:, 1:]

        loss = F.cross_entropy(
            output.reshape(-1, output.size(-1)),
            target.reshape(-1),
            ignore_index=self.tgt_tokenizer.pad_token_id,
        )
        perplexity = torch.exp(loss)

        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        self.log(
            "train/perplexity",
            perplexity,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        src, tgt = batch
        output = self.model(src, tgt)
        target = tgt[:, 1:]

        loss = F.cross_entropy(
            output.reshape(-1, output.size(-1)),
            target.reshape(-1),
            ignore_index=self.tgt_tokenizer.pad_token_id,
        )
        perplexity = torch.exp(loss)

        self.log(
            "val/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        self.log(
            "val/perplexity",
            perplexity,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        if self.hparams.compute_bleu:  # type: ignore
            with torch.no_grad():
                predictions = self.model.inference(
                    src,
                    max_len=self.hparams.max_gen_length,  # type: ignore
                    bos_token_id=self.tgt_tokenizer.bos_token_id,
                    eos_token_id=self.tgt_tokenizer.eos_token_id,
                )

                for pred, ref in zip(predictions, tgt):
                    pred_ids = pred.cpu().tolist()
                    ref_ids = ref.cpu().tolist()

                    self.val_predictions.append(pred_ids)
                    self.val_references.append(ref_ids)

        return loss

    def on_validation_epoch_end(self):
        if self.hparams.compute_bleu and self.val_predictions:  # type: ignore
            bleu_score = self._compute_bleu()
            self.log(
                "val/bleu",
                bleu_score,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

        self.val_predictions.clear()
        self.val_references.clear()

    def _compute_bleu(self) -> float:
        """Compute BLEU score using sacrebleu."""
        if not self.val_predictions or not self.val_references:
            return 0.0

        predictions_text = []
        references_text = []

        for pred_ids, ref_ids in zip(self.val_predictions, self.val_references):
            pred_text = self.tgt_tokenizer.decode(pred_ids, skip_special_tokens=True)
            ref_text = self.tgt_tokenizer.decode(ref_ids, skip_special_tokens=True)
            predictions_text.append(pred_text)
            references_text.append(ref_text)

        bleu = sacrebleu.corpus_bleu(
            predictions_text,
            [references_text],
            tokenize="13a",
        )

        if isinstance(self.logger, TensorBoardLogger) and len(predictions_text) > 0:
            example_idx = random.choice(list(range(len(predictions_text))))
            example_text = (
                f"Prediction: {predictions_text[example_idx]}\n"
                f"Reference: {references_text[example_idx]}\n"
                f"Prediction ids: {self.val_predictions[example_idx]}\n"
                f"Reference ids: {self.val_references[example_idx]}\n"
            )
            self.logger.experiment.add_text(
                "val/prediction_example",
                example_text,
                global_step=self.trainer.global_step,
            )

        return bleu.score

    def configure_optimizers(self):
        total_steps = int(self.trainer.estimated_stepping_batches)
        warmup_interval = self.hparams.warmup_interval  # type: ignore
        learning_rate = self.hparams.learning_rate  # type: ignore
        weight_decay = self.hparams.weight_decay  # type: ignore
        warmup_steps = int(total_steps * warmup_interval)

        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1e-2,
            end_factor=1.0,
            total_iters=warmup_steps,
        )

        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=1e-6,
        )

        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_steps],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
