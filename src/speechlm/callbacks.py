from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import DatasetDict
from transformers import TrainerCallback


class EvaluationCallback(TrainerCallback):
    def __init__(self, eval_dataset: Dict[str, DatasetDict]):
        self.eval_dataset = eval_dataset

    def get_evaluator(self, model, processing_class):
        @torch.inference_mode()
        def evaluator(batch: Dict[str, list]):
            pos_units = ["".join(f"<{unit}>" for unit in pair["pos"]) for pair in batch["units"]]
            neg_units = ["".join(f"<{unit}>" for unit in pair["neg"]) for pair in batch["units"]]
            units = pos_units + neg_units

            inputs = processing_class(units, padding=True, return_tensors="pt").to(model.device)

            logits = model(**inputs).logits.transpose(1, 2)

            labels = inputs.input_ids.masked_fill(inputs.attention_mask.bool().logical_not(), -100)
            labels = F.pad(labels, (0, 1), value=-100)
            labels = labels[:, 1:]

            # log likelihood
            scores = -F.cross_entropy(logits, labels, reduction="none")
            scores = scores.sum(dim=1) / labels.ne(-100).sum(dim=1)
            pos_scores, neg_scores = scores.chunk(2)

            metrics = torch.zeros_like(pos_scores)
            metrics[pos_scores > neg_scores] = 100
            metrics[pos_scores == neg_scores] = 50
            metrics[pos_scores < neg_scores] = 0

            batch["metrics"] = metrics.tolist()
            return batch

        return evaluator

    def on_step_end(self, args, state, control, model, processing_class, **kwargs):
        if state.global_step % args.eval_steps != 0 or not state.is_world_process_zero:
            return

        model.eval()

        map_kwargs = dict(batched=True, batch_size=args.per_device_eval_batch_size)

        sWUGGY = self.eval_dataset["sWUGGY"]["validation"].map(
            self.get_evaluator(model, processing_class), **map_kwargs
        )
        sBLIMP = self.eval_dataset["sBLIMP"]["validation"].map(
            self.get_evaluator(model, processing_class), **map_kwargs
        )

        def is_in_vocab(example):
            return example["frequency"] != 0

        def is_out_of_vocab(example):
            return example["frequency"] == 0

        pd.DataFrame(
            [
                np.mean(sWUGGY["metrics"]),
                np.mean(sWUGGY.filter(is_in_vocab)["metrics"]),
                np.mean(sWUGGY.filter(is_out_of_vocab)["metrics"]),
                np.mean(sBLIMP["metrics"]),
            ],
            index=["sWUGGY", "sWUGGY IV", "sWUGGY OOV", "sBLIMP"],
        ).to_csv(Path(args.output_dir) / f"score_dev_{state.global_step}.csv")

        model.train()

    def on_train_end(self, args, state, control, model, processing_class, **kwargs):
        if not state.is_world_process_zero:
            return

        model.eval()

        map_kwargs = dict(batched=True, batch_size=args.per_device_eval_batch_size)

        sWUGGY = self.eval_dataset["sWUGGY"]["test"].map(self.get_evaluator(model, processing_class), **map_kwargs)
        sBLIMP = self.eval_dataset["sBLIMP"]["test"].map(self.get_evaluator(model, processing_class), **map_kwargs)
        tSC = self.eval_dataset["tSC"]["test"].map(self.get_evaluator(model, processing_class), **map_kwargs)

        def is_in_vocab(example):
            return example["frequency"] != 0

        def is_out_of_vocab(example):
            return example["frequency"] == 0

        pd.DataFrame(
            [
                np.mean(sWUGGY["metrics"]),
                np.mean(sWUGGY.filter(is_in_vocab)["metrics"]),
                np.mean(sWUGGY.filter(is_out_of_vocab)["metrics"]),
                np.mean(sBLIMP["metrics"]),
                np.mean(tSC["metrics"]),
            ],
            index=["sWUGGY", "sWUGGY IV", "sWUGGY OOV", "sBLIMP", "tSC"],
        ).to_csv(Path(args.output_dir) / f"score_test_{state.global_step}.csv")


class DefrostCallback(TrainerCallback):
    def __init__(self, handle_input_embeddings, handle_output_embeddings, defrost_steps: int):
        self.handle_input_embeddings = handle_input_embeddings
        self.handle_output_embeddings = handle_output_embeddings
        self.defrost_steps = defrost_steps

    def on_step_end(self, args, state, control, model, **kwargs):
        if state.global_step == self.defrost_steps:
            self.handle_input_embeddings.remove()
            self.handle_output_embeddings.remove()
            model.model.layers.requires_grad_(True)
            model.model.norm.requires_grad_(True)
