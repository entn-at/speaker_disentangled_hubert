import os
import subprocess
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
from datasets import load_dataset
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback, TrainingArguments

from ..s5hubert import S5HubertForSyllableDiscovery
from .data import get_collate_fn
from .eval import _eval


class EvaluationCallback(TrainerCallback):
    def __init__(
        self,
        output_dir,
        APP_DIR,
        swuggy_dev_loader: torch.utils.data.DataLoader,
        sblimp_dev_loader: torch.utils.data.DataLoader,
        swuggy_test_loader: torch.utils.data.DataLoader,
        sblimp_test_loader: torch.utils.data.DataLoader,
        tSC_test_loader: torch.utils.data.DataLoader,
    ):
        self.output_dir = output_dir
        self.APP_DIR = APP_DIR
        self.swuggy_dev_loader = swuggy_dev_loader
        self.sblimp_dev_loader = sblimp_dev_loader
        self.swuggy_test_loader = swuggy_test_loader
        self.sblimp_test_loader = sblimp_test_loader
        self.tSC_test_loader = tSC_test_loader

    def on_step_end(self, args, state, control, model, **kwargs):
        if state.global_step % args.eval_steps != 0 or not state.is_world_process_zero:
            return

        model.eval()

        os.environ["APP_DIR"] = str(Path(self.APP_DIR).expanduser())

        if not Path(self.output_dir).is_dir():
            subprocess.run(["zrc", "submission:init", "sLM21", self.output_dir], env=os.environ)

        _eval(model, self.swuggy_dev_loader, Path(self.output_dir) / "lexical/dev.txt")
        _eval(model, self.sblimp_dev_loader, Path(self.output_dir) / "syntactic/dev.txt")

        subprocess.run(
            [
                "zrc",
                "benchmarks:run",
                "sLM21",
                self.output_dir,
                "--sets",
                "dev",
                "--task",
                "lexical",
                "syntactic",
            ]
        )

        df_swuggy = pd.read_csv(Path(self.output_dir) / "scores/score_lexical_dev_by_frequency.csv", index_col=0)
        df_sblimp = pd.read_csv(Path(self.output_dir) / "scores/score_syntactic_dev_by_type.csv", index_col=0)

        swuggy_all = (df_swuggy["n"] * df_swuggy["score"]).sum() / df_swuggy["n"].sum()
        swuggy_oov = df_swuggy.loc["oov", "score"]

        df_swuggy_iv = df_swuggy[df_swuggy.index != "oov"]
        swuggy_iv = (df_swuggy_iv["n"] * df_swuggy_iv["score"]).sum() / df_swuggy_iv["n"].sum()

        sblimp = (df_sblimp["n"] * df_sblimp["score"]).sum() / df_sblimp["n"].sum()

        logs = {
            "step": state.global_step,
            "eval_sWUGGY": swuggy_all,
            "eval_sWUGGY_in_vocab": swuggy_iv,
            "eval_sWUGGY_out_of_vocab": swuggy_oov,
            "eval_sBLIMP": sblimp,
        }

        if state.epoch is not None:
            logs["epoch"] = state.epoch

        state.log_history.append(logs)

        model.train()

    def on_train_end(self, args, state, control, model, **kwargs):
        if not state.is_world_process_zero:
            return

        model.eval()

        _eval(model, self.swuggy_test_loader, Path(self.output_dir) / "lexical/test.txt")
        _eval(model, self.sblimp_test_loader, Path(self.output_dir) / "syntactic/test.txt")
        _eval(model, self.tSC_test_loader, Path(self.output_dir) / "tSC/test.txt")

        subprocess.run(
            [
                "zrc",
                "benchmarks:run",
                "sLM21",
                self.output_dir,
                "--skip-validation",
                "--sets",
                "test",
                "--task",
                "lexical",
                "syntactic",
            ]
        )

        df_swuggy = pd.read_csv(Path(self.output_dir) / "scores/score_lexical_test_by_frequency.csv", index_col=0)
        df_sblimp = pd.read_csv(Path(self.output_dir) / "scores/score_syntactic_test_by_type.csv", index_col=0)

        swuggy_all = (df_swuggy["n"] * df_swuggy["score"]).sum() / df_swuggy["n"].sum()
        swuggy_oov = df_swuggy.loc["oov", "score"]

        df_swuggy_iv = df_swuggy[df_swuggy.index != "oov"]
        swuggy_iv = (df_swuggy_iv["n"] * df_swuggy_iv["score"]).sum() / df_swuggy_iv["n"].sum()

        sblimp = (df_sblimp["n"] * df_sblimp["score"]).sum() / df_sblimp["n"].sum()

        # tSC
        data = defaultdict(dict)

        with open(Path(self.output_dir) / "tSC/test.txt") as f:
            for line in f:
                name, score = line.strip().split()
                n, id_, correct = name.split("_")
                score = float(score)
                data[id_][correct] = score

        data = [{"id": id_, "correct": data[id_]["correct"], "incorrect": data[id_]["incorrect"]} for id_ in data]
        df_tSC = pd.DataFrame(data)
        tSC = (df_tSC["correct"] >= df_tSC["incorrect"]).mean()

        pd.DataFrame(
            [swuggy_all, swuggy_iv, swuggy_oov, sblimp, tSC],
            index=["sWUGGY", "sWUGGY in-vocab", "sWUGGY out-of-vocab", "sBLIMP", "tSC"],
        ).to_csv(Path(self.output_dir) / "scores/score.csv")


class DefrostCallback(TrainerCallback):
    def __init__(self, handle_input_embeddings, handle_output_embeddings):
        self.handle_input_embeddings = handle_input_embeddings
        self.handle_output_embeddings = handle_output_embeddings

    def on_step_end(self, args, state, control, model, **kwargs):
        if state.global_step == args.warmup_steps:
            self.handle_input_embeddings.remove()
            self.handle_output_embeddings.remove()
            model.requires_grad_(True)


def train(config):
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model.name)
    vocab = tokenizer.get_vocab()
    vocab_size = (
        S5HubertForSyllableDiscovery.from_pretrained(config.speech2unit.model_name_or_path)
        .quantizer2.max()
        .int()
        .item()
        + 1
    )
    units = [f"<{unit}>" for unit in range(vocab_size)]
    for unit in units:
        assert unit not in vocab
    tokenizer.add_tokens(units)

    # Datasets
    train_dataset = load_dataset(config.dataset.name, "Libri-Light", split="train", keep_in_memory=True)
    swuggy = load_dataset(config.dataset.name, "sWUGGY")
    sblimp = load_dataset(config.dataset.name, "sBLIMP")
    tSC = load_dataset(config.dataset.name, "tSC")

    swuggy_dev_loader = torch.utils.data.DataLoader(
        swuggy["dev"],
        batch_size=config.training_args.per_device_eval_batch_size,
        collate_fn=get_collate_fn(tokenizer),
    )
    sblimp_dev_loader = torch.utils.data.DataLoader(
        sblimp["dev"],
        batch_size=config.training_args.per_device_eval_batch_size,
        collate_fn=get_collate_fn(tokenizer),
    )
    swuggy_test_loader = torch.utils.data.DataLoader(
        swuggy["test"],
        batch_size=config.training_args.per_device_eval_batch_size,
        collate_fn=get_collate_fn(tokenizer),
    )
    sblimp_test_loader = torch.utils.data.DataLoader(
        sblimp["test"],
        batch_size=config.training_args.per_device_eval_batch_size,
        collate_fn=get_collate_fn(tokenizer),
    )
    tSC_test_loader = torch.utils.data.DataLoader(
        tSC["test"],
        batch_size=config.training_args.per_device_eval_batch_size,
        collate_fn=get_collate_fn(tokenizer),
    )

    # Model
    model = AutoModelForCausalLM.from_pretrained(config.model.name)
    model.resize_token_embeddings(len(tokenizer))
    model.requires_grad_(False)
    model.get_input_embeddings().requires_grad_(True)
    model.get_output_embeddings().requires_grad_(True)
    handle_input_embeddings = model.get_input_embeddings().register_hook(
        lambda grad: torch.cat([torch.zeros_like(grad[: len(vocab)]), grad[len(vocab) :]])
    )
    handle_output_embeddings = model.get_output_embeddings().register_hook(
        lambda grad: torch.cat([torch.zeros_like(grad[: len(vocab)]), grad[len(vocab) :]])
    )

    training_args = TrainingArguments(**OmegaConf.to_container(config.training_args))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        data_collator=get_collate_fn(tokenizer),
        callbacks=[
            EvaluationCallback(
                config.training_args.output_dir,
                config.dataset.APP_DIR,
                swuggy_dev_loader,
                sblimp_dev_loader,
                swuggy_test_loader,
                sblimp_test_loader,
                tSC_test_loader,
            ),
            DefrostCallback(handle_input_embeddings, handle_output_embeddings),
        ],
    )
    trainer.train(resume_from_checkpoint=True)
