import subprocess
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from .data import get_collate_fn


def evaluate(config):
    model = AutoModelForCausalLM.from_pretrained(config.model.path).cuda()
    tokenizer = AutoTokenizer.from_pretrained(config.model.path)

    swuggy = load_dataset(config.dataset.name, "sWUGGY", split="test")
    swuggy_loader = torch.utils.data.DataLoader(
        swuggy,
        batch_size=config.dataloader.batch_size_per_device,
        collate_fn=get_collate_fn(tokenizer),
    )

    sblimp = load_dataset(config.dataset.name, "sBLIMP", split="test")
    sblimp_loader = torch.utils.data.DataLoader(
        sblimp,
        batch_size=config.dataloader.batch_size_per_device,
        collate_fn=get_collate_fn(tokenizer),
    )

    tSC = load_dataset(config.dataset.name, "tSC", split="test")
    tSC_loader = torch.utils.data.DataLoader(
        tSC,
        batch_size=config.dataloader.batch_size_per_device,
        collate_fn=get_collate_fn(tokenizer),
    )

    _eval(model, swuggy_loader, Path(config.dataset.result_dir) / "lexical/test.txt")
    _eval(model, sblimp_loader, Path(config.dataset.result_dir) / "syntactic/test.txt")
    _eval(model, tSC_loader, Path(config.dataset.result_dir) / "tSC/test.txt")

    subprocess.run(
        [
            "zrc",
            "benchmarks:run",
            "sLM21",
            config.dataset.result_dir,
            "--skip-validation",
            "--sets",
            "test",
            "--task",
            "lexical",
            "syntactic",
        ]
    )

    df_swuggy = pd.read_csv(Path(config.dataset.result_dir) / "scores/score_lexical_test_by_frequency.csv", index_col=0)
    df_sblimp = pd.read_csv(Path(config.dataset.result_dir) / "scores/score_syntactic_test_by_type.csv", index_col=0)

    swuggy_all = (df_swuggy["n"] * df_swuggy["score"]).sum() / df_swuggy["n"].sum()
    swuggy_oov = df_swuggy.loc["oov", "score"]

    df_swuggy_iv = df_swuggy[df_swuggy.index != "oov"]
    swuggy_iv = (df_swuggy_iv["n"] * df_swuggy_iv["score"]).sum() / df_swuggy_iv["n"].sum()

    sblimp = (df_sblimp["n"] * df_sblimp["score"]).sum() / df_sblimp["n"].sum()

    # tSC
    data = defaultdict(dict)

    with open(Path(config.dataset.result_dir) / "tSC/test.txt") as f:
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
        index=["sWUGGY all", "sWUGGY in-vocab", "sWUGGY out-of-vocab", "sBLIMP", "tSC"],
    ).to_csv(Path(config.dataset.result_dir) / "scores/score.csv")


@torch.inference_mode()
def _eval(
    model,
    loader: torch.utils.data.DataLoader,
    out_file,
):
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        for batch in loader:
            # Speech LM
            input_ids = batch["input_ids"].cuda()
            labels = batch["labels"].cuda()
            logits = model(input_ids=input_ids, labels=labels).logits.transpose(1, 2)

            labels = F.pad(labels, (0, 1), value=-100)
            shifted_labels = labels[:, 1:]

            scores = -F.cross_entropy(logits, shifted_labels, reduction="none")
            scores = scores.sum(dim=1) / scores.ne(0).sum(dim=1)
            scores = scores.tolist()

            for id, score in zip(batch["id"], scores):
                f.write(f"{id} {score}\n")
