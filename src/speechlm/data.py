import glob
import os
from pathlib import Path
from typing import Any, Dict

import torch
from datasets import Dataset, DatasetDict, Features, Sequence, Value, load_dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from ..s5hubert import S5HubertForSyllableDiscovery


def get_audio_collator(data_dir):
    data_dir = Path(data_dir).resolve()

    def collate_fn(batch):
        input_values = [item["audio"]["array"] for item in batch]
        attention_mask = [torch.ones_like(item["audio"]["array"], dtype=torch.long) for item in batch]
        id = [str(Path(item["audio"]["path"]).relative_to(data_dir).with_suffix("")) for item in batch]

        input_values = pad_sequence(input_values, batch_first=True)
        attention_mask = pad_sequence(attention_mask, batch_first=True)

        return {"input_values": input_values, "attention_mask": attention_mask, "id": id}

    return collate_fn


def get_collate_fn(
    tokenizer,
):
    def collate_fn(batch) -> Dict[str, Any]:
        input_ids = []
        id = []

        for item in batch:
            units = item["units"]
            input_ids.append("".join(f"<{unit}>" for unit in units))
            id.append(item["id"])

        inputs = tokenizer(input_ids, padding=True, return_tensors="pt")

        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        labels = input_ids.masked_fill(attention_mask.bool().logical_not(), -100)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "id": id,
        }

    return collate_fn


def _tokenize(
    encoder,
    data_loader: torch.utils.data.DataLoader,
):
    dataset = []

    for item in tqdm(data_loader):
        outputs = encoder(item["input_values"].cuda(), item["attention_mask"].cuda())

        for id, output in zip(item["id"], outputs):
            example = {
                "id": id,
                "units": output["units"].tolist(),
                "durations": output["durations"].tolist(),
            }
            dataset.append(example)

    features = Features(
        {
            "id": Value("string"),
            "units": Sequence(Value("int32")),
            "durations": Sequence(Value("int32")),
        }
    )

    return Dataset.from_list(dataset, features=features)


def tokenize_eval(config):
    app_dir = Path(config.dataset.APP_DIR).expanduser()
    tSC_dir = Path(config.dataset.tSC_DIR)

    swuggy_dev_dir = app_dir / "datasets/sLM21-dataset/lexical/dev"
    sblimp_dev_dir = app_dir / "datasets/sLM21-dataset/syntactic/dev"
    swuggy_test_dir = app_dir / "datasets/sLM21-dataset/lexical/test"
    sblimp_test_dir = app_dir / "datasets/sLM21-dataset/syntactic/test"

    swuggy_dev_paths = glob.glob(os.path.join(swuggy_dev_dir, "*.wav"))
    sblimp_dev_paths = glob.glob(os.path.join(sblimp_dev_dir, "*.wav"))
    swuggy_test_paths = glob.glob(os.path.join(swuggy_test_dir, "*.wav"))
    sblimp_test_paths = glob.glob(os.path.join(sblimp_test_dir, "*.wav"))
    tSC_test_paths = glob.glob(os.path.join(tSC_dir, "*.wav"))

    swuggy_dev_set = load_dataset("audiofolder", data_files=swuggy_dev_paths, split="train").with_format("torch")
    sblimp_dev_set = load_dataset("audiofolder", data_files=sblimp_dev_paths, split="train").with_format("torch")
    swuggy_test_set = load_dataset("audiofolder", data_files=swuggy_test_paths, split="train").with_format("torch")
    sblimp_test_set = load_dataset("audiofolder", data_files=sblimp_test_paths, split="train").with_format("torch")
    tSC_test_set = load_dataset("audiofolder", data_files=tSC_test_paths, split="train").with_format("torch")

    swuggy_dev_loader = torch.utils.data.DataLoader(swuggy_dev_set, collate_fn=get_audio_collator(swuggy_dev_dir))
    sblimp_dev_loader = torch.utils.data.DataLoader(sblimp_dev_set, collate_fn=get_audio_collator(sblimp_dev_dir))
    swuggy_test_loader = torch.utils.data.DataLoader(swuggy_test_set, collate_fn=get_audio_collator(swuggy_test_dir))
    sblimp_test_loader = torch.utils.data.DataLoader(sblimp_test_set, collate_fn=get_audio_collator(sblimp_test_dir))
    tSC_test_loader = torch.utils.data.DataLoader(tSC_test_set, collate_fn=get_audio_collator(tSC_dir))

    encoder = S5HubertForSyllableDiscovery.from_pretrained(config.speech2unit.model_name_or_path).cuda()

    swuggy_dev = _tokenize(encoder, swuggy_dev_loader)
    sblimp_dev = _tokenize(encoder, sblimp_dev_loader)
    swuggy_test = _tokenize(encoder, swuggy_test_loader)
    sblimp_test = _tokenize(encoder, sblimp_test_loader)
    tSC_test = _tokenize(encoder, tSC_test_loader)

    swuggy = DatasetDict({"dev": swuggy_dev, "test": swuggy_test})
    sblimp = DatasetDict({"dev": sblimp_dev, "test": sblimp_test})
    tSC = DatasetDict({"test": tSC_test})

    swuggy.push_to_hub(config.dataset.swuggy)
    sblimp.push_to_hub(config.dataset.sblimp)
    tSC.push_to_hub(config.dataset.tSC)


def tokenize_train(config, num_shards: int = 1, shard_index: int = 0):
    if Path(config.dataset.wav_dir).is_dir():
        data_files = sorted(
            glob.glob(os.path.join(config.dataset.wav_dir, "**/*" + config.dataset.ext_audio), recursive=True)
        )
        dataset = load_dataset("audiofolder", data_files=data_files, split="train", cache_dir=config.dataset.cache_dir)
        data_dir = config.dataset.wav_dir
    else:
        dataset = load_dataset(config.dataset.wav_dir, split="train", cache_dir=config.dataset.cache_dir)
        data_dir = "/"

    dataset = dataset.shard(num_shards=num_shards, index=shard_index)
    dataset = dataset.with_format("torch")

    loader = torch.utils.data.DataLoader(
        dataset,
        config.speech2unit.batch_size,
        num_workers=config.speech2unit.num_workers,
        collate_fn=get_audio_collator(data_dir),
    )

    encoder = S5HubertForSyllableDiscovery.from_pretrained(config.speech2unit.model_name_or_path).cuda()

    dataset = _tokenize(encoder, loader)
    dataset.save_to_disk(config.dataset.train, split=f"train.{shard_index}")
