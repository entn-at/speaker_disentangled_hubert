import glob
import os
from pathlib import Path
from typing import Any, Dict

import torch
from torch.nn.utils.rnn import pad_sequence


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


def get_tokenize_fn(encoder, data_dir):
    data_dir = Path(data_dir).resolve()

    def _tokenize(batch: Dict[str, list]):
        input_values = [item["array"] for item in batch["audio"]]
        attention_mask = [torch.ones_like(item["array"], dtype=torch.long) for item in batch["audio"]]
        batch["id"] = [
            str(Path(item["path"]).relative_to(data_dir).with_suffix("")) if "path" in item else ""
            for item in batch["audio"]
        ]

        input_values = pad_sequence(input_values, batch_first=True)
        attention_mask = pad_sequence(attention_mask, batch_first=True)

        outputs = encoder(input_values.cuda(), attention_mask.cuda())

        batch["units"] = [output["units"].tolist() for output in outputs]
        batch["durations"] = [output["durations"].tolist() for output in outputs]

        return batch

    return _tokenize


def tokenize_eval(config):
    from datasets import DatasetDict, Features, Sequence, Value, load_dataset

    from ..s5hubert import S5HubertForSyllableDiscovery

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

    swuggy_dev = load_dataset("audiofolder", data_files=swuggy_dev_paths, split="train").with_format("torch")
    sblimp_dev = load_dataset("audiofolder", data_files=sblimp_dev_paths, split="train").with_format("torch")
    swuggy_test = load_dataset("audiofolder", data_files=swuggy_test_paths, split="train").with_format("torch")
    sblimp_test = load_dataset("audiofolder", data_files=sblimp_test_paths, split="train").with_format("torch")
    tSC_test = load_dataset("audiofolder", data_files=tSC_test_paths, split="train").with_format("torch")

    encoder = S5HubertForSyllableDiscovery.from_pretrained(config.speech2unit.model_name_or_path).cuda()

    map_kwargs = dict(batched=True, batch_size=1, remove_columns="audio")

    swuggy_dev = swuggy_dev.map(get_tokenize_fn(encoder, swuggy_dev_dir), **map_kwargs)
    sblimp_dev = sblimp_dev.map(get_tokenize_fn(encoder, sblimp_dev_dir), **map_kwargs)
    swuggy_test = swuggy_test.map(get_tokenize_fn(encoder, swuggy_test_dir), **map_kwargs)
    sblimp_test = sblimp_test.map(get_tokenize_fn(encoder, sblimp_test_dir), **map_kwargs)
    tSC_test = tSC_test.map(get_tokenize_fn(encoder, tSC_dir), **map_kwargs)

    swuggy = DatasetDict({"dev": swuggy_dev, "test": swuggy_test})
    sblimp = DatasetDict({"dev": sblimp_dev, "test": sblimp_test})
    tSC = DatasetDict({"test": tSC_test})

    swuggy.push_to_hub(config.dataset.swuggy)
    sblimp.push_to_hub(config.dataset.sblimp)
    tSC.push_to_hub(config.dataset.tSC)


def tokenize_train(config, num_shards: int = 1, shard_index: int = 0):
    os.environ["HF_HOME"] = str(Path(config.dataset.HF_HOME).expanduser())

    from datasets import DatasetDict, Features, Sequence, Value, load_dataset

    from ..s5hubert import S5HubertForSyllableDiscovery

    if Path(config.dataset.wav_dir).is_dir():
        data_files = sorted(
            glob.glob(os.path.join(config.dataset.wav_dir, "**/*" + config.dataset.ext_audio), recursive=True)
        )
        dataset = load_dataset("audiofolder", data_files=data_files, split="train")
        data_dir = config.dataset.wav_dir
    else:
        dataset = load_dataset(config.dataset.wav_dir, split="train")
        data_dir = "/"

    encoder = S5HubertForSyllableDiscovery.from_pretrained(config.speech2unit.model_name_or_path).cuda()

    map_kwargs = dict(batched=True, batch_size=config.speech2unit.batch_size, remove_columns=["audio", "label"])

    dataset = dataset.shard(num_shards=num_shards, index=shard_index)
    dataset = dataset.with_format("torch")
    dataset = dataset.map(get_tokenize_fn(encoder, data_dir), **map_kwargs)
    dataset.save_to_disk(config.dataset.train, split=f"train.{shard_index}")
