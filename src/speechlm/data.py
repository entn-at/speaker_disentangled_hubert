import glob
import json
import math
import os
from pathlib import Path
from typing import Any, Dict

import torch
import torchaudio
from datasets import Audio, DatasetDict, Features, Sequence, Value, load_dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from ..s5hubert import S5HubertForSyllableDiscovery
from .utils import normalize_text


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
        id = [
            str(Path(item["path"]).relative_to(data_dir).with_suffix("")) if "path" in item else ""
            for item in batch["audio"]
        ]

        input_values = pad_sequence(input_values, batch_first=True)
        attention_mask = pad_sequence(attention_mask, batch_first=True)

        outputs = encoder(input_values.to(encoder.device), attention_mask.to(encoder.device))

        units = [output["units"].tolist() for output in outputs]
        durations = [output["durations"].tolist() for output in outputs]

        return {"id": id, "units": units, "durations": durations}

    return _tokenize


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

    features = Features(
        {
            "audio": Audio(sampling_rate=16000),
            "id": Value("string"),
            "units": Sequence(Value("int32")),
            "durations": Sequence(Value("int32")),
        }
    )

    swuggy_dev = load_dataset("audiofolder", data_files=swuggy_dev_paths, split="train", features=features)
    sblimp_dev = load_dataset("audiofolder", data_files=sblimp_dev_paths, split="train", features=features)
    swuggy_test = load_dataset("audiofolder", data_files=swuggy_test_paths, split="train", features=features)
    sblimp_test = load_dataset("audiofolder", data_files=sblimp_test_paths, split="train", features=features)
    tSC_test = load_dataset("audiofolder", data_files=tSC_test_paths, split="train", features=features)

    encoder = S5HubertForSyllableDiscovery.from_pretrained(config.speech2unit.model_name_or_path, device_map="cuda")

    map_kwargs = dict(batched=True, batch_size=1, remove_columns="audio")

    swuggy_dev = swuggy_dev.with_format("torch").map(get_tokenize_fn(encoder, swuggy_dev_dir), **map_kwargs)
    sblimp_dev = sblimp_dev.with_format("torch").map(get_tokenize_fn(encoder, sblimp_dev_dir), **map_kwargs)
    swuggy_test = swuggy_test.with_format("torch").map(get_tokenize_fn(encoder, swuggy_test_dir), **map_kwargs)
    sblimp_test = sblimp_test.with_format("torch").map(get_tokenize_fn(encoder, sblimp_test_dir), **map_kwargs)
    tSC_test = tSC_test.with_format("torch").map(get_tokenize_fn(encoder, tSC_dir), **map_kwargs)

    swuggy = DatasetDict({"dev": swuggy_dev, "test": swuggy_test})
    sblimp = DatasetDict({"dev": sblimp_dev, "test": sblimp_test})
    tSC = DatasetDict({"test": tSC_test})

    swuggy.push_to_hub(config.dataset.name, "sWUGGY")
    sblimp.push_to_hub(config.dataset.name, "sBLIMP")
    tSC.push_to_hub(config.dataset.name, "tSC")


def tokenize_train(config, num_shards: int = 1, shard_index: int = 0):
    dataset = load_dataset(config.dataset.train, split="train")
    dataset = dataset.shard(num_shards=num_shards, index=shard_index)

    encoder = S5HubertForSyllableDiscovery.from_pretrained(config.speech2unit.model_name_or_path, device_map="cuda")

    with open(f"{config.dataset.manifest_prefix}{shard_index}.json", "w") as f:
        for example in tqdm(dataset):
            load_path = os.path.join(config.dataset.ll_dir, example["recording"]["id"] + config.dataset.ext_audio)
            save_path = os.path.join(config.dataset.lh_dir, example["id"] + config.dataset.ext_audio)
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)

            input_values, sr = torchaudio.load(
                load_path,
                frame_offset=math.floor(16000 * max(example["start"], 0)),
                num_frames=math.floor(16000 * example["duration"]),
            )
            torchaudio.save(save_path, input_values, sr, encoding="PCM_S", bits_per_sample=16)

            outputs = encoder(input_values.to(encoder.device))

            text = example["supervisions"][0]["custom"]["texts"][1]
            text = normalize_text(text)

            manifest = {
                "audio_filepath": save_path,
                "text": text,
                "id": example["id"],
                "units": outputs[0]["units"].tolist(),
                "durations": outputs[0]["durations"].tolist(),
            }
            f.write(json.dumps(manifest) + "\n")
