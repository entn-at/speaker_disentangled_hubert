import glob
import json
import math
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np
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

            text = example["supervisions"][0]["custom"]["texts"][0]
            text = normalize_text(text)

            manifest = {
                "audio_filepath": save_path,
                "text": text,
                "id": example["id"],
                "units": outputs[0]["units"].tolist(),
                "durations": outputs[0]["durations"].tolist(),
            }
            f.write(json.dumps(manifest) + "\n")


def align_train(config, num_shards: int = 1):
    data_files = [f"{config.dataset.manifest_prefix}{shard_index}.json" for shard_index in range(num_shards)]
    dataset = load_dataset("json", data_files=data_files, split="train")

    id_to_aligned_text = dict()

    for shard_index in range(num_shards):
        manifest_prefix = Path(config.dataset.manifest_prefix).stem
        manifest_with_output_file_paths = os.path.join(
            config.dataset.lh_dir, f"shard{shard_index}/{manifest_prefix}{shard_index}_with_output_file_paths.json"
        )

        with open(manifest_with_output_file_paths) as f:
            for example in f:
                example = json.loads(example.strip())

                id_ = str(Path(example["audio_filepath"]).relative_to(config.dataset.lh_dir).with_suffix(""))
                aligned_text = []

                if "words_level_ctm_filepath" in example:
                    with open(example["words_level_ctm_filepath"]) as g:
                        for line in g:
                            _, _, start, duration, word, _, _, _ = line.split(" ")

                            aligned_text.append(
                                {
                                    "start_time": float(start),
                                    "end_time": float(start) + float(duration),
                                    "word": " " + word,
                                }
                            )

                id_to_aligned_text[id_] = aligned_text

    def add_aligned_units(example):
        example["aligned_text"] = id_to_aligned_text[example["id"]]

        if not example["aligned_text"]:
            end_time = sum(example["durations"]) * 0.02
            example["aligned_units"] = [
                {"start_time": 0.0, "end_time": end_time, "units": example["units"], "text": example["text"]}
            ]
            return example

        unit_timestamps = np.cumsum(example["durations"]) * 0.02
        word_timestamps = sorted(
            {item["start_time"] for item in example["aligned_text"]}
            | {item["end_time"] for item in example["aligned_text"]}
        )
        aligned_timestamps = sorted(
            set(unit_timestamps) & set(word_timestamps) | set([max(unit_timestamps[-1], word_timestamps[-1])])
        )

        aligned_units = []
        start_time = 0

        for end_time in aligned_timestamps:
            units = [
                unit
                for unit, unit_end_time in zip(example["units"], unit_timestamps)
                if start_time < unit_end_time <= end_time
            ]
            text = "".join(
                item["word"]
                for item in example["aligned_text"]
                if start_time <= item["start_time"] and item["end_time"] <= end_time
            )

            aligned_units.append({"start_time": start_time, "end_time": end_time, "units": units, "text": text})
            start_time = end_time

        example["aligned_units"] = aligned_units

        return example

    dataset = dataset.map(add_aligned_units, remove_columns="audio_filepath")
    dataset = DatasetDict({"train": dataset})
    dataset.push_to_hub(config.dataset.name, "Libri-Light")
