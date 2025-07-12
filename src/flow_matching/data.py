import glob
import os
from pathlib import Path
from typing import Any, Dict, Optional

import librosa
import torch
import torchaudio
from datasets import Array2D, Audio, Features, Sequence, Value, load_dataset
from torch.nn.utils.rnn import pad_sequence

from ..bigvgan.data import mel_spectrogram
from ..s5hubert import S5HubertForSyllableDiscovery


def get_collate_fn(
    wav_dir: Optional[str] = None,
    ext_audio: str = ".wav",
):
    def parse_item(item: Dict[str, Any]):
        input_ids = item["units"] + 1  # 0: pad
        spectrogram_labels = item["spectrogram"]
        durations = item["durations"]
        transcript = item["transcript"]
        id = item["id"]
        wav = torch.zeros(1)  # dummy

        if wav_dir:
            wav_path = os.path.join(wav_dir, id + ext_audio)
            wav, sr = torchaudio.load(wav_path)
            wav = wav.squeeze(0)

        return input_ids, spectrogram_labels, durations, transcript, id, wav

    def collate_fn(batch):
        input_ids = []
        spectrogram_labels = []
        duration_labels = []
        transcripts = []
        names = []
        input_values = []

        for item in batch:
            units, spectrogram, durations, transcript, id, wav = parse_item(item)
            input_ids.append(units)
            spectrogram_labels.append(spectrogram)
            duration_labels.append(durations)
            transcripts.append(transcript)
            names.append(id)
            input_values.append(wav)

        input_ids = pad_sequence(input_ids, batch_first=True)
        spectrogram_labels = pad_sequence(spectrogram_labels, batch_first=True, padding_value=-100)
        duration_labels = pad_sequence(duration_labels, batch_first=True)
        input_values = pad_sequence(input_values, batch_first=True)

        return {
            "input_ids": input_ids,
            "spectrogram_labels": spectrogram_labels,
            "duration_labels": duration_labels,
            "transcripts": transcripts,
            "names": names,
            "input_values": input_values,
        }

    return collate_fn


def tokenize(config):
    encoder = S5HubertForSyllableDiscovery.from_pretrained(config.speech2unit.model_name_or_path, device_map="cuda")

    features = Features(
        {
            "audio": Audio(),
            "id": Value("string"),
            "units": Sequence(Value("int32")),
            "durations": Sequence(Value("int32")),
            "transcript": Value("string"),
            "spectrogram": Array2D(shape=(None, 80), dtype="float32"),
        }
    )

    # LibriTTS-R
    data_files = {
        "train": glob.glob(os.path.join(config.dataset.wav_dir, "train-*/**/*.wav"), recursive=True),
        "dev": glob.glob(os.path.join(config.dataset.wav_dir, "dev-clean/**/*.wav"), recursive=True),
    }
    dataset = load_dataset("audiofolder", data_files=data_files, features=features)
    dataset = dataset.with_format("torch").cast_column("audio", Audio(sampling_rate=16000))
    dataset = dataset.map(get_tokenize_fn(encoder, config.dataset.wav_dir, ".normalized.txt"), remove_columns="audio")
    dataset.push_to_hub(config.dataset.name, "LibriTTS-R")

    # Hi-Fi-CAPTAIN
    data_files = {"train": glob.glob(os.path.join(config.dataset.hfc_dir, "**/*.wav"), recursive=True)}
    dataset = load_dataset("audiofolder", data_files=data_files, features=features)
    dataset = dataset.with_format("torch").cast_column("audio", Audio(sampling_rate=16000))
    dataset = dataset.map(get_tokenize_fn(encoder, config.dataset.hfc_dir, ""), remove_columns="audio")
    dataset.push_to_hub(config.dataset.name, "Hi-Fi-CAPTAIN")

    # DailyTalk
    data_files = {"train": glob.glob(os.path.join(config.dataset.dailytalk_dir, "**/*.wav"), recursive=True)}
    dataset = load_dataset("audiofolder", data_files=data_files, features=features)
    dataset = dataset.with_format("torch").cast_column("audio", Audio(sampling_rate=16000))
    dataset = dataset.map(get_tokenize_fn(encoder, config.dataset.dailytalk_dir, ".txt"), remove_columns="audio")
    dataset.push_to_hub(config.dataset.name, "DailyTalk")


def get_tokenize_fn(encoder, data_dir, ext_txt: str = ".normalized.txt"):
    data_dir = Path(data_dir).resolve()

    def _tokenize(example):
        input_values = example["audio"]["array"].numpy()
        input_values = librosa.effects.trim(input_values, top_db=20)[0]
        input_values = torch.from_numpy(input_values)
        input_values = input_values.cuda()
        input_values = input_values / input_values.abs().max() * 0.95
        input_values = input_values.unsqueeze(0)

        spectrogram_labels = mel_spectrogram(input_values).squeeze(0)  # (80, len)
        spectrogram_labels = spectrogram_labels.transpose(0, 1)  # (len, 80)
        spectrogram_labels = spectrogram_labels.cpu().tolist()

        outputs = encoder(input_values)

        id = str(Path(example["audio"]["path"]).relative_to(data_dir).with_suffix(""))
        txt_path = Path(example["audio"]["path"]).with_suffix(ext_txt)

        transcript = ""
        if txt_path.is_file():
            with open(txt_path) as g:
                transcript = g.read().rstrip()

        example.update(
            {
                "id": id,
                "units": outputs[0]["units"].tolist(),
                "durations": outputs[0]["durations"].tolist(),
                "transcript": transcript,
                "spectrogram": spectrogram_labels,
            }
        )
        return example

    return _tokenize
