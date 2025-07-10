# S5-HuBERT: Self-Supervised Speaker-Separated Syllable HuBERT

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org)
[![colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ryota-komatsu/speaker_disentangled_hubert/blob/main/demo.ipynb)
[![arXiv](https://img.shields.io/badge/arXiv-2409.10103-<COLOR>.svg?logo=arXiv)](https://arxiv.org/abs/2409.10103)
[![model](https://img.shields.io/badge/%F0%9F%A4%97-Model-blue)](https://huggingface.co/ryota-komatsu/s5-hubert)
[![dataset](https://img.shields.io/badge/%F0%9F%A4%97-Datasets-blue)](https://huggingface.co/datasets/ryota-komatsu/s5-hubert)

This is the official repository of the IEEE SLT 2024 paper [Self-Supervised Syllable Discovery Based on Speaker-Disentangled HuBERT](https://arxiv.org/abs/2409.10103).

## Setup

```shell
sudo apt install git-lfs  # for UTMOS

conda create -y -n py310 -c pytorch -c nvidia -c conda-forge python=3.10.18 pip=24.0 faiss-gpu=1.11.0
conda activate py310
pip install -r requirements/requirements.txt

sh scripts/setup.sh
```

## Usage: encoding waveforms into pseudo-syllabic units

![](figures/usage.png)

```python
import torchaudio

from src.flow_matching import FlowMatchingWithBigVGan
from src.s5hubert import S5HubertForSyllableDiscovery

wav_path = "/path/to/wav"

# download pretrained models from hugging face hub
encoder = S5HubertForSyllableDiscovery.from_pretrained("ryota-komatsu/s5-hubert", device_map="cuda")
decoder = FlowMatchingWithBigVGan.from_pretrained("ryota-komatsu/s5-hubert-decoder", device_map="cuda")

# load a waveform
waveform, sr = torchaudio.load(wav_path)
waveform = torchaudio.functional.resample(waveform, sr, 16000)

# encode a waveform into syllabic units
outputs = encoder(waveform.cuda())

# syllabic units
units = outputs[0]["units"]  # [3950, 67, ..., 503]
units = units.unsqueeze(0) + 1  # 0: pad

# unit-to-speech synthesis
audio_values = decoder(units)
```

## Demo

Google Colab demo is found [here](https://colab.research.google.com/github/ryota-komatsu/speaker_disentangled_hubert/blob/main/demo.ipynb).

## Models

![](figures/model.png)

You can download a pretrained model from [Hugging Face](https://huggingface.co/ryota-komatsu/s5-hubert).

## Data Preparation

You can download datasets under `dataset_root`.
```shell
dataset_root=data  # be consistent with dataset.root in a config file

sh scripts/download_librispeech.sh ${dataset_root}
sh scripts/download_libritts.sh ${dataset_root}
sh scripts/download_librilight.sh ${dataset_root}  # 7TB
sh scripts/download_slm21.sh  # download sWUGGY and sBLIMP
```

> [!TIP]
> If you already have LibriSpeech, you can use it by editing [a config file](configs/speech2unit/default.yaml#L13);
> ```yaml
> dataset:
>   root: "/path/to/LibriSpeech/root" # ${dataset.root}/LibriSpeech/train-clean-100, train-clean-360, ...
> ```

Check the directory structure
```
dataset.root in a config file
└── LibriSpeech/
    ├── train-clean-100/
    ├── train-clean-360/
    ├── train-other-500/
    ├── dev-clean/
    ├── dev-other/
    ├── test-clean/
    ├── test-other/
    └── SPEAKERS.TXT
```

## Syllable discovery

```shell
python main_speech2unit.py --config configs/speech2unit/default.yaml
```

To run only a sub-task (train, syllable_segmentation, quantize, or evaluate), specify it as an argument.

```shell
python main_speech2unit.py train --config configs/speech2unit/default.yaml
```

## Unit-to-speech synthesis

```shell
python main_unit2speech.py train_flow_matching --config=configs/unit2speech/default.yaml
```

## Speech language modeling

```shell
GROUP_NAME=

qsub -g ${GROUP_NAME} scripts/run_speechlm.bash configs/speechlm/default.yaml
python main_speechlm.py evaluate --config=configs/speechlm/default.yaml
```

## Citation

```bibtex
@inproceedings{Komatsu_Self-Supervised_Syllable_Discovery_2024,
  author    = {Komatsu, Ryota and Shinozaki, Takahiro},
  title     = {Self-Supervised Syllable Discovery Based on Speaker-Disentangled HuBERT},
  year      = {2024},
  month     = {Dec.},
  booktitle = {IEEE Spoken Language Technology Workshop},
  pages     = {1131--1136},
  doi       = {10.1109/SLT61566.2024.10832325},
}
```