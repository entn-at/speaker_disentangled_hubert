import re
from threading import Thread
from typing import Tuple

import gradio as gr
import numpy as np
import torch
import torchaudio
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from src.flow_matching import FlowMatchingWithBigVGan
from src.s5hubert import S5HubertForSyllableDiscovery

device = "cuda" if torch.cuda.is_available() else "cpu"

# download pretrained models from hugging face hub
encoder = S5HubertForSyllableDiscovery.from_pretrained("ryota-komatsu/s5-hubert", device_map=device)
decoder = FlowMatchingWithBigVGan.from_pretrained("ryota-komatsu/s5-hubert-decoder-ft", device_map=device)


def synthesize(audio: str):
    # load a waveform
    waveform, sr = torchaudio.load(audio)
    waveform = torchaudio.functional.resample(waveform, sr, 16000)

    # encode a waveform into syllabic units
    units = encoder(waveform.to(encoder.device))[0]["units"]  # [3950, 67, ..., 503]

    chunk_size = 10
    past_input_ids = torch.empty(0, dtype=units.dtype, device=units.device)
    past_spectrogram = None
    past_durations = None

    for input_ids in torch.split(units, chunk_size):
        # unit-to-speech synthesis
        outputs = decoder(torch.cat([past_input_ids, input_ids]).unsqueeze(0), past_spectrogram, past_durations)
        generated_speech = outputs.waveform.squeeze(0).cpu().numpy()

        # update past context to last chunk only
        past_input_ids = input_ids
        past_spectrogram = outputs.spectrogram
        past_durations = outputs.durations

        yield 16000, generated_speech


if __name__ == "__main__":
    with gr.Blocks(title="Streaming Speech Resynthesis") as demo:
        with gr.Row():
            audio_in = gr.Audio(type="filepath", label="Original speech")

        with gr.Row():
            btn = gr.Button("Resynthesize")
            audio_out = gr.Audio(label="Generated speech", streaming=True, autoplay=True)

        btn.click(synthesize, inputs=audio_in, outputs=audio_out)

    demo.launch()
