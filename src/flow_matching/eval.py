import sys
import warnings
from pathlib import Path

import jiwer
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from .data import get_collate_fn
from .models import FlowMatchingWithBigVGan

sys.path.append("src/utmos")
warnings.simplefilter("ignore", FutureWarning)
warnings.simplefilter("ignore", DeprecationWarning)
from ..utmos.score import Score


@torch.inference_mode()
def evaluate(config):
    dataset = load_dataset(config.dataset.name, split="test", keep_in_memory=True).with_format("torch")
    loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=get_collate_fn(
            wav_dir=config.dataset.wav_dir,
            ext_audio=config.dataset.ext_audio,
        ),
    )

    model = FlowMatchingWithBigVGan.load_pretrained(
        config.flow_matching.model_name_or_path,
        config.vocoder.model_name_or_path,
    ).cuda()

    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    asr = AutoModelForSpeechSeq2Seq.from_pretrained(
        config.asr.model_name_or_path,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        device_map="cuda",
    )
    processor = AutoProcessor.from_pretrained(config.asr.model_name_or_path)
    pipe = pipeline(
        "automatic-speech-recognition",
        model=asr,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
    )

    scorer = Score(ckpt_path="src/utmos/epoch=3-step=7459.ckpt", input_sample_rate=16000, device="cuda")

    hyps = []
    refs = []
    hyp_scores = []
    ref_scores = []

    for batch in tqdm(loader):
        hyp_wav = model(batch["input_ids"].cuda())[0]
        ref_wav = batch["input_values"].cuda()

        hyp_score = scorer.score(hyp_wav)
        ref_score = scorer.score(ref_wav)

        hyp = pipe(hyp_wav.cpu().squeeze(0).numpy(), generate_kwargs={"language": "english"}, return_timestamps=True)
        ref = pipe(ref_wav.cpu().squeeze(0).numpy(), generate_kwargs={"language": "english"}, return_timestamps=True)

        hyps.append(hyp["text"])
        refs.append(ref["text"])
        hyp_scores.append(hyp_score)
        ref_scores.append(ref_score)

    transcripts = [processor.tokenizer.normalize(transcript) for transcript in loader.dataset["transcript"]]
    hyps = [processor.tokenizer.normalize(hyp) for hyp in hyps]
    refs = [processor.tokenizer.normalize(ref) for ref in refs]

    wer_hyp = jiwer.wer(transcripts, hyps) * 100
    cer_hyp = jiwer.cer(transcripts, hyps) * 100
    mos_hyp = np.mean(hyp_scores)

    wer_ref = jiwer.wer(transcripts, refs) * 100
    cer_ref = jiwer.cer(transcripts, refs) * 100
    mos_ref = np.mean(ref_scores)

    Path(config.path.result).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [wer_hyp, cer_hyp, mos_hyp, wer_ref, cer_ref, mos_ref],
        index=["WER (hyp)", "CER (hyp)", "MOS (hyp)", "WER (ref)", "CER (ref)", "MOS (ref)"],
    ).to_csv(config.path.result, float_format="%.2f")
