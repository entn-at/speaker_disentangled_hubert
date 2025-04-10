import json
import re
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torchaudio

from ...sdhubert.utils.syllable import BoundaryDetectionEvaluator, match_cluster
from ..utils.misc import compute_cluster_purity, compute_mutual_info, compute_syllable_purity


def evaluate(config):
    # load test dataset
    with open(config.dataset.test_alignment) as f:
        dataset = json.load(f)

    # load clustering model
    if config.model.model_type != "sylboost":
        quantizer1 = torch.from_numpy(joblib.load(config.path.quantizer1).cluster_centers_).cuda()
        quantizer2 = torch.from_numpy(np.load(config.path.quantizer2)).cuda()

    matching_counter = Counter()
    syllable_counter = Counter()

    segment_dir = Path(config.path.segment_dir)

    # unit frequency
    total_seconds = 0
    num_units = 0

    # maximum weight matching between ground truth syllables and predicted units
    for sample in dataset.values():
        # load predicted segments
        segment_name = sample["file_name"].replace(".flac", ".npy")
        segment_path = segment_dir / segment_name
        ckpt = np.load(segment_path, allow_pickle=True)[()]

        # segment boundary
        ref_boundary = np.array([[float(ref["start"]), float(ref["end"])] for ref in sample["syllables"]])
        hyp_boundary = ckpt["segments"]

        # maximize temporal intersection-over-union (IoU)
        ref_indices, hyp_indices = match_cluster(ref_boundary, hyp_boundary)

        # syllables
        ref_syllables = np.array([re.sub("[0-3]", "", ref["label"]) for ref in sample["syllables"]])
        if config.model.model_type != "sylboost":
            hyp_syllables = torch.cdist(torch.from_numpy(ckpt["segment_features"]).cuda(), quantizer1).argmin(1)
            hyp_syllables = quantizer2[hyp_syllables].cpu().numpy()
        else:
            hyp_syllables = ckpt["units"]

        matching_counter.update(zip(ref_syllables[ref_indices], hyp_syllables[hyp_indices]))
        syllable_counter.update(ref_syllables[ref_indices])

        # unit frequency
        audio_path = Path(config.dataset.root) / "LibriSpeech" / sample["file_name"]
        audio_path = str(audio_path)
        metadata = torchaudio.info(audio_path)
        total_seconds += metadata.num_frames / metadata.sample_rate
        num_units += len(hyp_boundary)

    # joint prob
    p_xy = np.zeros((len(syllable_counter), config.n_clusters.step2))
    p_xy = pd.DataFrame(p_xy, index=list(syllable_counter))
    for (ref, hyp), count in matching_counter.items():
        p_xy.loc[ref, hyp] = count
    p_xy = p_xy.to_numpy()
    p_xy = p_xy / np.sum(p_xy)

    clustering_results = {
        "syllable_purity": compute_syllable_purity(p_xy),
        "cluster_purity": compute_cluster_purity(p_xy),
        "mutual_info": compute_mutual_info(p_xy),
    }

    segmentation_results = BoundaryDetectionEvaluator(
        config.path.segment_dir,
        config.dataset.test_alignment,
        config.dataset.dev_alignment,
        tolerance=0.05,
        max_val_num=None,
    ).evaluate()

    unit_frequency = num_units / total_seconds

    results = {
        "segmentation": segmentation_results,
        "clustering": clustering_results,
        "unit_frequency": unit_frequency,
    }

    def round_float(results, ndigits: int = 3):
        if isinstance(results, dict):
            return {k: round_float(v) for k, v in results.items()}
        elif isinstance(results, float):
            return round(results, ndigits)
        else:
            return results

    results = round_float(results)

    Path(config.path.result).parent.mkdir(parents=True, exist_ok=True)
    with open(config.path.result, "w") as f:
        json.dump(results, f, indent=2)
