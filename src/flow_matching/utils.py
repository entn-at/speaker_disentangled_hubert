import random

import numpy as np
import torch

from ..s5hubert import S5HubertForSyllableDiscovery


def fix_random_seed(seed=0):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_lr_schedule(
    optimizer,
    total_steps: int,
    warmup_steps: int = 5000,
    base_lr: float = 1e-3,
    min_lr: float = 1e-4,
) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_schedule(current_step: int) -> float:
        if current_step < warmup_steps:
            return (min_lr + (base_lr - min_lr) * current_step / warmup_steps) / base_lr
        else:
            progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
            return (min_lr + (base_lr - min_lr) * (1 - progress)) / base_lr

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)


def get_input_embeddings(model_name_or_path: str, freeze: bool = True) -> torch.nn.Embedding:
    model = S5HubertForSyllableDiscovery.from_pretrained(model_name_or_path)
    vocab_size = model.quantizer2.max().int().item() + 1
    embeddings = torch.zeros(vocab_size, model.quantizer1.shape[1])

    for idx, unit in enumerate(model.quantizer2):
        embeddings[unit] += model.quantizer1[idx]

    embeddings /= torch.bincount(model.quantizer2).unsqueeze(1)
    embeddings = torch.cat((embeddings, torch.zeros(1, model.quantizer1.shape[1])))

    return torch.nn.Embedding.from_pretrained(embeddings, freeze=freeze, padding_idx=vocab_size)
