import random
import re

import numpy as np
import torch


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


def normalize_text(s: str) -> str:
    s = s.replace("‘", "'")
    s = s.replace("’", "'")
    tokens = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',.?")
    s_list = [x if x in tokens else " " for x in s]
    s = " ".join("".join(s_list).split()).strip()

    s = s.lower()

    s = re.sub(r"\ban'\s", "and ", s)
    s = re.sub(r"\baround'em\b", "around them", s)
    s = re.sub(r"\bbewilder'd\b", "bewildered", s)
    s = re.sub(r"\bcap'n\b", "captain", s)
    s = re.sub(r"\bcharm'em\b", "charm them", s)
    s = re.sub(r"\bdiff'rence\b", "difference", s)
    s = re.sub(r"\be'en\b", "even", s)
    s = re.sub(r"\bfetchin'\s", "fetching ", s)
    s = re.sub(r"\bgive'em\b", "give them", s)
    s = re.sub(r"\binv'tation", "invitation", s)
    s = re.sub(r"\bjes'\s", "just ", s)
    s = re.sub(r"\bmore'n\b", "more than", s)
    s = re.sub(r"\bof'em\b", "of them", s)
    s = re.sub(r"\bop'ning\b", "opening", s)
    s = re.sub(r"\bpass'd\b", "passed", s)
    s = re.sub(r"\bp'raps\b", "perhaps", s)
    s = re.sub(r"\bshorten'd\b", "shortened", s)
    s = re.sub(r"\bs'pose\b", "suppose", s)
    s = re.sub(r"\btellin'\s", "telling ", s)
    s = re.sub(r"\bvisitin'\s", "visiting ", s)
    s = re.sub(r"\bwith'em\b", "with them", s)

    s = s.capitalize()

    s = re.sub(r"\bi\b", "I", s)
    s = re.sub(r"\b '\b", "'", s)

    return s
