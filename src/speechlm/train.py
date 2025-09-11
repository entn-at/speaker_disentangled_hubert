import deepspeed
import numpy as np
import torch
from datasets import load_dataset
from deepspeed.utils.tensor_fragment import fragment_address
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from .callbacks import DefrostCallback, EvaluationCallback
from .data import get_collator

torch.serialization.add_safe_globals(
    [
        np.core.multiarray._reconstruct,
        np.ndarray,
        np.dtype,
        np.dtypes.UInt32DType,
        deepspeed.runtime.fp16.loss_scaler.LossScaler,
        deepspeed.runtime.zero.config.ZeroStageEnum,
        fragment_address,
    ]
)


def train(config):
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_args.name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    vocab = tokenizer.get_vocab()
    vocab_size = config.speech2unit.vocab_size
    units = [f"<{unit}>" for unit in range(vocab_size)]
    for unit in units:
        assert unit not in vocab
    tokenizer.add_tokens(units)

    # Datasets
    train_dataset = load_dataset(config.dataset.name, "Libri-Light", split="train", keep_in_memory=True)
    eval_dataset = {
        "sWUGGY": load_dataset(config.dataset.name, "sWUGGY"),
        "sBLIMP": load_dataset(config.dataset.name, "sBLIMP"),
        "tSC": load_dataset(config.dataset.name, "tSC"),
    }

    # Model
    model = AutoModelForCausalLM.from_pretrained(config.model_args.name)
    model.resize_token_embeddings(len(tokenizer), mean_resizing=config.model_args.mean_resizing)

    callbacks = [EvaluationCallback(eval_dataset)]

    if (
        not config.training_args.resume_from_checkpoint
        or int(config.training_args.resume_from_checkpoint.rsplit("-", 1)[1]) < config.model_args.defrost_steps
    ):
        model.model.layers.requires_grad_(False)
        model.model.norm.requires_grad_(False)
        handle_input_embeddings = model.get_input_embeddings().weight.register_hook(
            lambda grad: torch.cat([torch.zeros_like(grad[: len(vocab)]), grad[len(vocab) :]])
        )
        handle_output_embeddings = model.get_output_embeddings().weight.register_hook(
            lambda grad: torch.cat([torch.zeros_like(grad[: len(vocab)]), grad[len(vocab) :]])
        )
        callbacks.append(
            DefrostCallback(handle_input_embeddings, handle_output_embeddings, config.model_args.defrost_steps)
        )

    training_args = TrainingArguments(**OmegaConf.to_container(config.training_args))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        data_collator=get_collator(tokenizer),
        callbacks=callbacks,
    )
    trainer.train(resume_from_checkpoint=config.training_args.resume_from_checkpoint)
