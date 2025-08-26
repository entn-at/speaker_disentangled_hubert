import numpy as np
import torch
from datasets import load_dataset
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3ForCausalLM, Trainer, TrainingArguments

from .callbacks import DefrostCallback, EvaluationCallback
from .configs import Qwen3ForSpeechLMConfig
from .data import get_collator
from .tokenizer import SpeechLMTokenizerFast

torch.serialization.add_safe_globals([np.core.multiarray._reconstruct, np.ndarray, np.dtype, np.dtypes.UInt32DType])


def train(config):
    # Tokenizer
    tokenizer = SpeechLMTokenizerFast(config.speech2unit.vocab_size)
    # tokenizer = AutoTokenizer.from_pretrained(config.model.name)
    # if tokenizer.pad_token is None:
    #     tokenizer.pad_token = tokenizer.eos_token
    # vocab = tokenizer.get_vocab()
    # vocab_size = config.speech2unit.vocab_size
    # units = [f"<{unit}>" for unit in range(vocab_size)]
    # for unit in units:
    #     assert unit not in vocab
    # tokenizer.add_tokens(units)

    # Datasets
    train_dataset = load_dataset(config.dataset.name, "Libri-Light", split="train", keep_in_memory=True)
    eval_dataset = {
        "sWUGGY": load_dataset(config.dataset.name, "sWUGGY"),
        "sBLIMP": load_dataset(config.dataset.name, "sBLIMP"),
        "tSC": load_dataset(config.dataset.name, "tSC"),
    }

    # Model
    model = Qwen3ForCausalLM(Qwen3ForSpeechLMConfig(**OmegaConf.to_container(config.model_args)))
    # model = AutoModelForCausalLM.from_pretrained(config.model.name)
    # model.resize_token_embeddings(len(tokenizer), mean_resizing=config.model.mean_resizing)
    # model.requires_grad_(False)
    # model.get_input_embeddings().requires_grad_(True)
    # model.get_output_embeddings().requires_grad_(True)
    # handle_input_embeddings = model.get_input_embeddings().weight.register_hook(
    #     lambda grad: torch.cat([torch.zeros_like(grad[: len(vocab)]), grad[len(vocab) :]])
    # )
    # handle_output_embeddings = model.get_output_embeddings().weight.register_hook(
    #     lambda grad: torch.cat([torch.zeros_like(grad[: len(vocab)]), grad[len(vocab) :]])
    # )

    training_args = TrainingArguments(**OmegaConf.to_container(config.training_args))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        data_collator=get_collator(tokenizer),
        callbacks=[
            EvaluationCallback(eval_dataset),
            # DefrostCallback(handle_input_embeddings, handle_output_embeddings, config.model.defrost_steps),
        ],
    )
    trainer.train(resume_from_checkpoint=config.training_args.resume_from_checkpoint)
