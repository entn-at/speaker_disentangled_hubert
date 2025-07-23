import os
from pathlib import Path

import fire
from omegaconf import OmegaConf


class TaskRunner:
    def tokenize_train(self, config: str = "configs/speechlm/default.yaml", num_shards: int = 1, shard_index: int = 0):
        config = OmegaConf.load(config)
        os.environ["HF_HOME"] = str(Path(config.dataset.HF_HOME).expanduser())

        from src.speechlm.data import tokenize_train

        tokenize_train(config, num_shards, shard_index)

    def tokenize_eval(self, config: str = "configs/speechlm/default.yaml"):
        from src.speechlm.data import tokenize_eval

        config = OmegaConf.load(config)
        tokenize_eval(config)

    def train(self, config: str = "configs/speechlm/default.yaml"):
        config = OmegaConf.load(config)
        os.environ["HF_HOME"] = str(Path(config.dataset.HF_HOME).expanduser())

        from src.speechlm.train import train

        train(config)


if __name__ == "__main__":
    fire.Fire(TaskRunner)
