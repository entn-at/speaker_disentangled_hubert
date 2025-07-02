import fire
from omegaconf import OmegaConf

from src.speechlm.data import tokenize_eval, tokenize_train
from src.speechlm.eval import evaluate
from src.speechlm.train import train


class TaskRunner:
    def tokenize_train(self, config: str = "configs/speechlm/default.yaml", spkids: str = "123456789"):
        config = OmegaConf.load(config)
        tokenize_train(config, spkids)

    def tokenize_eval(self, config: str = "configs/speechlm/default.yaml"):
        config = OmegaConf.load(config)
        tokenize_eval(config)

    def train(self, config: str = "configs/speechlm/default.yaml"):
        config = OmegaConf.load(config)
        train(config)

    def eval(self, config: str = "configs/speechlm/default.yaml"):
        config = OmegaConf.load(config)
        evaluate(config)


if __name__ == "__main__":
    fire.Fire(TaskRunner)
