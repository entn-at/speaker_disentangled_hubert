from transformers import OPTConfig


class SpeechLMConfig(OPTConfig):
    def __init__(
        self,
        vocab_size: int = 16386,
        max_position_embeddings: int = 1024,
        dropout: float = 0.0,
        pad_token_id: int = 16385,
        bos_token_id: int = 16384,
        eos_token_id: int = 16385,
        **kwargs,
    ):
        super().__init__(
            vocab_size=vocab_size,
            max_position_embeddings=max_position_embeddings,
            dropout=dropout,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )
