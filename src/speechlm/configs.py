from transformers import OPTConfig, Qwen3Config


class OPTForSpeechLMConfig(OPTConfig):
    def __init__(
        self,
        vocab_size: int = 16386,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        ffn_dim: int = 3072,
        max_position_embeddings: int = 256,
        dropout: float = 0.0,
        num_attention_heads: int = 12,
        pad_token_id: int = 16385,
        bos_token_id: int = 16384,
        eos_token_id: int = 16385,
        **kwargs,
    ):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            ffn_dim=ffn_dim,
            max_position_embeddings=max_position_embeddings,
            dropout=dropout,
            num_attention_heads=num_attention_heads,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )


class Qwen3ForSpeechLMConfig(Qwen3Config):
    def __init__(
        self,
        vocab_size: int = 16386,
        hidden_size: int = 768,
        intermediate_size: int = 2048,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        num_key_value_heads: int = 12,
        head_dim: int = 64,
        max_position_embeddings: int = 256,
        tie_word_embeddings: bool = True,
        rope_theta: float = 1000.0,
        pad_token_id: int = 16385,
        bos_token_id: int = 16384,
        eos_token_id: int = 16385,
        **kwargs,
    ):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            max_position_embeddings=max_position_embeddings,
            tie_word_embeddings=tie_word_embeddings,
            rope_theta=rope_theta,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )
