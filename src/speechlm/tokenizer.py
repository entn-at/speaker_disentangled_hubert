from tokenizers import Regex, Tokenizer, models, processors
from tokenizers.pre_tokenizers import Split
from transformers import PreTrainedTokenizerFast


class SpeechLMTokenizerFast(PreTrainedTokenizerFast):
    def __init__(
        self,
        vocab_size: int = 16384,
        bos_token: str = "<|begin_of_text|>",
        eos_token: str = "<|end_of_text|>",
        unk_token: str = "<|unk|>",
    ):
        vocab = [f"<{unit}>" for unit in range(vocab_size)] + [bos_token, eos_token, unk_token]
        vocab = {token: token_id for token_id, token in enumerate(vocab)}

        bos_token_id = vocab[bos_token]
        single = f"{bos_token} $0"
        special_tokens = [(bos_token, bos_token_id)]

        tokenizer_object = Tokenizer(models.WordLevel(vocab, unk_token=unk_token))
        tokenizer_object.pre_tokenizer = Split(pattern=Regex(r"<\d+>"), behavior="isolated")
        tokenizer_object.post_processor = processors.TemplateProcessing(single=single, special_tokens=special_tokens)

        super().__init__(
            tokenizer_object=tokenizer_object,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=eos_token,
        )
