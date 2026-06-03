from tokenizers import Tokenizer, decoders
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from joey.config import SPECIAL_TOKENS, MASK_ID, PAD_ID, BOS_ID, EOS_ID


class JoeyTokenizer:
    def __init__(self, tk: Tokenizer):
        self._tk = tk
        self.mask_id = MASK_ID
        self.pad_id = PAD_ID
        self.bos_id = BOS_ID
        self.eos_id = EOS_ID

    @property
    def vocab_size(self) -> int:
        return self._tk.get_vocab_size()

    @classmethod
    def train(cls, files, vocab_size=16384):
        tk = Tokenizer(BPE(unk_token=None))
        tk.pre_tokenizer = ByteLevel(add_prefix_space=False)
        tk.decoder = decoders.ByteLevel()
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=SPECIAL_TOKENS,  # assigned ids 0..3 first
            show_progress=False,
        )
        tk.train(files, trainer)
        return cls(tk)

    def encode(self, text: str):
        return self._tk.encode(text).ids

    def decode(self, ids):
        return self._tk.decode(list(ids))

    def save(self, path: str):
        self._tk.save(path)

    @classmethod
    def load(cls, path: str):
        return cls(Tokenizer.from_file(path))
