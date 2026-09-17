import json
from collections import Counter


class BPETokenizer:
    """Small, dependency-free BPE tokenizer for the project."""

    SPECIAL_TOKENS = ("<PAD>", "<UNK>", "<BOS>", "<EOS>", "<SEP>")
    WORD_PREFIX = "▁"

    def __init__(self, vocab_size=10000):
        if vocab_size <= len(self.SPECIAL_TOKENS):
            raise ValueError("vocab_size must be larger than the special-token count")
        self.vocab_size = vocab_size
        self.vocab = {}
        self.inverse_vocab = {}
        self.merges = []

    @property
    def pad_id(self):
        return self.vocab["<PAD>"]

    @property
    def unk_id(self):
        return self.vocab["<UNK>"]

    @property
    def bos_id(self):
        return self.vocab["<BOS>"]

    @property
    def eos_id(self):
        return self.vocab["<EOS>"]

    @property
    def sep_id(self):
        return self.vocab["<SEP>"]

    @staticmethod
    def _words(text):
        return text.strip().split()

    def _initial_sequences(self, texts):
        sequences = []
        for text in texts:
            for word in self._words(text):
                sequences.append(tuple(self.WORD_PREFIX + ch if i == 0 else ch
                                       for i, ch in enumerate(word)))
        return sequences

    @staticmethod
    def _merge_pair(sequence, pair, merged):
        output = []
        i = 0
        while i < len(sequence):
            if i + 1 < len(sequence) and (sequence[i], sequence[i + 1]) == pair:
                output.append(merged)
                i += 2
            else:
                output.append(sequence[i])
                i += 1
        return tuple(output)

    def train(self, texts):
        """Train BPE merges from an iterable of corpus documents."""
        texts = list(texts)
        if not texts:
            raise ValueError("training corpus must not be empty")

        sequences = self._initial_sequences(texts)
        symbols = set(symbol for seq in sequences for symbol in seq)
        merges = []

        while len(symbols) + len(self.SPECIAL_TOKENS) < self.vocab_size:
            pair_counts = Counter(
                pair for seq in sequences for pair in zip(seq, seq[1:])
            )
            if not pair_counts:
                break

            pair, count = pair_counts.most_common(1)[0]
            if count < 2:
                break

            merged = pair[0] + pair[1]
            merges.append(pair)
            symbols.add(merged)
            sequences = [self._merge_pair(seq, pair, merged) for seq in sequences]

        tokens = list(self.SPECIAL_TOKENS) + sorted(symbols)
        self.vocab = {token: i for i, token in enumerate(tokens[:self.vocab_size])}
        self.inverse_vocab = {i: token for token, i in self.vocab.items()}
        self.merges = [pair for pair in merges if pair[0] + pair[1] in self.vocab]
        return self

    def _encode_word(self, word):
        sequence = tuple(self.WORD_PREFIX + ch if i == 0 else ch
                         for i, ch in enumerate(word))
        for pair in self.merges:
            sequence = self._merge_pair(sequence, pair, pair[0] + pair[1])
        return [self.vocab.get(symbol, self.unk_id) for symbol in sequence]

    def encode(self, text, add_bos=False, add_eos=False):
        if not self.vocab:
            raise RuntimeError("tokenizer must be trained or loaded before encoding")

        token_ids = [self.bos_id] if add_bos else []
        for word in self._words(text):
            token_ids.extend(self._encode_word(word))
        if add_eos:
            token_ids.append(self.eos_id)
        return token_ids

    def decode(self, token_ids, skip_special_tokens=True):
        pieces = []
        for token_id in token_ids:
            token = self.inverse_vocab.get(int(token_id), "<UNK>")
            if skip_special_tokens and token in self.SPECIAL_TOKENS:
                continue
            pieces.append(token)
        return "".join(pieces).replace(self.WORD_PREFIX, " ").strip()

    def save(self, path):
        payload = {
            "vocab_size": self.vocab_size,
            "vocab": self.vocab,
            "merges": [list(pair) for pair in self.merges],
        }
        with open(path, "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as file:
            payload = json.load(file)
        tokenizer = cls(payload["vocab_size"])
        tokenizer.vocab = {str(k): int(v) for k, v in payload["vocab"].items()}
        tokenizer.inverse_vocab = {v: k for k, v in tokenizer.vocab.items()}
        tokenizer.merges = [tuple(pair) for pair in payload["merges"]]
        return tokenizer


Tokenizer = BPETokenizer
