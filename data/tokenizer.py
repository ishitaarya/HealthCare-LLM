class Tokenizer:
    """
    Basic tokenizer interface for our LLM.

    This is a foundation implementation.
    We will replace/enhance it later with a proper
    subword tokenizer trained on our actual dataset.
    """

    def __init__(self):
        self.vocab = {}
        self.inverse_vocab = {}

    def encode(self, text):
        """Convert text into token IDs."""
        raise NotImplementedError("Tokenizer encoding is not implemented yet.")

    def decode(self, token_ids):
        """Convert token IDs back into text."""
        raise NotImplementedError("Tokenizer decoding is not implemented yet.")