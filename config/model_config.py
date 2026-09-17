class ModelConfig:
    """Central configuration for model architecture and training defaults."""

    def __init__(
        self,
        vocab_size=10000,
        context_length=128,
        embedding_dim=256,
        num_layers=4,
        num_heads=8,
        dropout=0.1,
        batch_size=16,
        learning_rate=3e-4,
    ):
        # Vocabulary and sequence
        self.vocab_size = vocab_size
        self.context_length = context_length

        # Transformer
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.num_heads = num_heads

        # Regularization
        self.dropout = dropout

        # Training
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self._validate()

    def _validate(self):
        """Validate configuration values and model shape constraints."""
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.context_length <= 0:
            raise ValueError("context_length must be positive")
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if self.embedding_dim % self.num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1)")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

    def to_dict(self):
        """Return the configuration as a serializable dictionary."""
        return {
            "vocab_size": self.vocab_size,
            "context_length": self.context_length,
            "embedding_dim": self.embedding_dim,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "dropout": self.dropout,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
        }
