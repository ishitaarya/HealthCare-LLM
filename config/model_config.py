class ModelConfig:

    def __init__(self):

        # Vocabulary and sequence
        self.vocab_size = 10000
        self.context_length = 256

        # Transformer
        self.embedding_dim = 256
        self.num_layers = 6
        self.num_heads = 8

        # Regularization
        self.dropout = 0.1

        # Training
        self.batch_size = 16
        self.learning_rate = 3e-4