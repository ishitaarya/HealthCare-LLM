import torch
import torch.nn as nn

from model.embeddings import TokenEmbedding
from model.transformer import TransformerBlock


class LLM(nn.Module):
    """
    Basic decoder-only Transformer language model.
    """

    def __init__(
        self,
        vocab_size,
        context_length,
        embedding_dim,
        num_layers,
        num_heads,
        dropout=0.1
    ):
        super().__init__()

        self.context_length = context_length

        # Token embeddings
        self.token_embedding = TokenEmbedding(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim
        )

        # Learned positional embeddings
        self.position_embedding = nn.Embedding(
            context_length,
            embedding_dim
        )

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                context_length=context_length,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        # Final normalization
        self.final_norm = nn.LayerNorm(embedding_dim)

        # Language model output head
        self.lm_head = nn.Linear(
            embedding_dim,
            vocab_size,
            bias=False
        )

    def forward(self, token_ids):
        """
        token_ids shape:
            (batch_size, sequence_length)

        Returns:
            logits with shape:
            (batch_size, sequence_length, vocab_size)
        """

        batch_size, sequence_length = token_ids.shape

        if sequence_length > self.context_length:
            raise ValueError(
                f"Sequence length {sequence_length} exceeds "
                f"context length {self.context_length}"
            )

        # Token embeddings
        x = self.token_embedding(token_ids)

        # Position IDs
        position_ids = torch.arange(
            sequence_length,
            device=token_ids.device
        )

        # Add positional information
        x = x + self.position_embedding(position_ids)

        # Transformer blocks
        for block in self.transformer_blocks:
            x = block(x)

        # Final normalization
        x = self.final_norm(x)

        # Predict next-token logits
        logits = self.lm_head(x)

        return logits
    