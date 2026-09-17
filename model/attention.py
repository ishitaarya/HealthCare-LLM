import torch
import torch.nn as nn
import math


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention.

    The causal mask prevents a token from looking
    at future tokens during training.
    """

    def __init__(self, embedding_dim, num_heads, context_length, dropout=0.1):
        super().__init__()

        if embedding_dim % num_heads != 0:
            raise ValueError(
                "embedding_dim must be divisible by num_heads"
            )

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        # Create Query, Key, and Value projections
        self.q_proj = nn.Linear(embedding_dim, embedding_dim)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim)

        # Combine the attention heads
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)

        self.dropout = nn.Dropout(dropout)

        # Causal mask:
        # True = token is allowed to look there
        mask = torch.tril(
            torch.ones(context_length, context_length, dtype=torch.bool)
        )

        self.register_buffer("causal_mask", mask)

    def forward(self, x):
        """
        x shape:
            (batch_size, sequence_length, embedding_dim)
        """

        batch_size, sequence_length, _ = x.shape

        # Create Query, Key, Value
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Split into attention heads
        q = q.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        k = k.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        v = v.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        # Attention scores
        scores = q @ k.transpose(-2, -1)

        # Scale scores
        scores = scores / math.sqrt(self.head_dim)

        # Apply causal mask
        mask = self.causal_mask[:sequence_length, :sequence_length]

        scores = scores.masked_fill(
            ~mask,
            float("-inf")
        )

        # Convert scores into probabilities
        attention_weights = torch.softmax(scores, dim=-1)

        attention_weights = self.dropout(attention_weights)

        # Weighted combination of values
        output = attention_weights @ v

        # Combine heads
        output = output.transpose(1, 2).contiguous()

        output = output.view(
            batch_size,
            sequence_length,
            self.embedding_dim
        )

        # Final projection
        output = self.out_proj(output)

        return output