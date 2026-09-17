import torch
import torch.nn as nn

from model.attention import CausalSelfAttention


class FeedForward(nn.Module):
    """
    Feed-forward network inside a Transformer block.
    """

    def __init__(self, embedding_dim, dropout=0.1):
        super().__init__()

        hidden_dim = 4 * embedding_dim

        self.network = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.network(x)


class TransformerBlock(nn.Module):
    """
    One Transformer block consisting of:

    LayerNorm
    Causal Self-Attention
    Residual Connection

    LayerNorm
    Feed-Forward Network
    Residual Connection
    """

    def __init__(
        self,
        embedding_dim,
        num_heads,
        context_length,
        dropout=0.1
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.attention = CausalSelfAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            context_length=context_length,
            dropout=dropout
        )

        self.feed_forward = FeedForward(
            embedding_dim=embedding_dim,
            dropout=dropout
        )

    def forward(self, x):

        # Attention + residual connection
        x = x + self.attention(self.norm1(x))

        # Feed-forward + residual connection
        x = x + self.feed_forward(self.norm2(x))

        return x