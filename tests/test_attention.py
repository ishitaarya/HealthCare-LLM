import torch

from config.model_config import ModelConfig
from model.attention import CausalSelfAttention


def test_attention_output_shape():

    config = ModelConfig()

    attention = CausalSelfAttention(
        embedding_dim=config.embedding_dim,
        num_heads=config.num_heads,
        context_length=config.context_length,
        dropout=config.dropout
    )

    x = torch.randn(
        2,
        16,
        config.embedding_dim
    )

    output = attention(x)

    assert output.shape == x.shape