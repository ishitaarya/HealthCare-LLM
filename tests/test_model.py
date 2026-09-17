import torch

from config.model_config import ModelConfig
from model.llm import LLM


def test_model_output_shape():

    config = ModelConfig()

    model = LLM(
        vocab_size=config.vocab_size,
        context_length=config.context_length,
        embedding_dim=config.embedding_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        dropout=config.dropout
    )

    input_ids = torch.randint(
        0,
        config.vocab_size,
        (2, 16)
    )

    logits = model(input_ids)

    assert logits.shape == (
        2,
        16,
        config.vocab_size
    )