import torch

from config.model_config import ModelConfig
from model.llm import LLM
from training.loss import language_model_loss
from training.optimizer import create_optimizer


def test_tiny_overfit():

    config = ModelConfig()

    # Create a smaller model for this test.
    vocab_size = 20
    context_length = 8
    embedding_dim = 64
    num_layers = 2
    num_heads = 4

    model = LLM(
        vocab_size=vocab_size,
        context_length=context_length,
        embedding_dim=embedding_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=0.0
    )

    optimizer = create_optimizer(
        model,
        learning_rate=1e-3
    )

    # Repeating sequence:
    #
    # 0 → 1
    # 1 → 2
    # 2 → 3
    # ...
    #
    # The model should learn this simple pattern.

    input_ids = torch.tensor([
        [0, 1, 2, 3, 4, 5, 6, 7],
        [1, 2, 3, 4, 5, 6, 7, 8],
        [2, 3, 4, 5, 6, 7, 8, 9],
        [3, 4, 5, 6, 7, 8, 9, 10],
    ])

    target_ids = torch.tensor([
        [1, 2, 3, 4, 5, 6, 7, 8],
        [2, 3, 4, 5, 6, 7, 8, 9],
        [3, 4, 5, 6, 7, 8, 9, 10],
        [4, 5, 6, 7, 8, 9, 10, 11],
    ])

    model.train()

    initial_loss = None
    final_loss = None

    for step in range(300):

        optimizer.zero_grad()

        logits = model(input_ids)

        loss = language_model_loss(
            logits,
            target_ids
        )

        if step == 0:
            initial_loss = loss.item()

        loss.backward()

        optimizer.step()

        final_loss = loss.item()

        if step % 50 == 0:
            print(
                f"Step {step:03d} | "
                f"Loss: {final_loss:.4f}"
            )

    print(f"Initial loss: {initial_loss:.4f}")
    print(f"Final loss:   {final_loss:.4f}")

    # The model should learn the simple pattern.
    assert final_loss < initial_loss