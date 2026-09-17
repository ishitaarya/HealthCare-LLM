import torch
import torch.nn.functional as F


def language_model_loss(logits, targets):
    """
    Calculate next-token prediction loss.

    logits:
        (batch_size, sequence_length, vocab_size)

    targets:
        (batch_size, sequence_length)
    """

    batch_size, sequence_length, vocab_size = logits.shape

    # Flatten predictions and targets
    logits = logits.reshape(
        batch_size * sequence_length,
        vocab_size
    )

    targets = targets.reshape(
        batch_size * sequence_length
    )

    # Cross-entropy loss
    loss = F.cross_entropy(
        logits,
        targets
    )

    return loss