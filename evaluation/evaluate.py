import torch

from training.loss import language_model_loss


@torch.no_grad()
def evaluate(model, data_loader, device="cpu"):
    """
    Evaluate the model on a dataset.

    Returns the average loss.
    """

    model.eval()

    total_loss = 0.0
    total_batches = 0

    for input_ids, target_ids in data_loader:

        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        logits = model(input_ids)

        loss = language_model_loss(
            logits,
            target_ids
        )

        total_loss += loss.item()
        total_batches += 1

    if total_batches == 0:
        return float("inf")

    return total_loss / total_batches