import torch


@torch.no_grad()
def generate(
    model,
    input_ids,
    max_new_tokens,
    context_length
):
    """
    Generate new tokens autoregressively.

    input_ids:
        Shape: (batch_size, sequence_length)
    """

    model.eval()

    for _ in range(max_new_tokens):

        # Keep only the most recent context if necessary
        input_context = input_ids[:, -context_length:]

        # Get model predictions
        logits = model(input_context)

        # Get predictions for the final token
        next_token_logits = logits[:, -1, :]

        # Select the most likely next token
        next_token = torch.argmax(
            next_token_logits,
            dim=-1,
            keepdim=True
        )

        # Add the new token
        input_ids = torch.cat(
            (input_ids, next_token),
            dim=1
        )

    return input_ids