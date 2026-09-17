import os
import torch


def save_checkpoint(
    model,
    optimizer,
    step,
    loss,
    config,
    filepath
):
    """
    Save the current training state.
    """

    checkpoint = {
        "step": step,
        "loss": loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": {
            key: value
            for key, value in vars(config).items()
        }
    }

    # Create directory if it doesn't exist
    directory = os.path.dirname(filepath)

    if directory:
        os.makedirs(directory, exist_ok=True)

    torch.save(checkpoint, filepath)

    print(f"Checkpoint saved: {filepath}")


def load_checkpoint(
    model,
    optimizer,
    filepath,
    device="cpu"
):
    """
    Load a previously saved training state.
    """

    checkpoint = torch.load(
        filepath,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    step = checkpoint["step"]
    loss = checkpoint["loss"]

    print(f"Checkpoint loaded: {filepath}")

    return step, loss