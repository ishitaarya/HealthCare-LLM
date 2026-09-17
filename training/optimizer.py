import torch


def create_optimizer(model, learning_rate=3e-4):
    """
    Create the optimizer used to train the LLM.
    """

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate
    )

    return optimizer