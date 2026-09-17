import torch

from training.loss import language_model_loss


class Trainer:
    """
    Handles the training process for the LLM.
    """

    def __init__(self, model, optimizer, device="cpu"):
        self.model = model
        self.optimizer = optimizer
        self.device = device

        self.model.to(self.device)

    def train_step(self, input_ids, target_ids):
        """
        Perform one training step.
        """

        # Put model into training mode
        self.model.train()

        # Move data to the selected device
        input_ids = input_ids.to(self.device)
        target_ids = target_ids.to(self.device)

        # Clear old gradients
        self.optimizer.zero_grad()

        # Forward pass
        logits = self.model(input_ids)

        # Calculate loss
        loss = language_model_loss(
            logits,
            target_ids
        )

        # Backward pass
        loss.backward()

        # Update model weights
        self.optimizer.step()

        return loss.item()