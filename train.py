import torch

from config.model_config import ModelConfig
from model.llm import LLM
from training.optimizer import create_optimizer
from training.trainer import Trainer
from training.checkpoint import save_checkpoint


def main():

    # --------------------------------
    # 1. Load configuration
    # --------------------------------

    config = ModelConfig()

    # --------------------------------
    # 2. Select device
    # --------------------------------

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Device:", device)

    # --------------------------------
    # 3. Create the LLM
    # --------------------------------

    model = LLM(
        vocab_size=config.vocab_size,
        context_length=config.context_length,
        embedding_dim=config.embedding_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        dropout=config.dropout
    )

    # --------------------------------
    # 4. Create optimizer
    # --------------------------------

    optimizer = create_optimizer(
        model,
        config.learning_rate
    )

    # --------------------------------
    # 5. Create trainer
    # --------------------------------

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=device
    )

    # --------------------------------
    # 6. Create temporary test data
    # --------------------------------

    input_ids = torch.randint(
        0,
        config.vocab_size,
        (config.batch_size, 32)
    )

    target_ids = torch.randint(
        0,
        config.vocab_size,
        (config.batch_size, 32)
    )

    # --------------------------------
    # 7. Perform one training step
    # --------------------------------

    loss = trainer.train_step(
        input_ids,
        target_ids
    )

    # --------------------------------
    # 8. Display result
    # --------------------------------

    print("Training step completed!")
    print("Loss:", loss)
    
    save_checkpoint(
    model=model,
    optimizer=optimizer,
    step=1,
    loss=loss,
    config=config,
    filepath="checkpoints/test_checkpoint.pt"
)


if __name__ == "__main__":
    main()