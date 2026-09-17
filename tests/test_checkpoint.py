from config.model_config import ModelConfig
from model.llm import LLM
from training.optimizer import create_optimizer
from training.checkpoint import save_checkpoint, load_checkpoint


def test_checkpoint(tmp_path):

    config = ModelConfig()

    model = LLM(
        vocab_size=config.vocab_size,
        context_length=config.context_length,
        embedding_dim=config.embedding_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        dropout=config.dropout
    )

    optimizer = create_optimizer(
        model,
        config.learning_rate
    )

    filepath = tmp_path / "checkpoint.pt"

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        step=10,
        loss=5.0,
        config=config,
        filepath=str(filepath)
    )

    step, loss = load_checkpoint(
        model=model,
        optimizer=optimizer,
        filepath=str(filepath)
    )

    assert step == 10
    assert loss == 5.0