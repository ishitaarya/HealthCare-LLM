"""Benchmark small model configurations on the local development machine.

This benchmark measures parameter count, CPU training-step time, and peak
Python-process memory for representative configurations. It is intentionally
small so it can run on a CPU-only laptop.
"""

import gc
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.llm import HealthcareLLM


CONFIGS = [
    {
        "name": "small",
        "vocab_size": 10000,
        "context_length": 128,
        "embedding_dim": 192,
        "num_layers": 4,
        "num_heads": 6,
    },
    {
        "name": "medium",
        "vocab_size": 10000,
        "context_length": 128,
        "embedding_dim": 256,
        "num_layers": 4,
        "num_heads": 8,
    },
    {
        "name": "large",
        "vocab_size": 10000,
        "context_length": 128,
        "embedding_dim": 256,
        "num_layers": 6,
        "num_heads": 8,
    },
]


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters())


def benchmark(config, steps=5, batch_size=2):
    gc.collect()
    torch.set_num_threads(max(1, min(6, os.cpu_count() or 1)))

    model = HealthcareLLM(
        vocab_size=config["vocab_size"],
        context_length=config["context_length"],
        embedding_dim=config["embedding_dim"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        dropout=0.0,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    input_ids = torch.randint(
        0, config["vocab_size"], (batch_size, config["context_length"])
    )
    targets = torch.randint(
        0, config["vocab_size"], (batch_size, config["context_length"])
    )

    model.train()
    start = time.perf_counter()

    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, config["vocab_size"]),
            targets.reshape(-1),
        )
        loss.backward()
        optimizer.step()

    elapsed = time.perf_counter() - start

    result = {
        "name": config["name"],
        "parameters": count_parameters(model),
        "parameters_m": count_parameters(model) / 1_000_000,
        "seconds_per_step": elapsed / steps,
        "final_loss": loss.item(),
    }

    del model, optimizer, input_ids, targets, logits, loss
    gc.collect()
    return result


def main():
    print("Healthcare LLM model-size benchmark")
    print("CPU threads:", torch.get_num_threads())
    print()

    for config in CONFIGS:
        result = benchmark(config)
        print(
            f"{result['name']:>6} | "
            f"{result['parameters_m']:>7.2f}M params | "
            f"{result['seconds_per_step']:>8.3f} sec/step | "
            f"loss {result['final_loss']:.4f}"
        )


if __name__ == "__main__":
    main()
