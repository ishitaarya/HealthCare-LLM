# Hardware Specification

## Development Machine

- Manufacturer: HP
- Model: HP Pavilion Laptop 15-eh1xxx
- RAM: 16 GB installed (16,455,938,048 bytes reported by Windows)
- CPU: AMD Ryzen 5 5500U with Radeon Graphics
- CPU cores: 6
- Logical processors: 12
- GPU: AMD Radeon(TM) Graphics
- Reported GPU memory: 536,870,912 bytes (512 MiB)
- Operating system: Windows 11 Home Single Language
- Architecture: 64-bit

## Training Constraints

This machine should be treated as a CPU-first development and training system unless a supported GPU training path is separately validated.

The reported integrated Radeon graphics memory is 512 MiB, so the project will not assume that a large model can be trained in GPU memory.

The final model configuration must prioritize:

- RAM usage that leaves headroom for Windows and development tools
- CPU training feasibility
- Small-batch training
- Gradient accumulation when needed
- Checkpointing
- Short initial experiments before longer training runs

## Configuration Decision

The final parameter count, embedding dimension, number of layers, context length, and batch size are intentionally not fixed here. They will be selected in Phase 2 Step 3 after considering this hardware profile and measuring an actual forward/backward training step.

## Status

Phase 2 — Step 2: Hardware inspection complete.
