import os

import torch
import numpy as np
import numpy.typing as npt
from typing import BinaryIO, IO

def get_batch(
        dataset: npt.NDArray,
        batch_size: int,
        context_length: int,
        device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if context_length <= 0:
        raise ValueError(f"context_length must be positive, got {context_length}")
    if len(dataset) <= context_length:
        raise ValueError(f"dataset has {len(dataset)} tokens, needs more than context_length ({context_length})")
    
    max_sample_start_indices = len(dataset) - context_length - 1
    batch_sample_start_indices = np.random.randint(low=0, high=max_sample_start_indices+1, size=batch_size, dtype=np.int64)

    offsets = np.arange(0, context_length)
    input_batch_start_indices = batch_sample_start_indices[:, None] + offsets[None, :]

    inputs = torch.tensor(dataset[input_batch_start_indices], device=device, dtype=torch.long)
    targets = torch.tensor(dataset[input_batch_start_indices + 1], device=device, dtype=torch.long)

    return inputs, targets


def save_checkpoint(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        iteration: int,
        out: str | os.PathLike | BinaryIO | IO[bytes]
):
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'iteration': iteration
    }
    torch.save(checkpoint, out)


def load_checkpoint(
        src: str | os.PathLike | BinaryIO | IO[bytes],
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None
):
    device = next(model.parameters()).device
    checkpoint = torch.load(src, map_location=device)

    model.load_state_dict(checkpoint['model'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])

    return checkpoint['iteration']