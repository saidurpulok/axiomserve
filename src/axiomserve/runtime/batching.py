"""Physical padding and logical positions for dense Qwen2 batches."""

import torch


def padded_inputs(
    sequences: list[list[int]], pad_id: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    width = max(map(len, sequences))
    ids = torch.full((len(sequences), width), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for row, sequence in enumerate(sequences):
        ids[row, -len(sequence) :] = torch.tensor(sequence, device=device)
        mask[row, -len(sequence) :] = 1
    positions = (mask.cumsum(-1) - 1).clamp_min(0)
    return ids, mask, positions


def decode_inputs(
    tokens: list[int], lengths: list[int], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    width = max(lengths)
    ids = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(1)
    mask = torch.zeros((len(tokens), width + 1), dtype=torch.long, device=device)
    for row, length in enumerate(lengths):
        mask[row, -(length + 1) :] = 1
    positions = torch.tensor(lengths, dtype=torch.long, device=device).unsqueeze(1)
    return ids, mask, positions
