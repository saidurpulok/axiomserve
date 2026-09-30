"""Dense, unpadded per-request KV storage; the only Transformers cache adapter."""

from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from transformers.cache_utils import DynamicCache, DynamicLayer


@dataclass
class KVState:
    # Each tensor is [1, kv_heads, sequence, head_dim], including rotated keys.
    layers: tuple[tuple[torch.Tensor, torch.Tensor], ...]

    @property
    def length(self) -> int:
        return self.layers[0][0].shape[-2]

    @property
    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for pair in self.layers for t in pair)

    def describe(self) -> dict[str, object]:
        return {
            "length": self.length,
            "bytes": self.nbytes,
            "shapes": [list(key.shape) for key, _ in self.layers],
        }


def pack(states: list[KVState]) -> DynamicCache:
    """Left-pad real caches without changing their logical RoPE positions."""
    width = max(state.length for state in states)
    cache = DynamicCache()
    for layer_index in range(len(states[0].layers)):
        keys, values = [], []
        for state in states:
            key, value = state.layers[layer_index]
            padding = (0, 0, width - state.length, 0)
            keys.append(functional.pad(key, padding))
            values.append(functional.pad(value, padding))
        cache.update(torch.cat(keys), torch.cat(values), layer_index)
    return cache


def unpack(cache: DynamicCache, lengths: list[int]) -> list[KVState]:
    """Clone slices so a short request does not retain the whole batch allocation."""
    layers: list[tuple[torch.Tensor, torch.Tensor]] = []
    for layer in cache.layers:
        if not isinstance(layer, DynamicLayer) or layer.keys is None or layer.values is None:
            raise TypeError("expected initialized full-attention DynamicCache layers")
        layers.append((layer.keys, layer.values))
    return [
        KVState(
            tuple(
                (
                    key[row : row + 1, :, -length:, :].clone(),
                    value[row : row + 1, :, -length:, :].clone(),
                )
                for key, value in layers
            )
        )
        for row, length in enumerate(lengths)
    ]
