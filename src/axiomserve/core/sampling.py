"""Sampling uses per-request CPU RNGs, independent of batch composition."""

import torch
from pydantic import BaseModel, ConfigDict, Field


class SamplingParams(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    max_tokens: int = Field(default=64, ge=1)
    temperature: float = Field(default=0, ge=0)
    top_k: int = Field(default=0, ge=0)
    top_p: float = Field(default=1, gt=0, le=1)
    seed: int = Field(default=0, ge=0, le=2**63 - 1)
    ignore_eos: bool = False


def sample(logits: torch.Tensor, params: SamplingParams, rng: torch.Generator) -> int:
    """Apply top-k then nucleus filtering, retaining the threshold-crossing token."""
    scores = logits.detach().float().cpu()
    if not torch.isfinite(scores).all():
        raise ValueError("model returned non-finite logits")
    if params.temperature == 0:
        return int(scores.argmax().item())
    scores = scores / params.temperature
    if 0 < params.top_k < scores.numel():
        threshold = torch.topk(scores, params.top_k).values[-1]
        scores = scores.masked_fill(scores < threshold, -torch.inf)
    if params.top_p < 1:
        sorted_scores, order = scores.sort(descending=True)
        remove = sorted_scores.softmax(-1).cumsum(-1) > params.top_p
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        scores[order[remove]] = -torch.inf
    return int(torch.multinomial(scores.softmax(-1), 1, generator=rng).item())
