"""Opt-in pretrained CUDA check; excluded from normal CPU CI."""

import os

import pytest
import torch
from transformers import GenerationConfig

from axiomserve.config import RuntimeConfig
from axiomserve.runtime.model_runner import ModelRunner


@pytest.mark.gpu
@pytest.mark.pretrained
@pytest.mark.skipif(os.getenv("AXIOM_TEST_GPU") != "1", reason="set AXIOM_TEST_GPU=1 to opt in")
def test_pretrained_cuda_greedy_oracle() -> None:
    assert torch.cuda.is_available(), "CUDA opt-in was set but GPU is unavailable"
    runner = ModelRunner.load(RuntimeConfig(device="cuda", local_files_only=True))
    prompt = runner.encode("The capital of France is")
    result = runner.prefill([prompt])
    manual = []
    for _ in range(8):
        token = int(result.logits[0].argmax())
        manual.append(token)
        if token in runner.eos_ids:
            break
        result = runner.decode([token], result.caches)
    with torch.inference_mode():
        ids = torch.tensor([prompt], device=runner.device)
        oracle = runner.model.generate(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            generation_config=GenerationConfig(
                max_new_tokens=8,
                do_sample=False,
                repetition_penalty=1.0,
                eos_token_id=sorted(runner.eos_ids),
                pad_token_id=runner.pad_id,
            ),
        )
    assert manual == oracle[0, len(prompt) :].tolist()
