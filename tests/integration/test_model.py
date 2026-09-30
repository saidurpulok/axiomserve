import torch

from axiomserve.runtime.model_runner import ModelRunner


def test_batched_prefill_and_decode_match_unbatched_full_forward(tiny_runner: ModelRunner) -> None:
    sequences = [[2, 3], [4, 5, 6, 7, 8]]
    result = tiny_runner.prefill(sequences)
    for row, sequence in enumerate(sequences):
        single = tiny_runner.prefill([sequence])
        torch.testing.assert_close(result.logits[row], single.logits[0], atol=1e-6, rtol=1e-5)
    states = result.caches
    for tokens in [[9, 10], [11, 12], [13, 14]]:
        result = tiny_runner.decode(tokens, states)
        for row, token in enumerate(tokens):
            sequences[row].append(token)
            full = tiny_runner.prefill([sequences[row]])
            torch.testing.assert_close(result.logits[row], full.logits[0], atol=1e-6, rtol=1e-5)
        states = result.caches
    assert [state.length for state in states] == [5, 8]
    # Add a fresh request to an older cache cohort; this is the continuous packing path.
    fresh = tiny_runner.prefill([[15]]).caches[0]
    added = tiny_runner.decode([16, 17], [states[0], fresh])
    full = tiny_runner.prefill([sequences[0] + [16], [15, 17]])
    torch.testing.assert_close(added.logits, full.logits, atol=1e-6, rtol=1e-5)


def test_manual_greedy_matches_generate_oracle(tiny_runner: ModelRunner) -> None:
    prompt = [2, 3, 4]
    generated = []
    result = tiny_runner.prefill([prompt])
    for _ in range(6):
        token = int(result.logits[0].argmax())
        generated.append(token)
        result = tiny_runner.decode([token], result.caches)
    with torch.inference_mode():
        oracle = tiny_runner.model.generate(
            input_ids=torch.tensor([prompt]),
            attention_mask=torch.ones(1, 3, dtype=torch.long),
            max_new_tokens=6,
            do_sample=False,
            eos_token_id=None,
            pad_token_id=0,
        )
    assert generated == oracle[0, len(prompt) :].tolist()
