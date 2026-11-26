import copy

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from clauseforge.training.suffix_loss import supervised_suffix_loss


def test_suffix_projection_matches_full_causal_loss_and_gradients() -> None:
    torch.manual_seed(42)
    config = Qwen2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        attention_dropout=0,
        tie_word_embeddings=False,
    )
    full = Qwen2ForCausalLM(config).eval()
    reduced = copy.deepcopy(full)
    ids = torch.randint(0, 64, (1, 20))
    labels = ids.clone()
    labels[:, :15] = -100
    batch = {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": labels}
    expected = full(**batch).loss
    actual = supervised_suffix_loss(reduced, batch)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    expected.backward()
    actual.backward()
    for (name, left), (_, right) in zip(
        full.named_parameters(), reduced.named_parameters(), strict=False
    ):
        if left.grad is not None:
            torch.testing.assert_close(
                right.grad, left.grad, rtol=2e-5, atol=1e-6, msg=name
            )
