"""Exact causal loss for single-example batches with a masked prompt prefix.

Qwen's vocabulary projection is needed only at positions preceding supervised
tokens. Avoid allocating full-sequence vocabulary logits for ignored labels.
"""
from typing import Any
import torch
import torch.nn.functional as F


def supervised_suffix_loss(model: Any, batch: dict[str, Any]):
    labels = batch['labels']
    if labels.ndim != 2 or labels.shape[0] != 1 or getattr(getattr(model, 'config', None), 'model_type', '') != 'qwen2':
        return model(**batch).loss
    count = int((labels[0] != -100).sum().item())
    if count == 0 or count >= labels.shape[1] or not bool((labels[0, -count:] != -100).all()):
        return model(**batch).loss
    # The preceding position predicts the first supervised token; the final
    # input position has no next-token target and is excluded from the loss.
    output = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                   logits_to_keep=count + 1, use_cache=False)
    logits = output.logits[:, :-1, :].float()
    if logits.shape[1] != count:
        raise ValueError('Model did not honor suffix logits selection')
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels[:, -count:].reshape(-1))
