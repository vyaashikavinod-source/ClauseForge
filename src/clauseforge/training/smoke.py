"""Local tokenizer fixture used only to prove the Phase 3A pipeline."""

from __future__ import annotations

from collections.abc import Iterable

from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast


def build_smoke_tokenizer(texts: Iterable[str]) -> PreTrainedTokenizerFast:
    tokenizer = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.WordLevelTrainer(  # type: ignore[no-untyped-call]
        vocab_size=2048,
        special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"],
    )
    tokenizer.train_from_iterator(texts, trainer)
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="[PAD]",
        unk_token="[UNK]",
        bos_token="[BOS]",
        eos_token="[EOS]",
    )
