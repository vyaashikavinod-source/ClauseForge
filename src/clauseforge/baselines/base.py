"""Model-agnostic classification interface."""

from __future__ import annotations

from typing import Protocol, TypeAlias, runtime_checkable

import numpy as np
from numpy.typing import NDArray

FloatMatrix: TypeAlias = NDArray[np.float64]
StringArray: TypeAlias = NDArray[np.str_]


@runtime_checkable
class ClassifierProtocol(Protocol):
    """Interface shared by classical and future transformer classifiers."""

    name: str
    classes_: StringArray

    def fit(self, texts: list[str], labels: list[str]) -> None: ...

    def predict(self, texts: list[str]) -> StringArray: ...

    def predict_scores(self, texts: list[str]) -> FloatMatrix: ...

    def predict_proba(self, texts: list[str]) -> FloatMatrix | None: ...

    def configuration(self) -> dict[str, object]: ...
