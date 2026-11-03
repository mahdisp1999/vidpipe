"""Map configured stage names to their implementation classes."""

from __future__ import annotations

from .stages.base import Stage
from .stages.identity import IdentityStage
from .stages.model import ModelStage

STAGES: dict[str, type[Stage]] = {
    "identity": IdentityStage,
    "model": ModelStage,
}


def resolve(name: str) -> type[Stage]:
    try:
        return STAGES[name]
    except KeyError as error:
        raise ValueError(f"unknown stage {name!r}; known stages: {sorted(STAGES)}") from error
