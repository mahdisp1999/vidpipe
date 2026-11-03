"""Pass-through stage used for testing the reader/writer pipeline."""

from __future__ import annotations

from typing import Any

from .base import Frame, FrameSpec, Stage


class IdentityStage(Stage):
    output_label = "identity"
    is_denoise = False

    def load(self, device: str, params: Any) -> None:
        pass

    def plan(self, source_spec: FrameSpec) -> FrameSpec:
        return source_spec

    def push(self, frame: Frame) -> list[Frame]:
        return [frame]
