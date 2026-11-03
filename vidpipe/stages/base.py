"""Small interface shared by the video-processing stages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class FrameSpec:
    width: int
    height: int
    fps: float
    frame_size: int


@dataclass
class Frame:
    data: bytes
    spec: FrameSpec


class Stage:
    def load(self, device: str, params: Any) -> None:
        raise NotImplementedError

    def plan(self, source_spec: FrameSpec) -> FrameSpec:
        raise NotImplementedError

    def push(self, frame: Frame) -> list[Frame]:
        raise NotImplementedError

    def flush(self) -> list[Frame]:
        return []

    def release(self) -> None:
        pass
