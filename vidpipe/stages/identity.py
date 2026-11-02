"""Identity stage: pass-through, used for testing the pipeline.

# مرحلهٔ هویت: فریم را بدون تغییر عبور می‌دهد؛ برای تست لولهٔ پردازش.
"""

from __future__ import annotations

from typing import Any, List

from .base import Frame, FrameSpec, Stage


class IdentityStage(Stage):
    def load(self, device: str, params: Any) -> None:
        # هویت هیچ منبعی بارگذاری نمی‌کند.
        pass

    def plan(self, src_spec: FrameSpec) -> FrameSpec:
        # خروجی همان ورودی است.
        return src_spec

    def push(self, frame: Frame) -> List[Frame]:
        # عبور مستقیم فریم.
        return [frame]

    def flush(self) -> List[Frame]:
        # هیچ فریم بافری وجود ندارد.
        return []