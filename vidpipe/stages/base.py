"""Stage interface and minimal Frame/FrameSpec dataclasses.

# رابط مرحله: هر مرحله بین صف reader و صف writer قرار می‌گیرد.
# FrameSpec ابعاد/نرخ فریم ورودی و خروجی را توصیف می‌کند (برای محاسبهٔ حافظه).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional


@dataclass
class FrameSpec:
    """Spec of a frame stream: dimensions, fps, bytes per frame.

    # مشخصات جریان فریم: ابعاد، نرخ فریم و اندازهٔ هر فریم بر حسب بایت.
    """
    width: int
    height: int
    fps: float
    frame_size: int  # بایت هر فریم


@dataclass
class Frame:
    """A single raw frame (bytes) plus its spec.

    # یک فریم خام به همراه مشخصاتش.
    """
    data: bytes
    spec: FrameSpec


class Stage:
    """Base stage interface. Subclasses implement the pipeline contract.

    # رابط پایهٔ مرحله. هر مرحله باید این متدها را پیاده کند.
    """

    def load(self, device: str, params: Any) -> None:
        """Load resources for this stage. Called once at startup.

        # بارگذاری منابع مرحله؛ یک بار در شروع اجرا.
        """
        raise NotImplementedError

    def plan(self, src_spec: FrameSpec) -> FrameSpec:
        """Return the output spec given the input spec.

        # مشخصات خروجی را با توجه به مشخصات ورودی برمی‌گرداند.
        """
        raise NotImplementedError

    def push(self, frame: Frame) -> List[Frame]:
        """Process one input frame, return zero or more output frames.

        # یک فریم ورودی را پردازش و صفر یا چند فریم خروجی برمی‌گرداند.
        """
        raise NotImplementedError

    def flush(self) -> List[Frame]:
        """Emit any buffered frames at end of stream.

        # در پایان جریان، فریم‌های بافر شده را خارج می‌کند.
        """
        return []

    def release(self) -> None:
        """Free resources. Called once at shutdown.

        # آزادسازی منابع؛ یک بار در پایان اجرا.
        """
        pass