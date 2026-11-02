"""مرحلهٔ مدل: یک فایل .pth را با spandrel لود و روی هر فریم اجرا می‌کند.

تایلینگ: تصویر را تکه‌تکه (کاشی) به مدل می‌دهیم تا حافظه کم مصرف شود.
برای این‌که مرز کاشی‌ها درز نداشته باشد، کل تصویر یک‌بار با حاشیه (padding)
بزرگ می‌شود. هر کاشی همراه پیکسل‌های همسایه‌اش به مدل داده می‌شود و بعد
از اجرا، حاشیهٔ اضافه بریده می‌شود.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

import numpy as np

from .base import Frame, FrameSpec, Stage

log = logging.getLogger(__name__)

# torch و spandrel اختیاری‌اند؛ بدونشان فقط این مرحله قابل استفاده نیست.
try:
    import torch
    import torch.nn.functional as F
    from spandrel import ImageModelDescriptor, ModelLoader
    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
    F = None  # type: ignore
    ImageModelDescriptor = None  # type: ignore
    ModelLoader = None  # type: ignore
    _HAS_TORCH = False


class ModelStage(Stage):
    """یک مدل را روی هر فریم با تایلینگ اجرا می‌کند؛ resize خروجی اختیاری است."""

    def __init__(self) -> None:
        self.model: Optional[Any] = None
        self.device = "cpu"
        self.scale = 1
        self.tile_size = 512
        self.tile_pad = 16
        self.output_scale: Optional[float] = None
        self.half = False

    def load(self, device: str, params: Any) -> None:
        """مدل را یک‌بار بارگذاری می‌کند و روی دستگاه می‌برد."""
        if not _HAS_TORCH:
            raise RuntimeError("torch and spandrel are required to run model stages")
        self.device = device
        self.tile_size = int(params.get("tile_size", 512))
        self.tile_pad = int(params.get("tile_pad", 16))
        self.output_scale = params.get("output_scale")
        self.half = bool(params.get("half", False))

        model_path = params["model"]
        # spandrel معماری مدل را از روی وزن‌ها تشخیص می‌دهد.
        desc = ModelLoader().load_from_file(model_path)
        if not isinstance(desc, ImageModelDescriptor):
            raise RuntimeError(f"model {model_path!r} is not an image model")
        self.scale = int(desc.scale)
        desc.to(device).eval()
        # fp16 فقط روی CUDA و فقط اگر مدل پشتیبانی کند.
        if self.half and device == "cuda" and desc.supports_half:
            desc.half()
        self.model = desc
        log.info("model=%s arch=%s scale=%d device=%s loaded once",
                 model_path, desc.architecture, self.scale, device)

    def plan(self, src_spec: FrameSpec) -> FrameSpec:
        """اندازهٔ خروجی: ورودی × scale مدل، و اگر output_scale بود، ورودی × output_scale."""
        if self.output_scale is not None:
            w = int(round(src_spec.width * self.output_scale))
            h = int(round(src_spec.height * self.output_scale))
        else:
            w = src_spec.width * self.scale
            h = src_spec.height * self.scale
        return FrameSpec(width=w, height=h, fps=src_spec.fps, frame_size=w * h * 3)

    def push(self, frame: Frame) -> List[Frame]:
        """یک فریم rgb24 می‌گیرد و فریم پردازش‌شده برمی‌گرداند."""
        if self.model is None:
            raise RuntimeError("model not loaded")
        h, w = frame.spec.height, frame.spec.width

        # بایت‌های خام -> تنسور float در بازهٔ 0..1 با شکل (1, 3, H, W)
        img = np.frombuffer(frame.data, dtype=np.uint8).reshape(h, w, 3).copy()
        x = torch.from_numpy(img).permute(2, 0, 1).float().div(255.0).unsqueeze(0)
        x = x.to(self.device)
        if self.half and self.device == "cuda":
            x = x.half()

        with torch.no_grad():
            out = self._run_tiled(x)

        # اگر output_scale داده شده، خروجی به اندازهٔ برنامه‌ریزی‌شده می‌رود.
        spec = self.plan(frame.spec)
        if self.output_scale is not None:
            out = F.interpolate(out, size=(spec.height, spec.width),
                                mode="bicubic", align_corners=False)

        # float -> uint8 با گرد کردن (نه قطع کردن) تا رنگ‌ها تیره‌تر نشوند.
        out = out.float().squeeze(0).clamp(0.0, 1.0).mul(255.0).round().byte()
        data = out.permute(1, 2, 0).contiguous().cpu().numpy().tobytes()
        return [Frame(data=data, spec=spec)]

    def _run_tiled(self, x: "torch.Tensor") -> "torch.Tensor":
        """مدل را روی کاشی‌ها اجرا می‌کند و خروجی کامل را می‌سازد.

        # چرا کل تصویر را یک‌بار pad می‌کنیم؟
        # اگر هر کاشی جدا pad شود، لبهٔ کاشی‌ها بازتاب خودِ کاشی می‌شود و
        # روی مرز درز دیده می‌شود. با pad کردن کل تصویر، هر کاشی پیکسل‌های
        # واقعی همسایه را می‌بیند.
        """
        _, c, h, w = x.shape
        s = self.scale
        pad = self.tile_pad
        tile = self.tile_size

        # reflect فقط وقتی کار می‌کند که حاشیه از اندازهٔ تصویر کوچک‌تر باشد.
        mode = "reflect" if pad < min(h, w) else "replicate"
        x_pad = F.pad(x, (pad, pad, pad, pad), mode=mode)

        out = torch.zeros((1, c, h * s, w * s), dtype=x.dtype, device=x.device)
        for y in range(0, h, tile):
            y_end = min(y + tile, h)
            for x0 in range(0, w, tile):
                x_end = min(x0 + tile, w)
                # کاشی + حاشیهٔ همسایه از تصویر pad‌شده
                crop = x_pad[:, :, y:y_end + 2 * pad, x0:x_end + 2 * pad]
                res = self.model(crop)
                # فقط بخش اصلی کاشی (بدون حاشیه) را برمی‌داریم
                core = res[:, :, pad * s:pad * s + (y_end - y) * s,
                           pad * s:pad * s + (x_end - x0) * s]
                out[:, :, y * s:y_end * s, x0 * s:x_end * s] = core
        return out

    def flush(self) -> List[Frame]:
        """این مرحله بافر ندارد."""
        return []

    def release(self) -> None:
        """مدل را آزاد و حافظهٔ CUDA را خالی می‌کند."""
        self.model = None
        if _HAS_TORCH and torch.cuda.is_available():
            torch.cuda.empty_cache()