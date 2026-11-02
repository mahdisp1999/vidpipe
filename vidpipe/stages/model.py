"""Generic model stage: runs one .pth model (restore or upscale) with tiling.

# مرحلهٔ مدل: یک فایل .pth را با spandrel بارگذاری و با تایلینگ اجرا می‌کند.
# هم برای restore (1x) و هم برای upscale (Nx) استفاده می‌شود؛ تنها جایی که مدل اجرا می‌شود.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

import numpy as np

from .base import Frame, FrameSpec, Stage

log = logging.getLogger(__name__)

# torch اختیاری است؛ اگر نصب نباشد، مدل قابل اجرا نیست.
try:
    import torch  # type: ignore
    from spandrel import ImageModelDescriptor, ModelLoader  # type: ignore
    _HAS_TORCH = True
except ImportError:  # pragma: no cover - torch یا spandrel نصب نیست
    torch = None  # type: ignore
    ImageModelDescriptor = None  # type: ignore
    ModelLoader = None  # type: ignore
    _HAS_TORCH = False


class ModelStage(Stage):
    """Runs one model on each frame, tiled, with optional output resize.

    # یک مدل را روی هر فریم با تایلینگ اجرا می‌کند؛ resize خروجی اختیاری است.
    """

    def __init__(self) -> None:
        self.model: Optional[ImageModelDescriptor] = None
        self.device = "cpu"
        self.scale = 1
        self.tile_size = 512
        self.tile_pad = 16
        self.output_scale: Optional[float] = None
        self.half = False

    def load(self, device: str, params: Any) -> None:
        # بارگذاری مدل با spandrel؛ یک بار در هر اجرا.
        if not _HAS_TORCH:
            raise RuntimeError("torch and spandrel are required to run model stages")
        self.device = device
        self.tile_size = int(params.get("tile_size", 512))
        self.tile_pad = int(params.get("tile_pad", 16))
        self.output_scale = params.get("output_scale")
        self.half = bool(params.get("half", False))
        model_path = params["model"]
        # spandrel معماری را از وزن‌ها تشخیص می‌دهد.
        desc = ModelLoader().load_from_file(model_path)
        if not isinstance(desc, ImageModelDescriptor):
            raise RuntimeError(f"model {model_path!r} is not an image model")
        self.model = desc
        self.scale = int(desc.scale)
        # انتقال به دستگاه و حالت inference.
        self.model.to(device).eval()
        # fp16 فقط روی CUDA و اگر مدل پشتیبانی کند.
        if self.half and device == "cuda" and desc.supports_half:
            self.model.half()
        log.info(
            "model=%s arch=%s scale=%d device=%s loaded once",
            model_path, desc.architecture, self.scale, device,
        )

    def plan(self, src_spec: FrameSpec) -> FrameSpec:
        # اندازهٔ خروجی = ورودی × scale مدل، سپس × output_scale اگر داده شده باشد.
        # اگر output_scale داده شود، اندازهٔ نهایی = ورودی × output_scale.
        if self.output_scale is not None:
            w = int(round(src_spec.width * self.output_scale))
            h = int(round(src_spec.height * self.output_scale))
        else:
            w = src_spec.width * self.scale
            h = src_spec.height * self.scale
        return FrameSpec(width=w, height=h, fps=src_spec.fps, frame_size=w * h * 3)

    def push(self, frame: Frame) -> List[Frame]:
        # uint8 HWC RGB -> float NCHW در 0..1 روی دستگاه؛ اجرای مدل با تایلینگ؛
        # تبدیل به uint8 HWC؛ و در صورت output_scale، resize به اندازهٔ برنامه‌ریزی‌شده.
        if self.model is None:
            raise RuntimeError("model not loaded")
        img = np.frombuffer(frame.data, dtype=np.uint8).reshape(
            frame.spec.height, frame.spec.width, 3)
        # HWC uint8 -> NCHW float در 0..1 روی دستگاه.
        t = torch.from_numpy(img).permute(2, 0, 1).float().div_(255.0)
        t = t.unsqueeze(0).to(self.device)
        if self.half and self.device == "cuda":
            t = t.half()
        out = self._run_tiled(t)
        # resize خروجی به اندازهٔ برنامه‌ریزی‌شده (bicubic روی دستگاه، بدون وابستگی جدید).
        if self.output_scale is not None:
            out = torch.nn.functional.interpolate(
                out, size=(frame.spec.height * self.scale, frame.spec.width * self.scale),
                mode="bicubic", align_corners=False)
        # NCHW float -> HWC uint8.
        out = out.squeeze(0).clamp_(0.0, 1.0).mul_(255.0).byte()
        out = out.permute(1, 2, 0).cpu().numpy()
        spec = self.plan(frame.spec)
        return [Frame(data=out.tobytes(), spec=spec)]

    def _run_tiled(self, t: "torch.Tensor") -> "torch.Tensor":
        """Run the model on tiles with reflect padding; seams line up exactly.

        # اجرای مدل روی تایل‌ها با padding بازتابی؛ درزها دقیقاً هم‌راستا می‌شوند.
        # چرا padding: مدل‌های SR به پیکسل‌های اطراف نیاز دارند؛ بدون padding لبهٔ هر
        # تایل مصنوعی می‌شود. چرا برش: خروجی مدل به اندازهٔ padding × scale بزرگ‌تر است؛
        # این حاشیهٔ اضافه را برمی‌داریم تا تایل‌ها دقیقاً کنار هم بنشینند.
        """
        _, _, h, w = t.shape
        scale = self.scale
        pad = self.tile_pad
        tile = self.tile_size
        out = torch.zeros(
            (1, t.shape[1], h * scale, w * scale),
            dtype=t.dtype, device=t.device,
        )
        # گام تایل‌ها؛ آخرین تایل به لبهٔ تصویر می‌چسبد.
        for y in range(0, h, tile):
            y_end = min(y + tile, h)
            for x in range(0, w, tile):
                x_end = min(x + tile, w)
                # برش تایل واقعی از تصویر (نه صفر) و padding بازتابی.
                crop = t[:, :, y:y_end, x:x_end]
                padded = torch.nn.functional.pad(
                    crop, (pad, pad, pad, pad), mode="reflect")
                with torch.no_grad():
                    res = self.model(padded)
                # برش حاشیهٔ padding از خروجی (pad × scale پیکسل از هر طرف).
                p = pad * scale
                res = res[:, :, p:p + (y_end - y) * scale, p:p + (x_end - x) * scale]
                out[:, :, y * scale:y_end * scale, x * scale:x_end * scale] = res
        return out

    def flush(self) -> List[Frame]:
        # هیچ فریم بافری وجود ندارد.
        return []

    def release(self) -> None:
        # آزادسازی مدل و پاک‌سازی حافظهٔ CUDA.
        if self.model is not None:
            self.model = None
        if _HAS_TORCH and torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()