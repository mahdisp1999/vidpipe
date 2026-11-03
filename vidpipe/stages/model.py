"""PyTorch/Spandrel image-model stage with batched tiling and CUDA OOM recovery."""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Optional

import numpy as np

from .base import Frame, FrameSpec, Stage

log = logging.getLogger(__name__)

try:
    import torch
    import torch.nn.functional as F
    from spandrel import ImageModelDescriptor, ModelLoader
except ImportError:  # The identity-only pipeline can still run without model packages.
    torch = None
    F = None
    ImageModelDescriptor = None
    ModelLoader = None


class ModelStage(Stage):
    """Apply one image model to each frame, splitting large frames into tiles."""

    def __init__(self) -> None:
        self.model: Optional[Any] = None
        self.device = "cpu"
        self.model_path = ""
        self.model_name = "model"
        self.output_label = "model"
        self.is_denoise = False
        self.scale = 1
        self.tile_size = 192
        self.tile_pad = 16
        self.batch_size = 1
        self.batch_size_max = 1
        self.output_scale: Optional[float] = None
        self.half = False
        self.tiles_completed = 0
        self.tiles_total = 0
        self.last_batch_size = 1
        self.oom_retries = 0
        self.is_processing = False

    def load(self, device: str, params: Any) -> None:
        """Load weights once and apply safe, validated runtime settings."""
        if torch is None or ModelLoader is None:
            raise RuntimeError("PyTorch and spandrel are required for model stages")

        self.device = device
        self.model_path = str(params["model"])
        self.model_name = os.path.splitext(os.path.basename(self.model_path))[0]
        self.output_label = self.model_name
        self.is_denoise = bool(re.search(r"denois|\bwdn\b|noise.?reduc", self.model_name, re.I))
        self.tile_size = int(params.get("tile_size", 192))
        self.tile_pad = int(params.get("tile_pad", 16))
        self.batch_size_max = max(1, int(params.get("batch_size_max", 4)))
        self.batch_size = min(max(1, int(params.get("batch_size", 1))), self.batch_size_max)
        self.output_scale = params.get("output_scale")
        self.half = bool(params.get("half", False))

        descriptor = ModelLoader().load_from_file(self.model_path)
        if not isinstance(descriptor, ImageModelDescriptor):
            raise RuntimeError(f"model {self.model_path!r} is not an image model")

        self.scale = int(descriptor.scale)
        descriptor.to(device).eval()
        if self.half and device == "cuda":
            if descriptor.supports_half:
                descriptor.half()
            else:
                log.warning("model %s does not support FP16; keeping FP32", self.model_name)
                self.half = False

        self.model = descriptor
        log.info(
            "loaded model=%s architecture=%s scale=%dx device=%s precision=%s "
            "tile=%d batch=%d (max=%d)",
            self.model_name,
            descriptor.architecture,
            self.scale,
            device,
            "FP16" if self.half else "FP32",
            self.tile_size,
            self.batch_size,
            self.batch_size_max,
        )

    def plan(self, source_spec: FrameSpec) -> FrameSpec:
        if self.output_scale is not None:
            width = int(round(source_spec.width * self.output_scale))
            height = int(round(source_spec.height * self.output_scale))
        else:
            width = source_spec.width * self.scale
            height = source_spec.height * self.scale
        return FrameSpec(width, height, source_spec.fps, width * height * 3)

    def push(self, frame: Frame) -> list[Frame]:
        if self.model is None or torch is None:
            raise RuntimeError("model stage has not been loaded")

        height, width = frame.spec.height, frame.spec.width
        image = np.frombuffer(frame.data, dtype=np.uint8).reshape(height, width, 3).copy()
        tensor = torch.from_numpy(image).permute(2, 0, 1).float().div_(255.0).unsqueeze(0)
        tensor = tensor.to(self.device)
        if self.half and self.device == "cuda":
            tensor = tensor.half()

        self.is_processing = True
        try:
            with torch.inference_mode():
                output = self._run_tiled(tensor)

                output_spec = self.plan(frame.spec)
                if self.output_scale is not None:
                    output = F.interpolate(
                        output,
                        size=(output_spec.height, output_spec.width),
                        mode="bicubic",
                        align_corners=False,
                    )

                output = output.float().squeeze(0).clamp(0.0, 1.0).mul(255.0).round().byte()
                # This is the one required GPU-to-CPU hand-off for the completed frame.
                data = output.permute(1, 2, 0).contiguous().cpu().numpy().tobytes()
            return [Frame(data=data, spec=output_spec)]
        finally:
            self.is_processing = False

    @staticmethod
    def _is_cuda_oom(error: RuntimeError) -> bool:
        return "out of memory" in str(error).lower()

    def _run_tiled(self, tensor: "torch.Tensor") -> "torch.Tensor":
        """Retry with a smaller batch, then smaller tiles, after a CUDA OOM."""
        tile_size = self.tile_size
        batch_size = self.batch_size

        while True:
            try:
                result = self._run_tiled_once(tensor, tile_size, batch_size)
                self.tile_size = tile_size
                self.batch_size = batch_size
                return result
            except RuntimeError as error:
                if self.device != "cuda" or not self._is_cuda_oom(error):
                    raise

                self.oom_retries += 1
                if batch_size > 1:
                    batch_size = max(1, batch_size // 2)
                    log.warning(
                        "CUDA out of memory; retrying this frame with tile batch size %d",
                        batch_size,
                    )
                elif tile_size > 64:
                    tile_size = max(64, tile_size // 2)
                    log.warning(
                        "CUDA out of memory at batch size 1; retrying with tile size %d",
                        tile_size,
                    )
                else:
                    raise RuntimeError(
                        "CUDA out of memory even with tile_size=64 and batch_size=1. "
                        "Close other GPU workloads, use a smaller model, or try CPU."
                    ) from error

                self.tiles_completed = 0
                self.tiles_total = 0
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

    def _run_tiled_once(
        self,
        tensor: "torch.Tensor",
        tile_size: int,
        batch_size: int,
    ) -> "torch.Tensor":
        """Run equal-shaped tiles together; edge tiles are grouped by shape."""
        _, channels, height, width = tensor.shape
        scale = self.scale
        pad = self.tile_pad

        if pad:
            pad_mode = "reflect" if pad < min(height, width) and min(height, width) > 1 else "replicate"
            padded = F.pad(tensor, (pad, pad, pad, pad), mode=pad_mode)
        else:
            padded = tensor

        output = torch.zeros(
            (1, channels, height * scale, width * scale),
            dtype=tensor.dtype,
            device=tensor.device,
        )

        # Edge tiles can be smaller. Grouping by crop shape lets torch.cat build
        # valid batches without padding a second time or wasting extra VRAM.
        groups: dict[tuple[int, int], list[tuple[int, int, int, int]]] = {}
        for top in range(0, height, tile_size):
            bottom = min(top + tile_size, height)
            for left in range(0, width, tile_size):
                right = min(left + tile_size, width)
                shape = (bottom - top + 2 * pad, right - left + 2 * pad)
                groups.setdefault(shape, []).append((top, bottom, left, right))

        self.tiles_total = sum(len(items) for items in groups.values())
        self.tiles_completed = 0
        self.last_batch_size = 1

        for positions in groups.values():
            for offset in range(0, len(positions), batch_size):
                batch_positions = positions[offset:offset + batch_size]
                crops = [
                    padded[:, :, top:bottom + 2 * pad, left:right + 2 * pad]
                    for top, bottom, left, right in batch_positions
                ]
                batch = crops[0] if len(crops) == 1 else torch.cat(crops, dim=0)
                predictions = self.model(batch)
                self.last_batch_size = len(batch_positions)

                for index, (top, bottom, left, right) in enumerate(batch_positions):
                    tile_output = predictions[index:index + 1]
                    core = tile_output[
                        :,
                        :,
                        pad * scale:pad * scale + (bottom - top) * scale,
                        pad * scale:pad * scale + (right - left) * scale,
                    ]
                    output[:, :, top * scale:bottom * scale, left * scale:right * scale] = core
                    self.tiles_completed += 1

        return output

    def release(self) -> None:
        self.model = None
        if torch is not None and self.device == "cuda":
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
