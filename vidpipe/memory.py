"""Device selection, bounded frame-queue sizing and GPU memory telemetry."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger(__name__)
_GB = 1024 ** 3
_FIXED_RAM_MARGIN_GB = 1.2

try:
    import torch
except ImportError:  # PyTorch is needed only for model stages and CUDA telemetry.
    torch = None


@dataclass
class MemoryPlan:
    device: str
    device_reason: str
    free_ram_gb: float
    free_vram_gb: Optional[float]
    total_vram_gb: Optional[float]
    mode: str
    ram_pool_gb: float
    queue_budget_gb: float
    depth: int


def _free_ram_gb() -> float:
    """Read available RAM on Linux; return zero when the OS value is unavailable."""
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as file:
            for line in file:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024 / _GB
    except (OSError, ValueError):
        pass
    return 0.0


def _cuda_available() -> bool:
    return bool(torch is not None and torch.cuda.is_available())


def _free_vram_gb() -> Optional[float]:
    if not _cuda_available():
        return None
    try:
        free, _total = torch.cuda.mem_get_info()
        return free / _GB
    except Exception:  # A telemetry failure must not stop video processing.
        return None


def _total_vram_gb() -> Optional[float]:
    if not _cuda_available():
        return None
    try:
        return torch.cuda.get_device_properties(0).total_memory / _GB
    except Exception:
        return None


def gpu_memory_stats() -> Optional[dict[str, float]]:
    """Return free, total, allocated and reserved VRAM using PyTorch only."""
    if not _cuda_available():
        return None
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        return {
            "free_gb": free_bytes / _GB,
            "total_gb": total_bytes / _GB,
            "allocated_gb": torch.cuda.memory_allocated() / _GB,
            "reserved_gb": torch.cuda.memory_reserved() / _GB,
        }
    except Exception:
        return None


def device_name() -> str:
    if _cuda_available():
        try:
            return torch.cuda.get_device_name(0)
        except Exception:
            return "NVIDIA GPU"
    return "CPU"


def select_device(requested: str) -> tuple[str, str]:
    cuda = _cuda_available()
    if requested == "auto":
        if cuda:
            return "cuda", "CUDA is available"
        log.warning("CUDA is not available; using CPU")
        return "cpu", "CUDA is not available"
    if requested == "gpu":
        if not cuda:
            raise RuntimeError("GPU was requested, but PyTorch cannot access CUDA")
        return "cuda", "GPU was requested"
    if requested == "cpu":
        return "cpu", "CPU was requested"
    raise ValueError(f"invalid device: {requested!r}")


def _clamp_depth(value: int, maximum: int) -> int:
    return max(1, min(value, maximum))


def configure_vram_limit(device: str, memory: dict) -> None:
    """Apply an optional PyTorch VRAM cap before model weights are loaded."""
    limit_gb = memory["max_vram_gb"] if memory["mode"] == "manual" else 0.0
    if device != "cuda" or not limit_gb:
        return

    total_gb = _total_vram_gb()
    if not total_gb:
        log.warning("could not read total VRAM; skipping the configured VRAM limit")
        return

    fraction = min(1.0, limit_gb / total_gb)
    torch.cuda.set_per_process_memory_fraction(fraction)
    log.info("PyTorch CUDA allocator limit set to %.2f GB (%.1f%% of GPU memory)",
             min(limit_gb, total_gb), fraction * 100)


def derive_memory_plan(config: dict, source_spec, output_spec) -> MemoryPlan:
    """Choose a safe frame-queue depth from available RAM.

    The optional PyTorch VRAM allocator limit is set separately before model
    loading. ModelStage handles runtime CUDA OOM by reducing tile batches and
    tile size; this function only sizes the RAM-side queues and reports VRAM.
    """
    device, reason = select_device(config["device"])
    free_ram = _free_ram_gb()
    free_vram = _free_vram_gb()
    total_vram = _total_vram_gb()
    memory = config["memory"]

    if memory["mode"] == "manual":
        depth = memory["prefetch_depth"]
        ram_pool = memory["max_ram_gb"]
        queue_budget = ram_pool

        # In manual mode, max_ram_gb is an optional total budget for both
        # queues together. One queue slot consists of one input and one output
        # frame, so this gives a conservative queue-depth limit.
        if ram_pool > 0:
            frame_bytes = source_spec.frame_size + output_spec.frame_size
            slot_gb = frame_bytes / _GB
            if slot_gb:
                depth = min(depth, max(1, int(ram_pool / slot_gb)))

    else:
        ram_pool = max(0.0, free_ram * (1 - memory["safety_margin_ram"]) - _FIXED_RAM_MARGIN_GB)
        queue_budget = ram_pool * 0.5
        frame_bytes = source_spec.frame_size + output_spec.frame_size
        slot_gb = frame_bytes / _GB
        estimated_depth = int(math.floor(queue_budget / slot_gb)) if slot_gb else 2
        depth = _clamp_depth(estimated_depth, memory["prefetch_depth_max"])

    plan = MemoryPlan(
        device=device,
        device_reason=reason,
        free_ram_gb=free_ram,
        free_vram_gb=free_vram,
        total_vram_gb=total_vram,
        mode=memory["mode"],
        ram_pool_gb=ram_pool,
        queue_budget_gb=queue_budget,
        depth=depth,
    )
    log.info("device=%s (%s), name=%s", device, reason, device_name())
    log.info("RAM available=%.2f GB; VRAM free=%s; total=%s",
             free_ram,
             f"{free_vram:.2f} GB" if free_vram is not None else "n/a",
             f"{total_vram:.2f} GB" if total_vram is not None else "n/a")
    log.info("memory mode=%s; queue-depth ceiling=%d; configured RAM queue budget=%.2f GB",
             plan.mode, plan.depth, plan.queue_budget_gb)
    return plan
