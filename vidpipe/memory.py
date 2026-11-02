"""Device selection and memory derivation (auto and manual modes).

# انتخاب دستگاه و استنتاج حافظه (حالت خودکار و دستی).
# auto: CUDA اگر موجود باشد، وگرنه CPU با هشدار. gpu: اگر CUDA نباشد خطا می‌دهد.
# torch اختیاری است؛ اگر نصب نباشد، کد به CPU برمی‌گردد بدون کرش.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional, Tuple

log = logging.getLogger(__name__)

# تلاش برای وارد کردن torch؛ اگر نصب نباشد، CUDA در دسترس نیست.
try:
    import torch  # type: ignore
    _HAS_TORCH = True
except ImportError:  # pragma: no cover - torch نصب نیست
    torch = None  # type: ignore
    _HAS_TORCH = False

_RAM_MARGIN_GB = 1.2  # حاشیهٔ ثابت RAM (گیگابایت)
_GB = 1024 ** 3


@dataclass
class MemoryPlan:
    """Chosen device and memory values for the whole run.

    # مقادیر انتخاب‌شدهٔ دستگاه و حافظه برای کل اجرا.
    """
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
    """Free RAM in GB, reading /proc/meminfo (Linux).

    # RAM آزاد بر حسب گیگابایت؛ از /proc/meminfo خوانده می‌شود.
    """
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    kb = int(line.split()[1])
                    return kb * 1024 / _GB
    except OSError:
        pass
    # fallback: psutil اگر موجود باشد
    try:
        import psutil  # type: ignore
        return psutil.virtual_memory().available / _GB
    except ImportError:
        return 0.0


def _cuda_available() -> bool:
    """True if torch is present and CUDA is available.

    # اگر torch موجود و CUDA در دسترس باشد True برمی‌گرداند.
    """
    return bool(_HAS_TORCH and torch is not None and torch.cuda.is_available())


def _free_vram_gb() -> Optional[float]:
    """Free VRAM in GB via torch.cuda.mem_get_info(), or None if no CUDA.

    # VRAM آزاد بر حسب گیگابایت؛ اگر CUDA نباشد None.
    """
    if not _cuda_available():
        return None
    free, _total = torch.cuda.mem_get_info()  # type: ignore
    return free / _GB


def _total_vram_gb() -> Optional[float]:
    """Total VRAM in GB, or None if no CUDA.

    # کل VRAM بر حسب گیگابایت؛ اگر CUDA نباشد None.
    """
    if not _cuda_available():
        return None
    return torch.cuda.get_device_properties(0).total_memory / _GB  # type: ignore


def _device_name() -> str:
    """Human-readable device name.

    # نام قابل‌خواندن دستگاه.
    """
    if _cuda_available():
        return torch.cuda.get_device_name(0)  # type: ignore
    return "cpu"


def select_device(device: str) -> Tuple[str, str]:
    """Resolve the requested device to an actual device + reason.

    # دستگاه درخواستی را به دستگاه واقعی + دلیل تبدیل می‌کند.
    # auto: CUDA اگر موجود، وگرنه CPU با هشدار. gpu: بدون CUDA خطا. cpu: اجباری.
    """
    cuda = _cuda_available()
    if device == "auto":
        if cuda:
            return "cuda", "auto: CUDA available"
        log.warning("auto: CUDA not available, falling back to CPU")
        return "cpu", "auto: CUDA unavailable, CPU fallback"
    if device == "gpu":
        if not cuda:
            raise RuntimeError("device 'gpu' requested but CUDA is not available")
        return "cuda", "gpu: requested"
    if device == "cpu":
        return "cpu", "cpu: requested"
    raise ValueError(f"invalid device: {device!r} (expected auto|gpu|cpu)")


def _clamp_depth(depth: int, lo: int, hi: int) -> int:
    """Clamp depth into [lo, hi].

    # عمق را در بازهٔ [lo, hi] محدود می‌کند.
    """
    return max(lo, min(hi, depth))


def derive_memory_plan(cfg: dict, src_spec, out_spec) -> MemoryPlan:
    """Derive the memory plan from config (auto or manual).

    # برنامهٔ حافظه را از پیکربندی (خودکار یا دستی) استنتاج می‌کند.
    """
    device, reason = select_device(cfg["device"])
    free_ram = _free_ram_gb()
    free_vram = _free_vram_gb()
    total_vram = _total_vram_gb()
    mode = cfg["memory"]["mode"]

    if mode == "manual":
        # حالت دستی: مقادیر پیکربندی سقف سخت هستند.
        depth = cfg["memory"]["prefetch_depth"]
        ram_pool = cfg["memory"]["max_ram_gb"]
        queue_budget = ram_pool
        if device == "cuda" and cfg["memory"]["max_vram_gb"] and total_vram:
            # سقف VRAM دستی: کسر حافظهٔ فرایند را تنظیم می‌کند.
            torch.cuda.set_per_process_memory_fraction(  # type: ignore
                cfg["memory"]["max_vram_gb"] / total_vram)
    else:
        # حالت خودکار: فرمول‌های حاشیهٔ امن.
        margin_ram = cfg["memory"]["safety_margin_ram"]
        margin_vram = cfg["memory"]["safety_margin_vram"]
        ram_pool = free_ram * (1 - margin_ram) - _RAM_MARGIN_GB
        queue_budget = 0.5 * ram_pool
        slot_dec = src_spec.frame_size / _GB
        slot_enc = out_spec.frame_size / _GB
        denom = slot_dec + slot_enc
        depth = _clamp_depth(
            int(math.floor(queue_budget / denom)) if denom > 0 else 2,
            2, cfg["memory"]["prefetch_depth_max"])
        # vram_pool محاسبه می‌شود ولی در این مرحله فقط برای لاگ است.
        vram_pool = (free_vram * (1 - margin_vram)
                     if free_vram is not None else 0.0)
        log.info("auto vram_pool=%.2f GB (context/weights estimates deferred)", vram_pool)

    plan = MemoryPlan(
        device=device, device_reason=reason,
        free_ram_gb=free_ram, free_vram_gb=free_vram, total_vram_gb=total_vram,
        mode=mode, ram_pool_gb=ram_pool, queue_budget_gb=queue_budget,
        depth=depth,
    )
    _log_plan(plan)
    return plan


def _log_plan(plan: MemoryPlan) -> None:
    """Log every chosen value and the mode once at startup.

    # همهٔ مقادیر انتخاب‌شده و حالت را یک بار در شروع لاگ می‌کند.
    """
    log.info("device=%s (%s) name=%s", plan.device, plan.device_reason, _device_name())
    log.info("free_ram=%.2f GB free_vram=%s total_vram=%s",
             plan.free_ram_gb,
             f"{plan.free_vram_gb:.2f} GB" if plan.free_vram_gb is not None else "n/a",
             f"{plan.total_vram_gb:.2f} GB" if plan.total_vram_gb is not None else "n/a")
    log.info("mode=%s ram_pool=%.2f GB queue_budget=%.2f GB depth=%d",
             plan.mode, plan.ram_pool_gb, plan.queue_budget_gb, plan.depth)