"""Memory plan tests: auto formula, manual caps, device fallback.

# تست‌های برنامهٔ حافظه: فرمول خودکار، سقف دستی، بازگشت دستگاه.
# بدون نیاز به GPU واقعی؛ همهٔ توابع CUDA با monkeypatch شبیه‌سازی می‌شوند.
"""

import vidpipe.memory as mem
from vidpipe.stages.base import FrameSpec


def _cfg(**over):
    """Build a minimal config dict with memory defaults.

    # پیکربندی حداقلی با پیش‌فرض‌های حافظه می‌سازد.
    """
    base = {
        "device": "auto",
        "memory": {
            "mode": "auto",
            "safety_margin_ram": 0.15,
            "safety_margin_vram": 0.20,
            "max_ram_gb": 0.0,
            "max_vram_gb": 0.0,
            "prefetch_depth": 2,
            "prefetch_depth_max": 8,
            "tile_size": 0,
            "batch_size": 1,
            "batch_size_max": 8,
        },
    }
    # حافظه را با پیش‌فرض ادغام می‌کند، سپس بقیهٔ کلیدها را اعمال می‌کند.
    mem = dict(base["memory"])
    mem.update(over.pop("memory", {}))
    base["memory"] = mem
    base.update(over)
    return base


def _spec(w=1920, h=1080, fps=30.0):
    # مشخصات فریم با اندازهٔ محاسبه‌شده.
    return FrameSpec(width=w, height=h, fps=fps, frame_size=w * h * 3)


def test_auto_formula_fixed_inputs(monkeypatch):
    # ورودی ثابت: فرمول خودکار باید عمق مورد انتظار را بدهد.
    monkeypatch.setattr(mem, "_cuda_available", lambda: False)
    monkeypatch.setattr(mem, "_free_ram_gb", lambda: 16.0)
    monkeypatch.setattr(mem, "_free_vram_gb", lambda: None)
    monkeypatch.setattr(mem, "_total_vram_gb", lambda: None)
    cfg = _cfg()
    src = _spec()
    plan = mem.derive_memory_plan(cfg, src, src)
    # ram_pool = 16 * 0.85 - 1.2 = 12.4 ; queue_budget = 6.2
    assert plan.mode == "auto"
    assert plan.device == "cpu"
    assert abs(plan.ram_pool_gb - 12.4) < 1e-6
    assert abs(plan.queue_budget_gb - 6.2) < 1e-6
    # slot_dec = slot_enc = 1920*1080*3/GB ~ 0.00579 ; depth = floor(6.2/0.01158) ~ 535 -> clamp 8
    assert plan.depth == 8


def test_auto_depth_clamped_lower(monkeypatch):
    # عمق محاسبه‌شده کمتر از حد پایین (2) باید به 2 بچسبد.
    monkeypatch.setattr(mem, "_cuda_available", lambda: False)
    monkeypatch.setattr(mem, "_free_ram_gb", lambda: 1.0)
    monkeypatch.setattr(mem, "_free_vram_gb", lambda: None)
    monkeypatch.setattr(mem, "_total_vram_gb", lambda: None)
    cfg = _cfg()
    src = _spec()
    plan = mem.derive_memory_plan(cfg, src, src)
    # ram_pool = 1*0.85-1.2 = -0.35 ; queue_budget = -0.175 -> depth floor(neg) -> clamp 2
    assert plan.depth == 2


def test_auto_depth_clamped_upper(monkeypatch):
    # عمق محاسبه‌شده بیشتر از prefetch_depth_max باید به آن بچسبد.
    monkeypatch.setattr(mem, "_cuda_available", lambda: False)
    monkeypatch.setattr(mem, "_free_ram_gb", lambda: 64.0)
    monkeypatch.setattr(mem, "_free_vram_gb", lambda: None)
    monkeypatch.setattr(mem, "_total_vram_gb", lambda: None)
    cfg = _cfg(memory={"prefetch_depth_max": 4})
    src = _spec()
    plan = mem.derive_memory_plan(cfg, src, src)
    assert plan.depth == 4


def test_manual_caps_win(monkeypatch):
    # حالت دستی: مقادیر پیکربندی سقف سخت هستند و بر مقادیر probed غلبه می‌کنند.
    monkeypatch.setattr(mem, "_cuda_available", lambda: False)
    monkeypatch.setattr(mem, "_free_ram_gb", lambda: 16.0)
    monkeypatch.setattr(mem, "_free_vram_gb", lambda: None)
    monkeypatch.setattr(mem, "_total_vram_gb", lambda: None)
    cfg = _cfg(memory={
        "mode": "manual",
        "prefetch_depth": 3,
        "tile_size": 256,
        "batch_size": 2,
        "max_ram_gb": 4.0,
    })
    src = _spec()
    plan = mem.derive_memory_plan(cfg, src, src)
    assert plan.mode == "manual"
    assert plan.depth == 3
    assert plan.ram_pool_gb == 4.0
    assert plan.queue_budget_gb == 4.0


def test_device_auto_fallback_cpu(monkeypatch):
    # CUDA در دسترس نیست: auto باید به CPU برگردد.
    monkeypatch.setattr(mem, "_cuda_available", lambda: False)
    device, reason = mem.select_device("auto")
    assert device == "cpu"
    assert "CPU" in reason


def test_device_gpu_raises_without_cuda(monkeypatch):
    # CUDA در دسترس نیست: gpu باید خطای واضح بدهد.
    monkeypatch.setattr(mem, "_cuda_available", lambda: False)
    try:
        mem.select_device("gpu")
        assert False
    except RuntimeError as e:
        assert "CUDA" in str(e)


def test_device_cpu_forced(monkeypatch):
    # cpu اجباری: بدون توجه به CUDA، cpu برمی‌گردد.
    monkeypatch.setattr(mem, "_cuda_available", lambda: True)
    device, reason = mem.select_device("cpu")
    assert device == "cpu"