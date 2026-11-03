import vidpipe.memory as memory
from vidpipe.stages.base import FrameSpec


def _config():
    return {
        "device": "auto",
        "queue_depth": 4,
        "memory": {
            "mode": "auto",
            "safety_margin_ram": 0.15,
            "max_ram_gb": 0.0,
            "max_vram_gb": 0.0,
            "prefetch_depth": 2,
            "prefetch_depth_max": 8,
        },
    }


def _spec(width=1920, height=1080):
    return FrameSpec(width, height, 30.0, width * height * 3)


def test_auto_memory_plan_clamps_depth(monkeypatch):
    monkeypatch.setattr(memory, "_cuda_available", lambda: False)
    monkeypatch.setattr(memory, "_free_ram_gb", lambda: 16.0)
    monkeypatch.setattr(memory, "_free_vram_gb", lambda: None)
    monkeypatch.setattr(memory, "_total_vram_gb", lambda: None)
    plan = memory.derive_memory_plan(_config(), _spec(), _spec())
    assert plan.device == "cpu"
    assert abs(plan.ram_pool_gb - 12.4) < 1e-6
    assert abs(plan.queue_budget_gb - 6.2) < 1e-6
    assert plan.depth == 8


def test_auto_memory_plan_uses_safe_minimum(monkeypatch):
    monkeypatch.setattr(memory, "_cuda_available", lambda: False)
    monkeypatch.setattr(memory, "_free_ram_gb", lambda: 1.0)
    monkeypatch.setattr(memory, "_free_vram_gb", lambda: None)
    monkeypatch.setattr(memory, "_total_vram_gb", lambda: None)
    plan = memory.derive_memory_plan(_config(), _spec(), _spec())
    assert plan.depth == 1


def test_manual_mode_uses_configured_prefetch_depth(monkeypatch):
    monkeypatch.setattr(memory, "_cuda_available", lambda: False)
    monkeypatch.setattr(memory, "_free_ram_gb", lambda: 8.0)
    monkeypatch.setattr(memory, "_free_vram_gb", lambda: None)
    monkeypatch.setattr(memory, "_total_vram_gb", lambda: None)
    config = _config()
    config["memory"]["mode"] = "manual"
    config["memory"]["prefetch_depth"] = 3
    config["memory"]["max_ram_gb"] = 4.0
    plan = memory.derive_memory_plan(config, _spec(), _spec())
    assert plan.depth == 3


def test_manual_ram_budget_caps_queue_depth(monkeypatch):
    monkeypatch.setattr(memory, "_cuda_available", lambda: False)
    monkeypatch.setattr(memory, "_free_ram_gb", lambda: 8.0)
    monkeypatch.setattr(memory, "_free_vram_gb", lambda: None)
    monkeypatch.setattr(memory, "_total_vram_gb", lambda: None)
    config = _config()
    config["memory"].update({"mode": "manual", "prefetch_depth": 8, "max_ram_gb": 0.01})

    plan = memory.derive_memory_plan(config, _spec(1920, 1080), _spec(1920, 1080))

    assert plan.depth == 1
    assert plan.queue_budget_gb == 0.01


def test_manual_vram_limit_is_applied_before_model_loading(monkeypatch):
    calls = []
    monkeypatch.setattr(memory, "_total_vram_gb", lambda: 8.0)
    monkeypatch.setattr(memory.torch.cuda, "set_per_process_memory_fraction",
                        lambda fraction: calls.append(fraction))

    memory.configure_vram_limit("cuda", {"mode": "manual", "max_vram_gb": 6.0})

    assert calls == [0.75]


def test_auto_mode_does_not_set_a_manual_vram_limit(monkeypatch):
    calls = []
    monkeypatch.setattr(memory, "_total_vram_gb", lambda: 8.0)
    monkeypatch.setattr(memory.torch.cuda, "set_per_process_memory_fraction",
                        lambda fraction: calls.append(fraction))

    memory.configure_vram_limit("cuda", {"mode": "auto", "max_vram_gb": 6.0})

    assert calls == []
