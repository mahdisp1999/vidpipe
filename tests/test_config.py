"""تست‌های اعتبارسنجی پیکربندی.

حاشیهٔ خارج از بازه، عمق بیشتر از حداکثر، نام مرحلهٔ ناشناخته و دستگاه
نامعتبر باید رد شوند.
"""

import pytest

from vidpipe.config import load_config

# پیکربندی پایهٔ معتبر. هر تست فقط یک مقدار را عوض می‌کند.
BASE = """
codec: libx264
crf: 18
preset: medium
queue_depth: 4
device: auto
memory:
  mode: auto
  safety_margin_ram: 0.15
  safety_margin_vram: 0.20
  max_ram_gb: 0.0
  max_vram_gb: 0.0
  prefetch_depth: 2
  prefetch_depth_max: 8
  tile_size: 0
  batch_size: 1
  batch_size_max: 8
stages:
  - name: identity
"""


def _write(tmp_path, text):
    """یک فایل پیکربندی موقت می‌نویسد و مسیرش را برمی‌گرداند."""
    p = tmp_path / "config.yaml"
    p.write_text(text, encoding="utf-8")
    return str(p)


def test_valid_config_loads(tmp_path):
    # پیکربندی معتبر باید بدون خطا لود شود.
    cfg = load_config(_write(tmp_path, BASE))
    assert cfg["device"] == "auto"
    assert cfg["memory"]["mode"] == "auto"
    assert cfg["stages"] == [{"name": "identity"}]


def test_out_of_range_margin_rejected(tmp_path):
    # حاشیهٔ RAM خارج از بازهٔ [0, 0.9] باید رد شود.
    text = BASE.replace("safety_margin_ram: 0.15", "safety_margin_ram: 0.95")
    with pytest.raises(ValueError):
        load_config(_write(tmp_path, text))


def test_depth_gt_max_rejected(tmp_path):
    # prefetch_depth بزرگ‌تر از prefetch_depth_max باید رد شود.
    text = BASE.replace("prefetch_depth: 2", "prefetch_depth: 9")
    with pytest.raises(ValueError):
        load_config(_write(tmp_path, text))


def test_unknown_stage_rejected(tmp_path):
    # نام مرحلهٔ ناشناخته باید رد شود.
    text = BASE.replace("- name: identity", "- name: nope")
    with pytest.raises(ValueError):
        load_config(_write(tmp_path, text))


def test_invalid_device_rejected(tmp_path):
    # دستگاه نامعتبر باید رد شود.
    text = BASE.replace("device: auto", "device: banana")
    with pytest.raises(ValueError):
        load_config(_write(tmp_path, text))