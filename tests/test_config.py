"""Config validation tests.

# تست‌های اعتبارسنجی پیکربندی: حاشیه‌های خارج از بازه و عمق بیش از حداکثر رد می‌شوند.
"""

import pytest

from vidpipe.config import load_config


def _write(tmp_path, text):
    # یک فایل پیکربندی موقت می‌نویسد و مسیرش را برمی‌گرداند.
    p = tmp_path / "config.yaml"
    p.write_text(text, encoding="utf-8")
    return str(p)


def test_valid_config_loads(tmp_path):
    # پیکربندی معتبر باید بدون خطا بارگذاری شود.
    path = _write(tmp_path, """
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
  - identity
""")
    cfg = load_config(path)
    assert cfg["device"] == "auto"
    assert cfg["memory"]["mode"] == "auto"
    assert cfg["stages"] == ["identity"]


def test_out_of_range_margin_rejected(tmp_path):
    # حاشیهٔ RAM خارج از بازهٔ [0, 0.9] باید رد شود.
    path = _write(tmp_path, """
codec: libx264
crf: 18
preset: medium
queue_depth: 4
device: auto
memory:
  mode: auto
  safety_margin_ram: 0.95
  safety_margin_vram: 0.20
  max_ram_gb: 0.0
  max_vram_gb: 0.0
  prefetch_depth: 2
  prefetch_depth_max: 8
  tile_size: 0
  batch_size: 1
  batch_size_max: 8
stages:
  - identity
""")
    with pytest.raises(ValueError):
        load_config(path)


def test_depth_gt_max_rejected(tmp_path):
    # prefetch_depth بیشتر از prefetch_depth_max باید رد شود.
    path = _write(tmp_path, """
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
  prefetch_depth: 9
  prefetch_depth_max: 8
  tile_size: 0
  batch_size: 1
  batch_size_max: 8
stages:
  - identity
""")
    with pytest.raises(ValueError):
        load_config(path)


def test_unknown_stage_rejected(tmp_path):
    # نام مرحلهٔ ناشناخته باید رد شود.
    path = _write(tmp_path, """
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
  - nope
""")
    with pytest.raises(ValueError):
        load_config(path)


def test_invalid_device_rejected(tmp_path):
    # دستگاه نامعتبر باید رد شود.
    path = _write(tmp_path, """
codec: libx264
crf: 18
preset: medium
queue_depth: 4
device: banana
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
  - identity
""")
    with pytest.raises(ValueError):
        load_config(path)