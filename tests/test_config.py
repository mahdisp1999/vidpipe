import pytest

from vidpipe.config import load_config

BASE = """
codec: libx264
crf: 18
preset: medium
queue_depth: 4
device: auto
memory:
  mode: auto
  safety_margin_ram: 0.15
  max_ram_gb: 0.0
  max_vram_gb: 0.0
  prefetch_depth: 2
  prefetch_depth_max: 8
  tile_size: 0
  batch_size: 2
  batch_size_max: 4
stages:
  - name: identity
"""


def _write(tmp_path, text):
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_valid_config_loads(tmp_path):
    config = load_config(_write(tmp_path, BASE))
    assert config["device"] == "auto"
    assert config["memory"]["batch_size"] == 2
    assert config["stages"] == [{"name": "identity"}]


def test_invalid_ram_margin_rejected(tmp_path):
    text = BASE.replace("safety_margin_ram: 0.15", "safety_margin_ram: 0.95")
    with pytest.raises(ValueError):
        load_config(_write(tmp_path, text))


def test_batch_larger_than_max_rejected(tmp_path):
    text = BASE.replace("batch_size: 2", "batch_size: 6")
    with pytest.raises(ValueError):
        load_config(_write(tmp_path, text))


def test_crf_can_be_disabled_for_encoder_specific_quality(tmp_path):
    config = load_config(_write(tmp_path, BASE.replace("crf: 18", "crf: null")))
    assert config["crf"] is None


def test_unknown_stage_rejected(tmp_path):
    text = BASE.replace("- name: identity", "- name: nope")
    with pytest.raises(ValueError):
        load_config(_write(tmp_path, text))


def test_relative_model_path_is_resolved_from_config_directory(tmp_path):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    model_file = model_dir / "fake.pth"
    model_file.write_bytes(b"placeholder")
    text = BASE.replace(
        "  - name: identity",
        "  - name: model\n    model: ./models/fake.pth\n    tile_size: 128",
    )
    config = load_config(_write(tmp_path, text))
    assert config["stages"][0]["model"] == str(model_file)


def test_relative_model_path_must_exist(tmp_path):
    text = BASE.replace(
        "  - name: identity",
        "  - name: model\n    model: ./models/missing.pth\n    tile_size: 128",
    )
    with pytest.raises(ValueError, match="model file not found"):
        load_config(_write(tmp_path, text))


def test_memory_section_must_be_a_mapping(tmp_path):
    memory_block = """memory:
  mode: auto
  safety_margin_ram: 0.15
  max_ram_gb: 0.0
  max_vram_gb: 0.0
  prefetch_depth: 2
  prefetch_depth_max: 8
  tile_size: 0
  batch_size: 2
  batch_size_max: 4
"""
    text = BASE.replace(memory_block, "memory: 7\n")
    with pytest.raises(ValueError, match="memory must be a YAML mapping"):
        load_config(_write(tmp_path, text))


def test_nonzero_memory_tile_size_must_be_at_least_64(tmp_path):
    text = BASE.replace("tile_size: 0", "tile_size: 32")
    with pytest.raises(ValueError, match="memory.tile_size"):
        load_config(_write(tmp_path, text))
