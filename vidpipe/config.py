"""Load and validate the YAML configuration used by vidpipe."""

from __future__ import annotations

import copy
import os
from typing import Any

import yaml

from .registry import resolve

_DEFAULTS: dict[str, Any] = {
    "device": "auto",
    "pixel_format": "yuv420p",
    "audio_codec": "copy",
    "queue_depth": 4,
    "memory": {
        "mode": "auto",
        "safety_margin_ram": 0.15,
        "max_ram_gb": 0.0,
        "max_vram_gb": 0.0,
        "prefetch_depth": 2,
        "prefetch_depth_max": 8,
        "tile_size": 0,
        "batch_size": 2,
        "batch_size_max": 4,
    },
    "stages": [{"name": "identity"}],
}


def _merge_defaults(data: dict[str, Any]) -> dict[str, Any]:
    """Fill omitted settings without sharing mutable defaults between runs."""
    result = copy.deepcopy(_DEFAULTS)
    result.update(data)
    result["memory"] = {**_DEFAULTS["memory"], **data.get("memory", {})}
    return result


def load_config(path: str) -> dict[str, Any]:
    """Read YAML and reject invalid values before starting FFmpeg or loading models."""
    if not os.path.isfile(path):
        raise ValueError(f"config file not found: {path!r}")

    with open(path, "r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}

    if not isinstance(raw, dict):
        raise ValueError("config must be a YAML mapping")
    if "memory" in raw and not isinstance(raw["memory"], dict):
        raise ValueError("memory must be a YAML mapping")

    required = ("codec", "crf", "preset", "queue_depth")
    for key in required:
        if key not in raw:
            raise ValueError(f"missing config field: {key}")

    if not isinstance(raw["codec"], str) or not raw["codec"].strip():
        raise ValueError("codec must be a non-empty string")
    if raw["crf"] is not None and (not isinstance(raw["crf"], int) or not 0 <= raw["crf"] <= 63):
        raise ValueError("crf must be null or an integer between 0 and 63")
    if not isinstance(raw["preset"], str) or not raw["preset"].strip():
        raise ValueError("preset must be a non-empty string")
    if not isinstance(raw["queue_depth"], int) or raw["queue_depth"] < 1:
        raise ValueError("queue_depth must be a positive integer")

    config = _merge_defaults(raw)

    if config["device"] not in ("auto", "gpu", "cpu"):
        raise ValueError("device must be one of: auto, gpu, cpu")
    for key in ("pixel_format", "audio_codec"):
        if not isinstance(config[key], str) or not config[key].strip():
            raise ValueError(f"{key} must be a non-empty string")

    memory = config["memory"]
    if memory["mode"] not in ("auto", "manual"):
        raise ValueError("memory.mode must be auto or manual")
    value = memory["safety_margin_ram"]
    if not isinstance(value, (int, float)) or not 0 <= value <= 0.9:
        raise ValueError("memory.safety_margin_ram must be between 0 and 0.9")
    for key in ("prefetch_depth", "prefetch_depth_max", "batch_size", "batch_size_max"):
        value = memory[key]
        if not isinstance(value, int) or value < 1:
            raise ValueError(f"memory.{key} must be a positive integer")
    if memory["prefetch_depth"] > memory["prefetch_depth_max"]:
        raise ValueError("memory.prefetch_depth must be <= memory.prefetch_depth_max")
    if memory["batch_size"] > memory["batch_size_max"]:
        raise ValueError("memory.batch_size must be <= memory.batch_size_max")
    if (not isinstance(memory["tile_size"], int)
            or (memory["tile_size"] != 0 and memory["tile_size"] < 64)):
        raise ValueError("memory.tile_size must be 0 or an integer >= 64")
    for key in ("max_ram_gb", "max_vram_gb"):
        value = memory[key]
        if not isinstance(value, (int, float)) or value < 0:
            raise ValueError(f"memory.{key} must be a non-negative number")

    stages = config["stages"]
    if not isinstance(stages, list) or not stages:
        raise ValueError("stages must be a non-empty list")

    for stage in stages:
        if not isinstance(stage, dict) or not isinstance(stage.get("name"), str):
            raise ValueError("each stages entry must be a mapping with a string 'name'")
        name = stage["name"]
        resolve(name)
        if name != "model":
            continue

        model_path = stage.get("model")
        if not isinstance(model_path, str) or not model_path:
            raise ValueError("model stage requires a non-empty 'model' path")

        # Relative model paths belong to the config file, not the caller's
        # current working directory. This makes --config reliable from anywhere.
        if not os.path.isabs(model_path):
            model_path = os.path.join(os.path.dirname(os.path.abspath(path)), model_path)
        stage["model"] = os.path.normpath(model_path)
        if not os.path.isfile(stage["model"]):
            raise ValueError(f"model file not found: {stage['model']!r}")

        tile_size = stage.get("tile_size", memory["tile_size"] or 192)
        if not isinstance(tile_size, int) or tile_size < 64:
            raise ValueError("model stage tile_size must be an integer >= 64")
        tile_pad = stage.get("tile_pad", 16)
        if not isinstance(tile_pad, int) or tile_pad < 0:
            raise ValueError("model stage tile_pad must be a non-negative integer")
        for key in ("batch_size", "batch_size_max"):
            value = stage.get(key)
            if value is not None and (not isinstance(value, int) or value < 1):
                raise ValueError(f"model stage {key} must be a positive integer")
        if stage.get("output_scale") is not None:
            value = stage["output_scale"]
            if not isinstance(value, (int, float)) or value <= 0:
                raise ValueError("model stage output_scale must be > 0")
        stage_batch = stage.get("batch_size", memory["batch_size"])
        stage_batch_max = stage.get("batch_size_max", memory["batch_size_max"])
        if stage_batch > stage_batch_max:
            raise ValueError("model stage batch_size must be <= batch_size_max")
        if "half" in stage and not isinstance(stage["half"], bool):
            raise ValueError("model stage half must be true or false")

    return config
