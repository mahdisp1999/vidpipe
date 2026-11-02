"""بارگذار و اعتبارسنجی فایل پیکربندی YAML.

کلیدهایی که در فایل نیستند مقدار پیش‌فرض می‌گیرند.
هر خطای پیکربندی با ValueError و پیام واضح رد می‌شود.
"""

from __future__ import annotations

import copy
import os
from typing import Any, Dict

import yaml

from .registry import resolve

REQUIRED_FIELDS = ("codec", "crf", "preset", "queue_depth")
VALID_DEVICES = ("auto", "gpu", "cpu")
VALID_MEMORY_MODES = ("auto", "manual")

# مقادیر پیش‌فرض وقتی کلیدی در فایل نباشد.
_DEFAULTS: Dict[str, Any] = {
    "device": "auto",
    "memory": {
        "mode": "auto",
        "safety_margin_ram": 0.15,
        "safety_margin_vram": 0.20,
        "max_ram_gb": 0.0,
        "max_vram_gb": 0.0,
        "prefetch_depth": 2,
        "prefetch_depth_max": 8,
        "batch_size_max": 8,
    },
    "stages": [{"name": "identity"}],
}


def _merge_defaults(data: Dict[str, Any]) -> Dict[str, Any]:
    """کلیدهای غایب را با پیش‌فرض پر می‌کند.

    # deepcopy لازم است تا لیست و دیکشنری پیش‌فرض بین اجراهای مختلف مشترک نماند.
    """
    out = copy.deepcopy(_DEFAULTS)
    out.update(data)
    # بخش memory جدا ادغام می‌شود تا کلیدهای غایبش هم پیش‌فرض بگیرند.
    out["memory"] = {**_DEFAULTS["memory"], **data.get("memory", {})}
    return out


def load_config(path: str) -> Dict[str, Any]:
    """فایل YAML را می‌خواند، پیش‌فرض‌ها را پر می‌کند و اعتبارسنجی می‌کند."""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("config must be a mapping")

    # فیلدهای اجباری مرحلهٔ ۱
    for field in REQUIRED_FIELDS:
        if field not in data:
            raise ValueError(f"missing config field: {field}")
    if not isinstance(data["codec"], str):
        raise ValueError("codec must be string")
    if not isinstance(data["crf"], int):
        raise ValueError("crf must be int")
    if not isinstance(data["preset"], str):
        raise ValueError("preset must be string")
    if not isinstance(data["queue_depth"], int) or data["queue_depth"] < 1:
        raise ValueError("queue_depth must be positive int")

    data = _merge_defaults(data)

    # دستگاه
    if data["device"] not in VALID_DEVICES:
        raise ValueError(f"device must be one of {VALID_DEVICES}")

    # حافظه
    mem = data["memory"]
    if mem["mode"] not in VALID_MEMORY_MODES:
        raise ValueError(f"memory.mode must be one of {VALID_MEMORY_MODES}")
    for key in ("safety_margin_ram", "safety_margin_vram"):
        val = mem[key]
        if not isinstance(val, (int, float)) or not (0 <= val <= 0.9):
            raise ValueError(f"memory.{key} must be between 0 and 0.9")
    for key in ("prefetch_depth", "prefetch_depth_max", "batch_size_max"):
        if not isinstance(mem[key], int) or mem[key] < 1:
            raise ValueError(f"memory.{key} must be positive int")
    if mem["prefetch_depth"] > mem["prefetch_depth_max"]:
        raise ValueError("memory.prefetch_depth must be <= memory.prefetch_depth_max")
    for key in ("max_ram_gb", "max_vram_gb"):
        if not isinstance(mem[key], (int, float)) or mem[key] < 0:
            raise ValueError(f"memory.{key} must be non-negative number")

    # مراحل: فهرستی از دیکشنری‌ها که هرکدام کلید name دارند.
    stages = data["stages"]
    if not isinstance(stages, list) or not stages:
        raise ValueError("stages must be a non-empty list")
    for entry in stages:
        if not isinstance(entry, dict):
            raise ValueError("stages entries must be mappings, e.g. '- name: identity'")
        name = entry.get("name")
        if not isinstance(name, str):
            raise ValueError("each stage must have a string 'name'")
        resolve(name)  # نام ناشناس اینجا خطای واضح می‌دهد

        if name == "model":
            model_path = entry.get("model")
            if not isinstance(model_path, str) or not model_path:
                raise ValueError("model stage requires a 'model' file path")
            if not os.path.isfile(model_path):
                raise ValueError(f"model file not found: {model_path!r}")
            tile_size = entry.get("tile_size", 512)
            if not isinstance(tile_size, int) or tile_size < 64:
                raise ValueError("model stage tile_size must be int >= 64")
            tile_pad = entry.get("tile_pad", 16)
            if not isinstance(tile_pad, int) or tile_pad < 0:
                raise ValueError("model stage tile_pad must be int >= 0")
            output_scale = entry.get("output_scale")
            if output_scale is not None and (
                not isinstance(output_scale, (int, float)) or output_scale <= 0
            ):
                raise ValueError("model stage output_scale must be > 0 if given")

    return data