"""Generate readable, collision-safe output names from the actual stage chain."""

from __future__ import annotations

import re
from pathlib import Path


def _safe_tag(value: str) -> str:
    value = re.sub(r"[^\w.+-]+", "-", value, flags=re.UNICODE).strip("-._")
    return value or "stage"


def processing_tag(stages: list, source_spec, output_spec) -> str:
    labels = []
    contains_denoise = False

    for stage in stages:
        label = getattr(stage, "output_label", stage.__class__.__name__)
        if label and label != "identity":
            labels.append(_safe_tag(str(label)))
        contains_denoise = contains_denoise or bool(getattr(stage, "is_denoise", False))

    if not labels:
        labels.append("identity")

    scale_x = output_spec.width / source_spec.width
    scale_y = output_spec.height / source_spec.height
    if abs(scale_x - 1.0) < 0.001 and abs(scale_y - 1.0) < 0.001:
        scale_tag = "no-upscale"
    elif abs(scale_x - scale_y) < 0.001:
        scale_tag = f"scale-{scale_x:g}x"
    else:
        scale_tag = f"scale-{scale_x:g}x-{scale_y:g}y"

    tags = ["__then__".join(labels), scale_tag]
    if contains_denoise:
        tags.append("denoised")
    return "__".join(tags)


def make_output_path(
    input_path: str,
    output_dir: str,
    stages: list,
    source_spec,
    output_spec,
    extension: str = ".mkv",
) -> str:
    """Return a unique name; an existing output is never silently overwritten."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    input_stem = Path(input_path).stem
    extension = extension if extension.startswith(".") else f".{extension}"
    stem = f"{_safe_tag(input_stem)}__{processing_tag(stages, source_spec, output_spec)}"

    candidate = Path(output_dir) / f"{stem}{extension}"
    counter = 2
    while candidate.exists():
        candidate = Path(output_dir) / f"{stem}_{counter}{extension}"
        counter += 1
    return str(candidate)
