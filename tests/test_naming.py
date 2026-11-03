from pathlib import Path
from types import SimpleNamespace

from vidpipe.naming import make_output_path, processing_tag
from vidpipe.stages.base import FrameSpec


def test_processing_tag_contains_chain_scale_and_denoise():
    stages = [
        SimpleNamespace(output_label="1xDeJPG_OmniSR", is_denoise=False),
        SimpleNamespace(output_label="realesr-general-wdn-x4v3", is_denoise=True),
    ]
    source = FrameSpec(1920, 1080, 30.0, 1920 * 1080 * 3)
    output = FrameSpec(7680, 4320, 30.0, 7680 * 4320 * 3)
    tag = processing_tag(stages, source, output)
    assert "1xDeJPG_OmniSR__then__realesr-general-wdn-x4v3" in tag
    assert "scale-4x" in tag
    assert "denoised" in tag


def test_generated_output_name_is_unique(tmp_path):
    source = FrameSpec(10, 10, 24.0, 300)
    output = FrameSpec(10, 10, 24.0, 300)
    stage = SimpleNamespace(output_label="identity", is_denoise=False)
    first = make_output_path("sample.mp4", str(tmp_path), [stage], source, output)
    Path(first).touch()
    second = make_output_path("sample.mp4", str(tmp_path), [stage], source, output)
    assert first.endswith("__identity__no-upscale.mkv")
    assert second.endswith("_2.mkv")
