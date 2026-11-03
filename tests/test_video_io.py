import json
from types import SimpleNamespace

import pytest

from vidpipe.video_io import VideoReader


def _ffprobe_result(stream):
    return SimpleNamespace(
        stdout=json.dumps({"streams": [stream], "format": {"duration": "10.0"}}),
        returncode=0,
    )


def test_probe_uses_exact_frame_count_when_rates_match(monkeypatch):
    stream = {
        "width": "1920",
        "height": "1080",
        "avg_frame_rate": "30/1",
        "r_frame_rate": "30/1",
        "nb_frames": "300",
        "duration": "10.0",
        "codec_name": "h264",
    }
    monkeypatch.setattr("vidpipe.video_io.subprocess.run", lambda *a, **k: _ffprobe_result(stream))

    result = VideoReader._probe("movie.mp4")

    assert result[2] == 30.0
    assert result[4] == 300
    assert result[5] is True
    assert result[6] == "h264"


def test_probe_marks_different_nominal_and_average_rates_as_estimate(monkeypatch):
    stream = {
        "width": "1280",
        "height": "720",
        "avg_frame_rate": "24/1",
        "r_frame_rate": "60/1",
        "nb_frames": "240",
        "duration": "10.0",
        "codec_name": "hevc",
    }
    monkeypatch.setattr("vidpipe.video_io.subprocess.run", lambda *a, **k: _ffprobe_result(stream))

    result = VideoReader._probe("vfr.mp4")

    assert result[2] == 24.0
    assert result[4] == 240
    assert result[5] is False


def test_probe_rejects_invalid_frame_rate(monkeypatch):
    stream = {
        "width": "1280",
        "height": "720",
        "avg_frame_rate": "0/0",
        "r_frame_rate": "0/0",
        "nb_frames": "N/A",
        "duration": "10.0",
        "codec_name": "h264",
    }
    monkeypatch.setattr("vidpipe.video_io.subprocess.run", lambda *a, **k: _ffprobe_result(stream))

    with pytest.raises(RuntimeError, match="valid video frame rate"):
        VideoReader._probe("invalid.mp4")
