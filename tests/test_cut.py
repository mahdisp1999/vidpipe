import glob
import inspect
import json
import subprocess

import pytest

import vidpipe.cut as cut
from vidpipe.cut import BackendUnavailable, remove_ranges

pytestmark = pytest.mark.skipif(
    subprocess.run(["ffmpeg", "-version"], capture_output=True).returncode != 0,
    reason="ffmpeg is required",
)


def _codec(path, stream):
    return subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", stream, "-show_entries",
         "stream=codec_name", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True).stdout.strip()


def _frame_hashes(path):
    """MD5 of every decoded video frame."""
    text = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v:0", "-f", "framemd5", "-"],
        capture_output=True, text=True, check=True).stdout
    return [line.split(",")[-1].strip() for line in text.splitlines() if line and not line.startswith("#")]


def _audio_packet_sizes(path):
    text = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "packet=size",
         "-of", "json", str(path)], capture_output=True, text=True, check=True).stdout
    return {int(p["size"]) for p in json.loads(text)["packets"]}


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    """12 s clip, a keyframe every second, with audio."""
    path = tmp_path_factory.mktemp("src") / "src.mkv"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "testsrc2=size=160x120:rate=25",
         "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000",
         "-t", "12", "-c:v", "libx264", "-g", "25", "-keyint_min", "25",
         "-sc_threshold", "0", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path)],
        check=True)
    return path


@pytest.fixture
def work(source, tmp_path):
    """A fresh directory that holds only a copy of the source, so listings are exact."""
    target = tmp_path / "in.mkv"
    target.write_bytes(source.read_bytes())
    return target


@pytest.mark.parametrize("backend", ["pipe", "tmpfs", "disk"])
def test_each_backend_cuts_without_reencode_and_leaves_no_files(work, backend):
    out = work.parent / "out.mkv"
    shm_before = set(glob.glob("/dev/shm/.vidpipe-cut-*"))

    assert remove_ranges(str(work), ["3-5", "8-9"], str(out), backend=backend) == 0

    # Only the input and the output remain: no temp segments, lists or temp outputs.
    assert sorted(p.name for p in work.parent.iterdir()) == ["in.mkv", "out.mkv"]
    assert set(glob.glob("/dev/shm/.vidpipe-cut-*")) == shm_before

    # 12 s minus 3 s removed = ~9 s. Cuts snap to keyframes; the one-pass pipe
    # backend is tight, the segment-based ones (legacy behaviour) drift ~1 s per part.
    upper = 9.6 if backend == "pipe" else 13.5
    assert 8.5 <= cut.probe_duration(out) <= upper

    # Stream-copy proof: same codecs, every decoded video frame is bit-identical
    # to a source frame (a re-encode would produce new pixels), audio packets unchanged.
    assert _codec(out, "v:0") == _codec(work, "v:0")
    assert _codec(out, "a:0") == _codec(work, "a:0")
    assert set(_frame_hashes(out)) <= set(_frame_hashes(work))
    assert _audio_packet_sizes(out) <= _audio_packet_sizes(work)


def test_auto_mode_prefers_pipe(work, capsys):
    remove_ranges(str(work), ["3-5"], str(work.parent / "out.mkv"))
    assert "Backend used: pipe" in capsys.readouterr().out


def test_pipe_works_for_mp4(source, tmp_path):
    src = tmp_path / "in.mp4"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
                    "-c", "copy", str(src)], check=True)
    out = tmp_path / "out.mp4"
    remove_ranges(str(src), ["2-4"], str(out), backend="pipe")
    assert 9.0 <= cut.probe_duration(out) <= 10.6


def test_falls_back_to_tmpfs_when_pipe_fails(work, monkeypatch, capsys):
    def broken(self, *args):
        raise subprocess.CalledProcessError(1, "ffmpeg")
    monkeypatch.setattr(cut.PipeBackend, "run", broken)
    remove_ranges(str(work), ["3-5"], str(work.parent / "out.mkv"))
    assert "Backend used: tmpfs" in capsys.readouterr().out


def test_falls_back_to_disk_when_ram_budget_is_too_small(work, monkeypatch, capsys):
    monkeypatch.setattr(cut.PipeBackend, "run", lambda self, *a: (_ for _ in ()).throw(BackendUnavailable("x")))
    monkeypatch.setattr(cut.TmpfsBackend, "ram_share", 0.0)
    out = work.parent / "out.mkv"
    remove_ranges(str(work), ["3-5"], str(out))
    assert "Backend used: disk" in capsys.readouterr().out
    assert out.exists()


def test_failed_run_leaves_no_output_or_temp_files(work, monkeypatch):
    def reject(path, keep):
        raise RuntimeError("bad output")
    monkeypatch.setattr(cut, "_verify", reject)
    with pytest.raises(RuntimeError):
        remove_ranges(str(work), ["3-5"], str(work.parent / "out.mkv"), backend="pipe")
    assert [p.name for p in work.parent.iterdir()] == ["in.mkv"]


def test_unknown_backend_is_rejected(work):
    with pytest.raises(ValueError, match="unknown backend"):
        remove_ranges(str(work), ["3-5"], backend="nope")


def test_existing_call_signature_still_binds():
    # main.py calls: remove_ranges(args.input, args.remove_range, args.output, args.overwrite)
    inspect.signature(remove_ranges).bind("in.mkv", ["1-2"], None, False)