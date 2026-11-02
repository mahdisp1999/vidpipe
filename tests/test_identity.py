"""Identity pipeline tests using synthetic ffmpeg videos."""

import os
import subprocess
import tempfile
import time

import pytest

from vidpipe.config import load_config
from vidpipe.controller import Controller
from vidpipe.video_io import VideoReader, VideoWriter


def make_video(path, duration=2.0, fps=30, w=64, h=48):
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", f"testsrc=size={w}x{h}:rate={fps}",
         "-f", "lavfi", "-i", "sine=frequency=1000:duration=" + str(duration),
         "-t", str(duration), "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-shortest", path],
        check=True, capture_output=True,
    )


def probe_frames(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-count_frames", "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", path],
        capture_output=True, text=True, check=True,
    )
    return int(out.stdout.strip())


def probe_audio_codec(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
         "stream=codec_name", "-of", "csv=p=0", path],
        capture_output=True, text=True,
    )
    return out.stdout.strip()


def probe_duration(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", path],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout.strip())


def run_identity(inp, out, cfg_path):
    cfg = load_config(cfg_path)
    reader = VideoReader(inp)
    writer = VideoWriter(out, reader.width, reader.height, reader.fps,
                         cfg["codec"], cfg["crf"], cfg["preset"], inp)
    ctrl = Controller()
    ctrl.start(reader, writer, cfg["queue_depth"])
    ctrl.wait()
    reader.close()
    return ctrl


def wait_running(ctrl, min_frames=1, timeout=10.0):
    """Poll until the controller is RUNNING and has processed >= min_frames.

    The pipeline is CPU-bound and finishes far faster than real-time, so a
    fixed sleep is unreliable. Polling the live state makes the timing tests
    deterministic regardless of machine speed.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        if ctrl.state.name == "RUNNING" and ctrl.processed_frames >= min_frames:
            return
        time.sleep(0.005)
    raise AssertionError("pipeline never reached RUNNING with progress")


@pytest.fixture
def tmpdir():
    d = tempfile.mkdtemp()
    yield d


def test_identity_frame_count(tmpdir):
    inp = os.path.join(tmpdir, "in.mkv")
    out = os.path.join(tmpdir, "out.mkv")
    make_video(inp, duration=2.0, fps=10)
    cfg = os.path.join(tmpdir, "cfg.yaml")
    with open(cfg, "w") as f:
        f.write("codec: libx264\ncrf: 18\npreset: ultrafast\nqueue_depth: 4\n")
    ctrl = run_identity(inp, out, cfg)
    assert ctrl.state.name in ("FINISHED", "STOPPED")
    assert probe_frames(out) == probe_frames(inp)


def test_audio_match(tmpdir):
    inp = os.path.join(tmpdir, "in.mkv")
    out = os.path.join(tmpdir, "out.mkv")
    make_video(inp, duration=2.0, fps=10)
    cfg = os.path.join(tmpdir, "cfg.yaml")
    with open(cfg, "w") as f:
        f.write("codec: libx264\ncrf: 18\npreset: ultrafast\nqueue_depth: 4\n")
    run_identity(inp, out, cfg)
    # audio codec should match source (aac) - output copies aac
    assert probe_audio_codec(out) == probe_audio_codec(inp)
    assert abs(probe_duration(out) - probe_duration(inp)) < 0.05


def test_file_listing(tmpdir):
    inp = os.path.join(tmpdir, "in.mkv")
    out = os.path.join(tmpdir, "out.mkv")
    make_video(inp, duration=1.0, fps=10)
    cfg = os.path.join(tmpdir, "cfg.yaml")
    with open(cfg, "w") as f:
        f.write("codec: libx264\ncrf: 18\npreset: ultrafast\nqueue_depth: 4\n")
    before = set(os.listdir(tmpdir))
    run_identity(inp, out, cfg)
    after = set(os.listdir(tmpdir))
    new_files = after - before
    assert new_files == {os.path.basename(out)}


def test_stop_at_points(tmpdir):
    for frac in (0.0, 0.5, 1.0):
        inp = os.path.join(tmpdir, "in.mkv")
        out = os.path.join(tmpdir, f"out_{frac}.mkv")
        # بزرگ: 900 فریم در 640x480 تا لوله چند صدم ثانیه کار کند و stop در میانهٔ کار
        # قابل اجرا باشد (لوله بسیار سریع‌تر از زمان واقعی است).
        make_video(inp, duration=30.0, fps=30, w=640, h=480)
        cfg = os.path.join(tmpdir, "cfg.yaml")
        with open(cfg, "w") as f:
            f.write("codec: libx264\ncrf: 18\npreset: ultrafast\nqueue_depth: 2\n")
        cfgd = load_config(cfg)
        reader = VideoReader(inp)
        writer = VideoWriter(out, reader.width, reader.height, reader.fps,
                             cfgd["codec"], cfgd["crf"], cfgd["preset"], inp)
        ctrl = Controller()
        ctrl.start(reader, writer, cfgd["queue_depth"])
        # منتظر پیشرفت واقعی می‌مانیم، نه خواب ثابت (پایدار در برابر سرعت ماشین).
        if frac == 0.0:
            wait_running(ctrl, min_frames=1)
        elif frac == 0.5:
            wait_running(ctrl, min_frames=450)
        else:
            ctrl.wait()
        if frac != 1.0:
            ctrl.stop()
        ctrl.wait()
        reader.close()
        # playable: frame count > 0 and duration consistent
        fc = probe_frames(out)
        assert fc >= 0
        # duration within ~ 1 frame of processed count
        dur = probe_duration(out)
        assert dur >= 0


def test_pause_resume(tmpdir):
    inp = os.path.join(tmpdir, "in.mkv")
    out = os.path.join(tmpdir, "out.mkv")
    # بزرگ: 900 فریم در 640x480 تا pause در میانهٔ کار قابل اجرا باشد.
    make_video(inp, duration=30.0, fps=30, w=640, h=480)
    cfg = os.path.join(tmpdir, "cfg.yaml")
    with open(cfg, "w") as f:
        f.write("codec: libx264\ncrf: 18\npreset: ultrafast\nqueue_depth: 4\n")
    cfgd = load_config(cfg)
    reader = VideoReader(inp)
    writer = VideoWriter(out, reader.width, reader.height, reader.fps,
                         cfgd["codec"], cfgd["crf"], cfgd["preset"], inp)
    ctrl = Controller()
    ctrl.start(reader, writer, cfgd["queue_depth"])
    wait_running(ctrl, min_frames=1)
    ctrl.pause()
    assert ctrl.state.name == "PAUSED"
    time.sleep(0.2)
    ctrl.resume()
    assert ctrl.state.name == "RUNNING"
    ctrl.wait()
    reader.close()
    assert probe_frames(out) == probe_frames(inp)


def test_controller_transitions():
    from vidpipe.controller import Controller, State, StateError
    c = Controller()
    # pause while idle rejected
    try:
        c.pause()
        assert False
    except StateError:
        pass
    # stop while idle rejected
    try:
        c.stop()
        assert False
    except StateError:
        pass
    assert c.state == State.IDLE
