from vidpipe.progress import ProgressReporter, format_duration, format_header


class _Buffer:
    """Fake stream. tty=True mimics a terminal, tty=False mimics a log file."""

    def __init__(self, tty=True):
        self.parts = []
        self._tty = tty

    def isatty(self):
        return self._tty

    def write(self, value):
        self.parts.append(value)

    def flush(self):
        pass


def _snapshot(**overrides):
    base = {
        "processed": 315,
        "total_frames": 1000,
        "exact_frame_count": True,
        "frames_read": 320,
        "queue_depth": 4,
        "input_queue": 2,
        "output_queue": 1,
        "tiles_completed": 4,
        "tiles_total": 12,
        "tile_size": 192,
        "tile_batch_size": 2,
        "model_name": "realesr-general-x4v3",
        "read_seconds": 0.008,
        "stage_seconds": 2.7,
        "write_seconds": 0.012,
    }
    base.update(overrides)
    return base


def test_format_duration():
    assert format_duration(3661) == "01:01:01"
    assert format_duration(None) == "--:--:--"


def test_format_header_aligns_colons():
    text = format_header([("Input", "x.mp4"), ("Resolution", "1x2")])
    assert text == "Input      : x.mp4\nResolution : 1x2"


def test_panel_has_six_lines_and_exact_percentages():
    reporter = ProgressReporter(interval=1, color=False)
    reporter.started_at -= 5
    lines = reporter.render(_snapshot(), "RUNNING").split("\n")
    assert len(lines) == 6
    assert "31.50% done" in lines[0]
    assert "68.50% left" in lines[0]
    assert "315 / 1,000" in lines[1]
    assert "685 frames" in lines[1]


def test_percentages_are_exact_for_small_totals():
    reporter = ProgressReporter(interval=1, color=False)
    line = reporter.render(_snapshot(processed=1, total_frames=3), "RUNNING").split("\n")[0]
    assert "33.33% done" in line
    assert "66.67% left" in line


def test_approximate_total_is_marked():
    reporter = ProgressReporter(interval=1, color=False)
    lines = reporter.render(_snapshot(exact_frame_count=False), "RUNNING").split("\n")
    assert "315 / ~1,000" in lines[1]
    assert "(approx.)" in lines[0]


def test_gpu_panel_shows_vram_and_utilization():
    reporter = ProgressReporter(interval=1, color=False)
    gpu = {"free_gb": 2.58, "total_gb": 8.0, "allocated_gb": 1.0,
           "reserved_gb": 1.5, "util_percent": 96}
    text = reporter.render(_snapshot(), "RUNNING", gpu)
    assert "5.42 / 8.00 GB" in text
    assert "96%" in text


def test_gpu_utilization_is_na_when_unavailable():
    reporter = ProgressReporter(interval=1, color=False)
    text = reporter.render(_snapshot(), "RUNNING", {"free_gb": 2.58, "total_gb": 8.0})
    assert "GPU" in text and "n/a" in text


def test_per_frame_timing_uses_processed_count():
    reporter = ProgressReporter(interval=1, color=False)
    lines = reporter.render(_snapshot(processed=2, stage_seconds=2.7), "RUNNING").split("\n")
    assert "process 1350 ms" in lines[5]


def test_oom_line_appears_only_after_retries():
    reporter = ProgressReporter(interval=1, color=False)
    assert "OOM" not in reporter.render(_snapshot(), "RUNNING")
    assert "retries 2" in reporter.render(_snapshot(oom_retries=2), "RUNNING")


def test_color_adds_bold_codes():
    reporter = ProgressReporter(interval=1, color=True)
    text = reporter.render(_snapshot(), "RUNNING")
    assert "\033[1m" in text


def test_panel_is_redrawn_in_place_on_tty():
    stream = _Buffer(tty=True)
    reporter = ProgressReporter(interval=1, color=False, stream=stream)
    reporter.print_update("a\nb")
    reporter.print_update("c\nd", final=True)
    assert "".join(stream.parts) == "\033[2Ka\n\033[2Kb\n\033[2A\033[2Kc\n\033[2Kd\n"


def test_log_output_writes_one_record_per_update():
    stream = _Buffer(tty=False)
    reporter = ProgressReporter(interval=1, color=False, stream=stream)
    reporter.print_update("a\nb")
    assert "".join(stream.parts) == "a | b\n"