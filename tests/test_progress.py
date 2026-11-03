from vidpipe.progress import ProgressReporter, format_duration


def test_format_duration():
    assert format_duration(3661) == "01:01:01"
    assert format_duration(None) == "--:--:--"


def test_progress_has_frame_count_and_eta_fields():
    reporter = ProgressReporter(interval=1, color=False)
    reporter.started_at -= 5
    line = reporter.render(
        {
            "processed": 10,
            "total_frames": 100,
            "exact_frame_count": True,
            "frames_read": 10,
        },
        "RUNNING",
    )
    assert "10/100 frames" in line
    assert "Remaining 90" in line
    assert "FPS" in line
    assert "ETA" in line


def test_progress_can_color_and_bold_important_fields():
    reporter = ProgressReporter(interval=1, color=True)
    line = reporter.render(
        {"processed": 5, "total_frames": 10, "exact_frame_count": True},
        "RUNNING",
    )
    assert "\033[1m" in line
    assert "\033[32m" in line or "\033[36m" in line


def test_progress_includes_processing_timing():
    reporter = ProgressReporter(interval=1, color=False)
    line = reporter.render(
        {
            "processed": 2,
            "total_frames": 10,
            "exact_frame_count": True,
            "frames_read": 2,
            "stage_seconds": 0.8,
        },
        "RUNNING",
    )
    assert "Decode/Process/Encode 0/400/0 ms/frame" in line


class _TerminalBuffer:
    def __init__(self):
        self.parts = []

    def isatty(self):
        return True

    def write(self, value):
        self.parts.append(value)

    def flush(self):
        pass


def test_progress_redraws_same_terminal_line_and_clears_old_text():
    stream = _TerminalBuffer()
    reporter = ProgressReporter(interval=1, color=False, stream=stream)
    reporter.print_update("first status")
    reporter.print_update("short", final=True)
    output = "".join(stream.parts)
    assert output == "\rfirst status\033[K\rshort\033[K\n"


def test_progress_uses_readable_labels_and_detailed_metrics():
    reporter = ProgressReporter(interval=1, color=False)
    line = reporter.render(
        {
            "processed": 315,
            "total_frames": 1000,
            "exact_frame_count": True,
            "frames_read": 320,
            "tiles_completed": 4,
            "tiles_total": 12,
            "tile_size": 192,
            "tile_batch_size": 2,
            "input_queue": 2,
            "output_queue": 1,
            "queue_depth": 4,
            "model_name": "test-model",
            "stage_timings": {"1:test-model": 1.2},
        },
        "RUNNING",
    )
    for expected in ("315/1,000 frames", "Remaining 685 (68.50%)", "FPS",
                     "Elapsed", "ETA", "Tiles 4/12@192px b2", "Q=2/4,1/4",
                     "Decode/Process/Encode", "VRAM", "test-model"):
        assert expected in line


def test_progress_redraws_one_terminal_line_in_place():
    stream = _TerminalBuffer()
    reporter = ProgressReporter(interval=1, color=False, stream=stream)
    reporter.print_update("progress one")
    reporter.print_update("progress two", final=True)
    output = "".join(stream.parts)
    assert output == "\rprogress one\033[K\rprogress two\033[K\n"
