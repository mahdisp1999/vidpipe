"""Readable, dependency-free live progress reporting for the CLI."""

from __future__ import annotations

import sys
import time

_RESET = "\033[0m"
_BOLD = "\033[1m"
_CYAN = "\033[36m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_MAGENTA = "\033[35m"
_RED = "\033[31m"


def format_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "--:--:--"
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class ProgressReporter:
    """Build one detailed status line and redraw it in-place on a TTY."""

    def __init__(self, interval: float = 1.0, color: bool | None = None, stream=None) -> None:
        self.interval = max(0.1, float(interval))
        self.stream = stream or sys.stdout
        self.is_tty = bool(getattr(self.stream, "isatty", lambda: False)())
        self.color = self.is_tty if color is None else bool(color)
        self.started_at = time.monotonic()
        self.last_update = self.started_at
        self.last_processed = 0
        self.active_seconds = 0.0
        self.smoothed_fps = 0.0
        self.previous_state = "RUNNING"

    def _paint(self, text: str, color: str, bold: bool = False) -> str:
        if not self.color:
            return text
        prefix = (_BOLD if bold else "") + color
        return f"{prefix}{text}{_RESET}"

    def render(self, snapshot: dict, state: str, gpu: dict | None = None,
               final: bool = False) -> str:
        now = time.monotonic()
        delta_time = max(0.0, now - self.last_update)
        was_active = self.previous_state not in ("PAUSED", "STOPPED", "STOPPING", "FINISHED", "FAILED")
        if was_active:
            self.active_seconds += delta_time

        processed = int(snapshot.get("processed", 0))
        frame_delta = max(0, processed - self.last_processed)
        # Ignore near-zero intervals (for example, the very first render),
        # otherwise a single frame can produce a meaningless multi-million FPS.
        if was_active and delta_time >= 0.05 and frame_delta:
            instant_fps = frame_delta / delta_time
            self.smoothed_fps = instant_fps if not self.smoothed_fps else (
                0.25 * instant_fps + 0.75 * self.smoothed_fps
            )
        self.last_processed = processed
        self.last_update = now
        self.previous_state = state

        total = snapshot.get("total_frames")
        exact = bool(snapshot.get("exact_frame_count", False))
        if total:
            total = int(total)
            percent = min(100.0, processed * 100.0 / total)
            remaining = max(0, total - processed)
            remaining_percent = max(0.0, 100.0 - percent)
            count = f"{processed:,}/{total:,}" if exact else f"{processed:,}/~{total:,}"
            left = f"{remaining:,} ({remaining_percent:.2f}%)"
            percent_text = f"{percent:6.2f}%" + ("" if exact else " approx.")
            filled = int(round(22 * percent / 100.0))
            bar = "=" * filled + "-" * (22 - filled)
            eta_seconds = remaining / self.smoothed_fps if self.smoothed_fps > 0 else None
        else:
            percent_text, count, left, bar, eta_seconds = "  n/a", f"{processed:,}", "unknown", "-" * 22, None

        average_fps = processed / self.active_seconds if self.active_seconds >= 0.05 else 0.0
        fps = self.smoothed_fps or average_fps
        elapsed = format_duration(now - self.started_at)
        eta = format_duration(eta_seconds)

        tile_total = snapshot.get("tiles_total")
        tile_done = snapshot.get("tiles_completed", 0)
        tile_text = f"{tile_done}/{tile_total}" if tile_total else "n/a"
        tile_size = snapshot.get("tile_size")
        batch_size = snapshot.get("tile_batch_size")
        if tile_size:
            tile_text += f"@{tile_size}px b{batch_size or 1}"

        model = str(snapshot.get("model_name") or "n/a")
        if len(model) > 24:
            model = model[:21] + "..."

        frames_read = int(snapshot.get("frames_read", 0))
        read_ms = 1000 * snapshot.get("read_seconds", 0.0) / max(1, frames_read)
        process_ms = 1000 * snapshot.get("stage_seconds", 0.0) / max(1, processed)
        write_ms = 1000 * snapshot.get("write_seconds", 0.0) / max(1, processed)

        depth = snapshot.get("queue_depth")
        read_queue = snapshot.get("input_queue", 0)
        write_queue = snapshot.get("output_queue", 0)
        queues = (f"Q={read_queue}/{depth},{write_queue}/{depth}"
                  if depth else f"Q={read_queue},{write_queue}")

        if gpu and gpu.get("total_gb", 0) > 0:
            used = max(0.0, gpu["total_gb"] - gpu["free_gb"])
            vram = (f"VRAM={used:.2f}/{gpu['total_gb']:.2f}GB "
                    f"(A{gpu.get('allocated_gb', 0):.1f}/R{gpu.get('reserved_gb', 0):.1f})")
        else:
            vram = "VRAM=n/a"

        state_color = _GREEN if state in ("FINISHED", "RUNNING") else _YELLOW
        if state == "FAILED":
            state_color = _RED

        # One compact dashboard line: the important values are first, while
        # model/queue/timing/VRAM details remain available at the end.
        parts = [
            self._paint(f"[{bar}] {percent_text}", _GREEN if final else _CYAN, True),
            self._paint(f"Frames {count} frames", _CYAN, True),
            f"Remaining {left}",
            self._paint(f"{fps:.2f} FPS", _MAGENTA, True),
            f"Elapsed {elapsed}",
            self._paint(f"ETA {eta}", _YELLOW, True),
            f"Model {model}",
            f"Tiles {tile_text}",
            queues,
            f"Decode/Process/Encode {read_ms:.0f}/{process_ms:.0f}/{write_ms:.0f} ms/frame",
            vram,
            self._paint(state, state_color, True),
        ]
        retries = int(snapshot.get("oom_retries", 0))
        if retries:
            parts.insert(-1, self._paint(f"OOM retries {retries}", _RED, True))
        return "  |  ".join(parts)

    def print_update(self, line: str, final: bool = False) -> None:
        if self.is_tty:
            # Carriage return returns to column zero; ESC[K erases any leftover
            # characters from the previous (possibly longer) status message.
            self.stream.write("\r" + line + "\033[K")
            if final:
                self.stream.write("\n")
        else:
            # A redirected stream has no terminal cursor to redraw. Keep useful
            # newline-delimited records for log files and pipelines.
            self.stream.write(line + "\n")
        self.stream.flush()
