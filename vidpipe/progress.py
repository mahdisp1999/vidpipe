"""Readable, dependency-free live progress panel for the CLI.

ایده‌ی کلی: پنل ثابت چند خطی است. بعد از هر بازنویسی، مکان‌نما با
ESC[nA به ابتدای پنل برمی‌گردد و هر خط با ESC[2K پاک و دوباره نوشته می‌شود.
"""

from __future__ import annotations

import sys
import time

_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_CYAN = "\033[36m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_MAGENTA = "\033[35m"
_RED = "\033[31m"

_BAR_WIDTH = 30
_LABEL_WIDTH = 10
_CLEAR_LINE = "\033[2K"     # پاک کردن کل خط فعلی
_CURSOR_UP = "\033[{}A"    # رفتن n خط به بالا


def format_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "--:--:--"
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def format_header(fields: list[tuple[str, str]]) -> str:
    """Build the one-time header as aligned `key : value` lines."""
    if not fields:
        return ""
    width = max(len(key) for key, _ in fields)
    return "\n".join(f"{key:<{width}} : {value}" for key, value in fields)


class ProgressReporter:
    """Build a fixed multi-line panel and redraw it in place on a TTY."""

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
        self._drawn_lines = 0  # تعداد خطوط پنلی که قبلاً چاپ شده

    def _paint(self, text: str, color: str, bold: bool = False) -> str:
        if not self.color:
            return text
        prefix = (_BOLD if bold else "") + color
        return f"{prefix}{text}{_RESET}"

    def _field(self, label: str, value: str, width: int, color: str = _CYAN) -> str:
        """یک فیلد «برچسب + مقدار» با عرض ثابت.

        نکته: پدینگ قبل از رنگ‌دهی انجام می‌شود تا کدهای ANSI روی عرض اثر نگذارند.
        """
        label_text = f"{label:<{_LABEL_WIDTH}}"
        value_text = f"{value:<{width}}"
        return f"{self._paint(label_text, _DIM)} {self._paint(value_text, color, True)}"

    def render(self, snapshot: dict, state: str, gpu: dict | None = None,
               final: bool = False) -> str:
        """Return the panel as a string with '\\n' between its lines."""
        now = time.monotonic()
        delta_time = max(0.0, now - self.last_update)
        was_active = self.previous_state not in ("PAUSED", "STOPPED", "STOPPING", "FINISHED", "FAILED")
        if was_active:
            self.active_seconds += delta_time

        processed = int(snapshot.get("processed", 0))
        frame_delta = max(0, processed - self.last_processed)
        # فاصله‌های خیلی کوتاه (مثلاً اولین رندر) را نادیده می‌گیریم تا FPS عدد بی‌معنی ندهد.
        if was_active and delta_time >= 0.05 and frame_delta:
            instant_fps = frame_delta / delta_time
            self.smoothed_fps = instant_fps if not self.smoothed_fps else (
                0.25 * instant_fps + 0.75 * self.smoothed_fps
            )
        self.last_processed = processed
        self.last_update = now
        self.previous_state = state

        exact = bool(snapshot.get("exact_frame_count", False))
        total_raw = snapshot.get("total_frames")
        if total_raw:
            total = int(total_raw)
            done = min(processed, total)
            remaining = total - done
            # درصد انجام‌شده و باقی‌مانده مستقیم از شمارش فریم حساب می‌شوند.
            # چون هر دو از یک مخرج هستند، جمعشان همیشه 100 می‌شود.
            done_pct = done * 100.0 / total
            left_pct = remaining * 100.0 / total
            frames_text = f"{processed:,} / {'' if exact else '~'}{total:,}"
            remaining_text = f"{remaining:,} frames"
            done_text = f"{done_pct:.2f}% done" + ("" if exact else " (approx.)")
            left_text = f"{left_pct:.2f}% left"
            filled = int(round(_BAR_WIDTH * done / total))
            eta_seconds = remaining / self.smoothed_fps if self.smoothed_fps > 0 else None
        else:
            frames_text = f"{processed:,} / ?"
            remaining_text = "unknown"
            done_text = "n/a"
            left_text = "n/a"
            filled = 0
            eta_seconds = None

        average_fps = processed / self.active_seconds if self.active_seconds >= 0.05 else 0.0
        fps = self.smoothed_fps or average_fps
        elapsed = format_duration(now - self.started_at)
        eta = format_duration(eta_seconds)

        tiles_total = snapshot.get("tiles_total")
        tile_text = "n/a"
        if tiles_total:
            tile_text = f"{snapshot.get('tiles_completed', 0)} / {tiles_total}"
            tile_size = snapshot.get("tile_size")
            if tile_size:
                tile_text += f" @{tile_size}px b{snapshot.get('tile_batch_size') or 1}"

        depth = snapshot.get("queue_depth")
        in_queue = int(snapshot.get("input_queue", 0))
        out_queue = int(snapshot.get("output_queue", 0))
        queue_text = (f"in {in_queue}/{depth}  out {out_queue}/{depth}"
                      if depth else f"in {in_queue}  out {out_queue}")

        if gpu and gpu.get("total_gb", 0) > 0:
            used = max(0.0, gpu["total_gb"] - gpu["free_gb"])
            vram_text = f"{used:.2f} / {gpu['total_gb']:.2f} GB"
        else:
            vram_text = "n/a"
        util = gpu.get("util_percent") if gpu else None
        gpu_text = f"{util}%" if util is not None else "n/a"

        frames_read = int(snapshot.get("frames_read", 0))
        read_ms = 1000 * snapshot.get("read_seconds", 0.0) / max(1, frames_read)
        process_ms = 1000 * snapshot.get("stage_seconds", 0.0) / max(1, processed)
        write_ms = 1000 * snapshot.get("write_seconds", 0.0) / max(1, processed)

        state_color = _GREEN if state in ("FINISHED", "RUNNING") else _YELLOW
        if state == "FAILED":
            state_color = _RED

        model = str(snapshot.get("model_name") or "n/a")
        retries = int(snapshot.get("oom_retries", 0))

        # خط ۱: وضعیت، نوار پیشرفت و درصدها (مهم‌ترین اطلاعات در بالا)
        status = self._paint(f"{state:<{_LABEL_WIDTH}}", state_color, True)
        bar = self._paint("█" * filled + "░" * (_BAR_WIDTH - filled), _GREEN if final else _CYAN)
        lines = [
            f"{status} {bar} {self._paint(done_text, _GREEN, True)}   {self._paint(left_text, _YELLOW, True)}",
            self._field("Frames", frames_text, 22) + "   " + self._field("Remaining", remaining_text, 14),
            (self._field("Speed", f"{fps:.2f} fps", 12) + "   "
             + self._field("Elapsed", elapsed, 8) + "   "
             + self._field("ETA", eta, 8, _MAGENTA)),
            self._field("Model", model[:24], 24) + "   " + self._field("Tiles", tile_text, 18),
            self._field("Queue", queue_text, 18) + "   " + self._field("VRAM", vram_text, 20),
            self._field("GPU", gpu_text, 6, _MAGENTA) + (
                "   " + self._field("OOM", f"retries {retries}", 12, _RED) if retries else ""
            ),
            self._field("Per frame",
                        f"decode {read_ms:.0f} ms | process {process_ms:.0f} ms | encode {write_ms:.0f} ms",
                        0),
        ]
        return "\n".join(lines)

    def print_header(self, fields: list[tuple[str, str]]) -> None:
        """Print the static run information once, before the live panel starts."""
        title = self._paint("VIDPIPE", _CYAN, True)
        self.stream.write(f"{title}\n{format_header(fields)}\n\n")
        self.stream.flush()

    def print_update(self, line: str, final: bool = False) -> None:
        """Draw a multi-line panel in place on a TTY; write one record otherwise.

        final فقط برای سازگاری امضا نگه داشته شده؛ هر خط با "\\n" تمام می‌شود،
        پس پایان پنل هم خودکار مشخص است.
        """
        lines = line.split("\n")
        if self.is_tty:
            # برگشت به ابتدای پنل قبلی و بازنویسی تک‌تک خطوط
            if self._drawn_lines:
                self.stream.write(_CURSOR_UP.format(self._drawn_lines))
            for item in lines:
                self.stream.write(_CLEAR_LINE + item + "\n")
            self._drawn_lines = len(lines)
        else:
            # در لاگ مکان‌نما وجود ندارد؛ هر به‌روزرسانی یک رکورد یک‌خطی است
            self.stream.write(" | ".join(item.strip() for item in lines) + "\n")
        self.stream.flush()