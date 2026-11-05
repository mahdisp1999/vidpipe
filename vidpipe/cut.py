"""Lossless removal of multiple time ranges using FFmpeg stream copy."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable


def parse_timestamp(value: str) -> float:
    """Parse seconds, MM:SS, or HH:MM:SS[.fraction] into seconds."""
    text = value.strip()
    if not text:
        raise ValueError("empty timestamp")
    try:
        parts = [float(part) for part in text.split(":")]
    except ValueError as error:
        raise ValueError(f"invalid timestamp: {value!r}") from error
    if not 1 <= len(parts) <= 3 or any(part < 0 for part in parts):
        raise ValueError(f"invalid timestamp: {value!r}")
    if len(parts) >= 2 and parts[-1] >= 60:
        raise ValueError(f"seconds must be below 60 in timestamp: {value!r}")
    if len(parts) == 3 and parts[-2] >= 60:
        raise ValueError(f"minutes must be below 60 in timestamp: {value!r}")
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    return parts[0] * 3600 + parts[1] * 60 + parts[2]


def normalize_remove_ranges(
    specifications: Iterable[str], duration: float
) -> list[tuple[float, float]]:
    """Parse, validate, sort, and merge overlapping removal intervals."""
    ranges: list[tuple[float, float]] = []
    for specification in specifications:
        if "-" not in specification:
            raise ValueError(
                f"invalid range {specification!r}; use START-END, e.g. 00:10:00-00:15:00"
            )
        start_text, end_text = specification.split("-", 1)
        start = parse_timestamp(start_text)
        end = parse_timestamp(end_text)
        if end <= start:
            raise ValueError(f"range end must be after its start: {specification!r}")
        if end > duration:
            raise ValueError(
                f"range {specification!r} exceeds video duration ({duration:.3f} seconds)"
            )
        ranges.append((start, end))

    if not ranges:
        raise ValueError("at least one --remove-range is required")

    ranges.sort()
    merged: list[list[float]] = []
    for start, end in ranges:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def build_keep_ranges(
    remove_ranges: Iterable[tuple[float, float]], duration: float
) -> list[tuple[float, float]]:
    """Return the complementary intervals that should be kept."""
    keep: list[tuple[float, float]] = []
    cursor = 0.0
    for start, end in remove_ranges:
        if start > cursor:
            keep.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < duration:
        keep.append((cursor, duration))
    if not keep:
        raise ValueError("the requested ranges remove the entire video")
    return keep


def probe_duration(input_path: Path) -> float:
    """Read the container duration with FFprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "json", str(input_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(result.stdout)
    value = (data.get("format") or {}).get("duration")
    if value in (None, "N/A"):
        raise RuntimeError("FFprobe could not determine the input duration")
    duration = float(value)
    if duration <= 0:
        raise RuntimeError(f"invalid input duration: {duration}")
    return duration


def _concat_quote(path: Path, prefix: str = "") -> str:
    """Quote a path for FFmpeg's concat-demuxer file syntax."""
    value = str(path.resolve()).replace("\\", "\\\\").replace("'", "\\'")
    return f"file '{prefix}{value}'\n"


class BackendUnavailable(RuntimeError):
    """A backend cannot run here; remove_ranges moves on to the next one."""


def _temp_output(output_path: Path) -> Path:
    """A free temporary name beside the output; the final file is replaced only after success."""
    fd, name = tempfile.mkstemp(
        prefix=f".{output_path.stem}.vidpipe-", suffix=output_path.suffix, dir=str(output_path.parent)
    )
    os.close(fd)
    Path(name).unlink()
    return Path(name)


def _verify(path: Path, keep) -> None:
    """Reject empty or truncated output. Stream copy snaps cuts to keyframes, so allow slack."""
    expected = sum(end - start for start, end in keep)
    got = probe_duration(path)
    if abs(got - expected) > 2.0 * len(keep) + 0.05 * expected:
        raise RuntimeError(f"output is {got:.3f}s, expected about {expected:.3f}s")


class PipeBackend:
    """One FFmpeg process reads the source once; the cut list arrives on stdin.

    No segment files and no list file are written, and nothing is buffered
    in RAM beyond FFmpeg's own packet queue.
    """

    name = "pipe"

    def run(self, input_path: Path, keep, duration: float, output_path: Path) -> None:
        lines = []
        for start, end in keep:
            lines.append(_concat_quote(input_path, prefix="file:"))
            if start > 0:
                lines.append(f"inpoint {start:.6f}\n")
            if end < duration:
                lines.append(f"outpoint {end:.6f}\n")
        temporary_output = _temp_output(output_path)
        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            # A list read from a pipe needs the pipe protocol and "file:" paths.
            "-f", "concat", "-safe", "0", "-protocol_whitelist", "file,pipe",
            "-i", "pipe:0", "-map", "0", "-c", "copy", str(temporary_output),
        ]
        print("[vidpipe] Cutting and joining in one pass (stream copy)...")
        try:
            subprocess.run(command, input="".join(lines), text=True, check=True)
            _verify(temporary_output, keep)
            os.replace(temporary_output, output_path)
        finally:
            temporary_output.unlink(missing_ok=True)


class DiskBackend:
    """Current behaviour: kept parts become files beside the output, then are joined."""

    name = "disk"

    def run(self, input_path: Path, keep, duration: float, output_path: Path) -> None:
        _copy_via_segments(input_path, keep, duration, output_path, output_path.parent)


def _free_ram_bytes() -> int:
    """MemAvailable from /proc/meminfo (Linux); 0 when unreadable."""
    try:
        with open("/proc/meminfo", encoding="utf-8") as file:
            for line in file:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError):
        pass
    return 0


class TmpfsBackend:
    """Same as DiskBackend, but the kept parts live in RAM (/dev/shm)."""

    name = "tmpfs"
    root = "/dev/shm"   # shortcut: add XDG_RUNTIME_DIR if a system has no /dev/shm
    margin = 1.15       # estimate = source size x kept fraction x margin
    ram_share = 0.5     # never plan to use more than half of what is free

    def run(self, input_path: Path, keep, duration: float, output_path: Path) -> None:
        if not os.path.isdir(self.root) or not os.access(self.root, os.W_OK):
            raise BackendUnavailable(f"{self.root} is not available")
        kept = sum(end - start for start, end in keep) / duration
        need = input_path.stat().st_size * kept * self.margin
        budget = min(shutil.disk_usage(self.root).free, _free_ram_bytes()) * self.ram_share
        if need > budget:
            raise BackendUnavailable(
                f"kept parts need ~{need / 2**30:.2f} GiB but only {budget / 2**30:.2f} GiB of RAM may be used"
            )
        _copy_via_segments(input_path, keep, duration, output_path, Path(self.root))


# Order matters: first = least disk I/O, last = always-works fallback.
BACKENDS = (PipeBackend, TmpfsBackend, DiskBackend)


def _run_backends(input_path: Path, keep, duration: float, output_path: Path, only: str | None) -> None:
    chosen = [cls() for cls in BACKENDS if only in (None, cls.name)]
    if not chosen:
        raise ValueError(f"unknown backend {only!r}; use one of {[c.name for c in BACKENDS]}")
    for index, backend in enumerate(chosen):
        try:
            backend.run(input_path, keep, duration, output_path)
            print(f"[vidpipe] Backend used: {backend.name}")
            return
        except (BackendUnavailable, subprocess.CalledProcessError, OSError, RuntimeError) as error:
            if index == len(chosen) - 1:
                raise
            print(f"[vidpipe] Backend {backend.name} failed ({error}); trying {chosen[index + 1].name}",
                  file=sys.stderr)


def remove_ranges(
    input_name: str,
    specifications: Iterable[str],
    output_name: str | None = None,
    overwrite: bool = False,
    backend: str | None = None,
) -> int:
    """Remove multiple intervals and concatenate kept parts without re-encoding."""
    input_path = Path(input_name).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"input video not found: {input_path}")

    duration = probe_duration(input_path)
    remove = normalize_remove_ranges(specifications, duration)
    keep = build_keep_ranges(remove, duration)

    if output_name:
        output_path = Path(output_name).expanduser()
        if not output_path.suffix:
            output_path = output_path.with_suffix(input_path.suffix or ".mkv")
    else:
        output_path = input_path.with_name(f"{input_path.stem}_cut{input_path.suffix}")
    output_path = output_path.resolve()

    if output_path == input_path:
        raise ValueError("output path must be different from the input path")
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"output already exists: {output_path}; use --overwrite to replace it"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"[vidpipe] Duration: {duration:.3f}s | "
        f"Removing {len(remove)} range(s) | Keeping {len(keep)} segment(s)"
    )
    # backend: "pipe", "tmpfs" or "disk" forces one; None tries them in that order.
    _run_backends(input_path, keep, duration, output_path, backend)

    removed_seconds = sum(end - start for start, end in remove)
    print(
        f"[vidpipe] Done: {output_path}\n"
        f"[vidpipe] Removed {removed_seconds:.3f}s; "
        f"kept approximately {duration - removed_seconds:.3f}s."
    )
    return 0


def _copy_via_segments(input_path: Path, keep, duration: float, output_path: Path, work_dir: Path) -> None:
    """Disk/tmpfs strategy: cut kept parts into work_dir, then join them into output_path."""
    # Temporary segments live beside the output, avoiding an unexpectedly small /tmp.
    with tempfile.TemporaryDirectory(
        prefix=".vidpipe-cut-", dir=str(work_dir)
    ) as temporary_directory:
        temp_dir = Path(temporary_directory)
        segment_paths: list[Path] = []
        segment_suffix = output_path.suffix or input_path.suffix or ".mkv"

        for index, (start, end) in enumerate(keep, start=1):
            segment_path = temp_dir / f"part_{index:03d}{segment_suffix}"
            command = ["ffmpeg", "-hide_banner", "-loglevel", "warning", "-y"]
            if start > 0:
                command += ["-ss", f"{start:.6f}"]
            command += ["-i", str(input_path)]
            if end < duration:
                command += ["-t", f"{end - start:.6f}"]
            command += [
                "-map", "0", "-c", "copy", "-avoid_negative_ts", "make_zero",
                str(segment_path),
            ]
            print(f"[vidpipe] Creating kept segment {index}/{len(keep)}...")
            subprocess.run(command, check=True)
            segment_paths.append(segment_path)

        concat_list = temp_dir / "concat.txt"
        concat_list.write_text(
            "".join(_concat_quote(path) for path in segment_paths),
            encoding="utf-8",
        )

        # Write to a temporary destination; replace the final file only after success.
        fd, temporary_output_name = tempfile.mkstemp(
            prefix=f".{output_path.stem}.vidpipe-",
            suffix=output_path.suffix,
            dir=str(output_path.parent),
        )
        os.close(fd)
        temporary_output = Path(temporary_output_name)
        temporary_output.unlink()

        try:
            command = [
                "ffmpeg", "-hide_banner", "-loglevel", "warning", "-y",
                "-f", "concat", "-safe", "0", "-i", str(concat_list),
                "-map", "0", "-c", "copy", str(temporary_output),
            ]
            print("[vidpipe] Concatenating segments (stream copy)...")
            subprocess.run(command, check=True)
            os.replace(temporary_output, output_path)
        finally:
            temporary_output.unlink(missing_ok=True)