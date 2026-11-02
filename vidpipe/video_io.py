"""ffmpeg-based input/output wrappers for raw rgb24 frames.

پوشش ورودی/خروجی ویدیو با subprocess های ffmpeg. فریم‌ها فقط در RAM جریان دارند.
"""

from __future__ import annotations

import json
import subprocess
import sys
from typing import Optional, Tuple


class VideoReader:
    """Reads raw rgb24 frames from an ffmpeg decode subprocess.

    # خوانندهٔ فریم: ffprobe ابعاد و fps را تشخیص می‌دهد؛ ffmpeg خروجی rgb24 می‌دهد.
    # اگر ورودی VFR باشد، با -vsync cfr به CFR اجبار می‌شود.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        self.width, self.height, self.fps = self._probe(path)
        self.frame_size = self.width * self.height * 3
        # forces CFR on decode if source is VFR
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-i", path,
            "-vsync", "cfr",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "pipe:1",
        ]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    @staticmethod
    def _probe(path: str) -> Tuple[int, int, float]:
        # خواندن ابعاد و نرخ فریم با ffprobe؛ VFR با r_frame_rate/fps تشخیص داده می‌شود.
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height,r_frame_rate,avg_frame_rate",
             "-of", "json", path],
            capture_output=True, text=True, check=True,
        )
        info = json.loads(out.stdout)
        s = info["streams"][0]
        w, h = int(s["width"]), int(s["height"])
        # prefer avg_frame_rate for VFR detection, fall back to r_frame_rate
        rr = s.get("r_frame_rate", "0/1")
        ar = s.get("avg_frame_rate", rr)
        num, den = (rr.split("/") + ["1"])[:2]
        r_fps = float(num) / float(den) if float(den) else 0.0
        num2, den2 = (ar.split("/") + ["1"])[:2]
        a_fps = float(num2) / float(den2) if float(den2) else 0.0
        # if rates differ -> VFR; CFR handling done in decode via -vsync cfr
        fps = a_fps if a_fps > 0 else r_fps
        if fps <= 0:
            raise RuntimeError("could not probe fps")
        return w, h, fps

    def read_frame(self) -> Optional[bytes]:
        # یک فریم rgb24 می‌خواند؛ None در پایان جریان.
        if self.proc.stdout is None:
            return None
        data = self.proc.stdout.read(self.frame_size)
        if len(data) < self.frame_size:
            return None
        return data

    def close(self) -> None:
        # بستن pipe و انتظار برای خروج ffmpeg (بدون kill).
        if self.proc.stdout:
            self.proc.stdout.close()
        self.proc.wait()


class VideoWriter:
    """Writes raw rgb24 frames to an ffmpeg encoder subprocess.

    # نویسندهٔ فریم: فریم‌های rgb24 به stdin ffmpeg داده می‌شوند.
    # صدا از فایل منبع به عنوان ورودی دوم با -map 1:a? -c:a copy کپی می‌شود.
    # هیچ فایل موقتی ساخته نمی‌شود.
    """

    def __init__(self, out_path: str, width: int, height: int, fps: float,
                 codec: str, crf: int, preset: str, source_path: str) -> None:
        self.out_path = out_path
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_size = width * height * 3
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}", "-r", repr(fps),
            "-i", "pipe:0",
            "-i", source_path,
            "-map", "0:v", "-map", "1:a?",
            "-c:v", codec, "-crf", str(crf), "-preset", preset,
            "-c:a", "copy",
            "-shortest",
            out_path,
        ]
        # codec/crf/preset are config-driven, not hard-coded (only libx264 required this step).
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def write_frame(self, data: bytes) -> None:
        # ارسال یک فریم به stdin encoder.
        if self.proc.stdin is None:
            raise RuntimeError("encoder stdin closed")
        self.proc.stdin.write(data)

    def close(self) -> None:
        # بستن stdin و انتظار برای خروج ffmpeg (بدون kill) تا MKV معتبر بماند.
        if self.proc.stdin:
            self.proc.stdin.close()
        self.proc.wait()
        # propagate encoder errors
        err = self.proc.stderr.read() if self.proc.stderr else b""
        if self.proc.returncode not in (0, None) and err:
            sys.stderr.write(err.decode(errors="replace"))

    @property
    def returncode(self) -> Optional[int]:
        return self.proc.returncode
