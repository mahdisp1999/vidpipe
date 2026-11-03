"""FFmpeg input/output helpers; frames move through memory as raw RGB24."""

from __future__ import annotations

import json
import os
import subprocess
from typing import Optional


class VideoReader:
    """Decode video frames with FFmpeg and expose basic ffprobe metadata."""

    def __init__(self, path: str, input_format: Optional[str] = None, decoder: Optional[str] = None):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"input video not found: {path}")
        self.path = path
        self._reached_eof = False
        self.width, self.height, self.fps, self.duration, self.total_frames, self.exact_frame_count, self.codec = self._probe(path)
        self.frame_size = self.width * self.height * 3

        command = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
        if input_format:
            command += ["-f", input_format]
        if decoder:
            command += ["-c:v", decoder]
        command += [
            "-i", path,
            "-vsync", "cfr",
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "pipe:1",
        ]
        self.command = command
        self.proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=None)

    @staticmethod
    def _probe(path: str) -> tuple[int, int, float, float, Optional[int], bool, str]:
        command = [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,duration,codec_name:format=duration",
            "-of", "json", path,
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        info = json.loads(result.stdout)
        streams = info.get("streams") or []
        if not streams:
            raise RuntimeError(f"no video stream found in {path!r}")
        stream = streams[0]

        width, height = int(stream["width"]), int(stream["height"])

        def parse_rate(value: str | None) -> float:
            if not value or value in ("N/A", "0/0"):
                return 0.0
            try:
                numerator, denominator = (value.split("/") + ["1"])[:2]
                denominator_value = float(denominator)
                return float(numerator) / denominator_value if denominator_value else 0.0
            except (TypeError, ValueError, ZeroDivisionError):
                return 0.0

        average_fps = parse_rate(stream.get("avg_frame_rate"))
        nominal_fps = parse_rate(stream.get("r_frame_rate"))
        fps = average_fps or nominal_fps
        if fps <= 0:
            raise RuntimeError("ffprobe could not determine a valid video frame rate")

        duration_value = stream.get("duration") or (info.get("format") or {}).get("duration")
        duration = float(duration_value) if duration_value not in (None, "N/A") else 0.0
        raw_count = stream.get("nb_frames")

        # FFmpeg decodes this pipeline as CFR. If the average and nominal
        # rates differ, nb_frames may not equal the number of frames emitted
        # after CFR normalization, so show an estimate instead of false precision.
        variable_rate = bool(
            average_fps and nominal_fps
            and abs(average_fps - nominal_fps) > max(1e-6, nominal_fps * 0.001)
        )
        exact = raw_count not in (None, "N/A", "0") and not variable_rate
        total_frames = int(raw_count) if exact else (round(duration * fps) if duration > 0 else None)
        return width, height, fps, duration, total_frames, exact, stream.get("codec_name", "unknown")

    def read_frame(self) -> Optional[bytes]:
        if self.proc.stdout is None:
            return None
        data = self.proc.stdout.read(self.frame_size)
        if not data:
            self._reached_eof = True
            return None
        if len(data) != self.frame_size:
            raise RuntimeError(
                f"decoder returned an incomplete frame ({len(data)} of {self.frame_size} bytes)"
            )
        return data

    def close(self, allow_incomplete: bool = False) -> None:
        if self.proc.stdout and not self.proc.stdout.closed:
            self.proc.stdout.close()
        return_code = self.proc.wait()
        if return_code not in (0, None) and not (allow_incomplete and not self._reached_eof):
            raise RuntimeError(f"FFmpeg decoder exited with code {return_code}")


class VideoWriter:
    """Encode RGB24 frames; copy the source audio stream unless disabled."""

    def __init__(
        self,
        out_path: str,
        width: int,
        height: int,
        fps: float,
        codec: str,
        crf: Optional[int],
        preset: str,
        source_path: str,
        pixel_format: str = "yuv420p",
        audio_codec: str = "copy",
        output_format: Optional[str] = None,
        video_options: Optional[list[tuple[str, str]]] = None,
        overwrite: bool = False,
    ) -> None:
        self.out_path = out_path
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_size = width * height * 3
        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y" if overwrite else "-n",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}", "-r", repr(fps),
            "-i", "pipe:0",
            "-i", source_path,
            "-map", "0:v",
            "-c:v", codec,
            "-preset", preset,
            "-pix_fmt", pixel_format,
        ]

        if crf is not None:
            command += ["-crf", str(crf)]
        for key, value in video_options or []:
            command += [f"-{key}", value]

        if audio_codec.lower() == "none":
            command += ["-an"]
        else:
            command += ["-map", "1:a?", "-c:a", audio_codec]

        command += ["-shortest"]
        if output_format:
            command += ["-f", output_format]
        command.append(out_path)
        self.command = command
        self.proc = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=None)

    def write_frame(self, data: bytes) -> None:
        if self.proc.stdin is None or self.proc.stdin.closed:
            raise RuntimeError("FFmpeg encoder input is closed")
        if len(data) != self.frame_size:
            raise ValueError(f"writer expected {self.frame_size} bytes, received {len(data)}")
        self.proc.stdin.write(data)

    def close(self) -> None:
        close_error = None
        if self.proc.stdin and not self.proc.stdin.closed:
            try:
                self.proc.stdin.close()
            except BrokenPipeError as error:
                close_error = error
        return_code = self.proc.wait()
        if return_code not in (0, None):
            raise RuntimeError(
                f"FFmpeg encoder exited with code {return_code}; output may be incomplete"
            ) from close_error
        if close_error is not None:
            raise close_error

    @property
    def returncode(self) -> Optional[int]:
        return self.proc.returncode
