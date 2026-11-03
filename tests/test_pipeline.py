import threading

from vidpipe.pipeline import Pipeline
from vidpipe.stages.base import FrameSpec, Stage


class FakeReader:
    width = 1
    height = 1
    fps = 25.0
    frame_size = 3
    total_frames = 3
    exact_frame_count = True

    def __init__(self):
        self.frames = [b"one", b"two", b"tri"]

    def read_frame(self):
        return self.frames.pop(0) if self.frames else None


class FakeWriter:
    def __init__(self):
        self.frames = []
        self.closed = False

    def write_frame(self, data):
        self.frames.append(data)

    def close(self):
        self.closed = True


class PassStage(Stage):
    def load(self, device, params):
        pass

    def plan(self, source_spec):
        return source_spec

    def push(self, frame):
        return [frame]


def test_pipeline_reports_frames_and_stage_timings():
    reader = FakeReader()
    writer = FakeWriter()
    pause = threading.Event()
    pause.set()
    stop = threading.Event()
    pipeline = Pipeline(reader, writer, 2, pause, stop, stages=[PassStage()])
    pipeline.start()
    pipeline.join()

    snapshot = pipeline.status_snapshot()
    assert writer.closed
    assert writer.frames == [b"one", b"two", b"tri"]
    assert snapshot["frames_read"] == 3
    assert snapshot["processed"] == 3
    assert snapshot["total_frames"] == 3
    assert snapshot["stage_seconds"] >= 0
