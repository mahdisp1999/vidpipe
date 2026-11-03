import torch
import torch.nn.functional as functional

import vidpipe.stages.model as model_module
from vidpipe.stages.model import ModelStage


class IdentityModel:
    """Small CPU model double: return every tile without changing it."""

    def __init__(self):
        self.batch_sizes = []

    def __call__(self, batch):
        self.batch_sizes.append(batch.shape[0])
        return batch


def test_equal_tiles_are_processed_in_batches_and_reassembled(monkeypatch):
    # Spandrel is an optional runtime dependency in this test environment.
    monkeypatch.setattr(model_module, "torch", torch)
    monkeypatch.setattr(model_module, "F", functional)
    stage = ModelStage()
    stage.model = IdentityModel()
    stage.scale = 1
    stage.tile_size = 4
    stage.tile_pad = 1
    stage.device = "cpu"

    frame = torch.rand(1, 3, 8, 8)
    output = stage._run_tiled_once(frame, tile_size=4, batch_size=2)

    assert torch.equal(output, frame)
    assert stage.model.batch_sizes == [2, 2]
    assert stage.tiles_completed == 4
    assert stage.tiles_total == 4


def test_cuda_oom_reduces_batch_before_tile_size(monkeypatch):
    monkeypatch.setattr(model_module, "torch", torch)
    stage = ModelStage()
    stage.device = "cuda"
    stage.batch_size = 4
    stage.tile_size = 128
    calls = []

    def run_once(_tensor, tile_size, batch_size):
        calls.append((tile_size, batch_size))
        if batch_size > 1:
            raise RuntimeError("CUDA out of memory")
        return "finished"

    stage._run_tiled_once = run_once
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    result = stage._run_tiled(torch.zeros(1, 3, 8, 8))

    assert result == "finished"
    assert calls == [(128, 4), (128, 2), (128, 1)]
    assert stage.batch_size == 1
    assert stage.tile_size == 128
    assert stage.oom_retries == 2


def test_cuda_oom_reduces_tile_size_after_batch_reaches_one(monkeypatch):
    monkeypatch.setattr(model_module, "torch", torch)
    stage = ModelStage()
    stage.device = "cuda"
    stage.batch_size = 1
    stage.tile_size = 128
    calls = []

    def run_once(_tensor, tile_size, batch_size):
        calls.append((tile_size, batch_size))
        if tile_size > 64:
            raise RuntimeError("CUDA out of memory")
        return "finished"

    stage._run_tiled_once = run_once
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    result = stage._run_tiled(torch.zeros(1, 3, 8, 8))

    assert result == "finished"
    assert calls == [(128, 1), (64, 1)]
    assert stage.tile_size == 64
    assert stage.oom_retries == 1
