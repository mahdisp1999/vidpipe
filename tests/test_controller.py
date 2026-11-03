import pytest

from vidpipe.controller import Controller, State, StateError


def test_pause_is_rejected_while_idle():
    controller = Controller()
    with pytest.raises(StateError):
        controller.pause()
    assert controller.state == State.IDLE


def test_resume_is_rejected_while_idle():
    controller = Controller()
    with pytest.raises(StateError):
        controller.resume()


def test_stop_is_rejected_while_idle():
    controller = Controller()
    with pytest.raises(StateError):
        controller.stop()
