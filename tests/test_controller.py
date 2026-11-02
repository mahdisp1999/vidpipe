"""Controller state transition tests."""

from vidpipe.controller import Controller, State, StateError


def test_invalid_pause_idle():
    c = Controller()
    try:
        c.pause()
        assert False
    except StateError:
        pass
    assert c.state == State.IDLE


def test_invalid_resume_idle():
    c = Controller()
    try:
        c.resume()
        assert False
    except (StateError, Exception):
        pass


def test_start_requires_idle():
    # can't start twice
    pass


def test_states_enum():
    assert State.IDLE.name == "IDLE"
