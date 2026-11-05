import pytest

from vidpipe.cut import build_keep_ranges, normalize_remove_ranges, parse_timestamp


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("75", 75.0),
        ("01:15", 75.0),
        ("00:01:15", 75.0),
        ("01:02:03.5", 3723.5),
    ],
)
def test_parse_timestamp(value, expected):
    assert parse_timestamp(value) == expected


@pytest.mark.parametrize("value", ["", "1:70", "1:2:70", "-1", "a:b"])
def test_parse_timestamp_rejects_invalid_values(value):
    with pytest.raises(ValueError):
        parse_timestamp(value)


def test_normalize_sorts_and_merges_overlapping_ranges():
    result = normalize_remove_ranges(
        ["00:40:00-00:50:00", "00:10:00-00:15:00", "00:14:00-00:18:00"],
        duration=7200,
    )
    assert result == [(600.0, 1080.0), (2400.0, 3000.0)]


def test_normalize_rejects_ranges_outside_duration():
    with pytest.raises(ValueError, match="exceeds video duration"):
        normalize_remove_ranges(["00:09:00-00:11:00"], duration=600)


def test_build_keep_ranges_for_multiple_removals():
    removals = [(600.0, 900.0), (2400.0, 3000.0), (4800.0, 5700.0)]
    assert build_keep_ranges(removals, 7200.0) == [
        (0.0, 600.0),
        (900.0, 2400.0),
        (3000.0, 4800.0),
        (5700.0, 7200.0),
    ]


def test_build_keep_ranges_rejects_removing_everything():
    with pytest.raises(ValueError, match="entire video"):
        build_keep_ranges([(0.0, 100.0)], 100.0)
