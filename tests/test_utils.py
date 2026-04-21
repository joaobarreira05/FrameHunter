from framehunter.utils import format_timestamp


def test_format_timestamp_precision():
    assert format_timestamp(857.4) == "00:14:17.399"


def test_format_timestamp_non_negative():
    assert format_timestamp(-3.0) == "00:00:00.000"
