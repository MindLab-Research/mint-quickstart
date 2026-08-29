from quickstart.example_utils import clamp, parse_ports, format_duration


def test_clamp():
    assert clamp(5, 0, 10) == 5
    assert clamp(-1, 0, 10) == 0


def test_parse_ports():
    assert parse_ports("8000,8001-8003") == [8000, 8001, 8002, 8003]


def test_format_duration():
    assert format_duration(3700) == "1h 1m 40s"
