import pytest
from cli.predict import parse_arguments


def test_parse_args_default_topn(monkeypatch):
    monkeypatch.setattr("sys.argv", ["predict.py", "song.wav"])

    args = parse_arguments()

    assert args.audio_file == "song.wav"
    assert args.top_n == 5


def test_missing_arg(monkeypatch):
    monkeypatch.setattr("sys.argv", ["predict.py"])

    with pytest.raises(SystemExit):
        parse_arguments()


def test_custom_topn(monkeypatch):
    monkeypatch.setattr("sys.argv", ["predict.py", "song.wav", "--top_n", "3"])

    args = parse_arguments()
    assert args.top_n == 3


def test_shorthandn(monkeypatch):
    monkeypatch.setattr("sys.argv", ["predict.py", "song.wav", "-n", "3"])

    args = parse_arguments()
    assert args.top_n == 3


def test_wrong_type_n(monkeypatch):
    monkeypatch.setattr("sys.argv", ["predict.py", "song.wav", "-n", "a"])

    with pytest.raises(SystemExit):
        parse_arguments()
