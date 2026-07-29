import pytest
from cli.validation import validate_audio_file


def test_file_error_raises(tmp_path):

    path = tmp_path / "non_existant.wav"

    with pytest.raises(FileNotFoundError) as excinfo:
        validate_audio_file(str(path))

    assert str(excinfo.value) == f"The file {path} does not exist."


def test_file_exists(tmp_path, write_sine_wav):

    path = tmp_path / "audio_file.wav"
    write_sine_wav(path)
    validate_audio_file(str(path))


def test_file_extension_error(tmp_path):

    path = tmp_path / "audio_file.txt"
    path.write_text("not audio")
    with pytest.raises(ValueError) as excinfo:
        validate_audio_file(str(path))

    assert ".txt" in str(excinfo.value)


def test_librosa_fails(tmp_path):

    path = tmp_path / "audio_file.wav"
    path.write_bytes(b"I am bytes")
    with pytest.raises(ValueError) as excinfo:
        validate_audio_file(str(path))

    assert "not be loaded by librosa" in str(excinfo.value)
