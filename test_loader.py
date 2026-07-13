import logging

import pytest

from loader import discover_audio, load_dataset, validate_audio


def _make_genre_tree(root, write_wav):
    """genreA: 2 valid wavs (one uppercase .WAV). genreB: 1 valid wav +
    1 corrupt wav + 1 stray non-wav file. genreC: empty folder."""
    genre_a = root / "genreA"
    genre_a.mkdir(parents=True)
    write_wav(genre_a / "genreA.00000.wav")
    write_wav(genre_a / "genreA.00001.WAV")

    genre_b = root / "genreB"
    genre_b.mkdir(parents=True)
    write_wav(genre_b / "genreB.00000.wav")
    (genre_b / "genreB.00001.wav").write_bytes(b"not actually a wav file")
    (genre_b / "notes.txt").write_text("ignore me")

    genre_c = root / "genreC"
    genre_c.mkdir(parents=True)


def test_discover_audio_labels_by_lowercased_folder(tmp_path, write_sine_wav):
    _make_genre_tree(tmp_path, write_sine_wav)

    pairs = discover_audio(tmp_path)

    genres = {genre for _, genre in pairs}
    assert genres == {"genrea", "genreb"}
    assert len(pairs) == 4


def test_discover_audio_matches_uppercase_extension(tmp_path, write_sine_wav):
    _make_genre_tree(tmp_path, write_sine_wav)

    pairs = discover_audio(tmp_path)

    names = {path.name for path, _ in pairs}
    assert "genreA.00001.WAV" in names


def test_discover_audio_ignores_non_wav_files(tmp_path, write_sine_wav):
    _make_genre_tree(tmp_path, write_sine_wav)

    pairs = discover_audio(tmp_path)

    names = {path.name for path, _ in pairs}
    assert "notes.txt" not in names


def test_discover_audio_warns_on_empty_folder(tmp_path, write_sine_wav, caplog):
    _make_genre_tree(tmp_path, write_sine_wav)

    with caplog.at_level(logging.WARNING):
        pairs = discover_audio(tmp_path)

    genres = {genre for _, genre in pairs}
    assert "genrec" not in genres
    assert any("genreC" in record.message for record in caplog.records)


def test_discover_audio_missing_root_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="Audio root not found"):
        discover_audio(tmp_path / "does_not_exist")


def test_validate_audio_true_for_valid_file(tmp_path, write_sine_wav):
    wav_path = tmp_path / "good.wav"
    write_sine_wav(wav_path)

    assert validate_audio(wav_path) is True


def test_validate_audio_false_for_corrupt_file(tmp_path):
    bad_path = tmp_path / "bad.wav"
    bad_path.write_bytes(b"not actually a wav file")

    assert validate_audio(bad_path) is False


def test_load_dataset_summary_counts(tmp_path, write_sine_wav):
    _make_genre_tree(tmp_path, write_sine_wav)

    valid_pairs, summary = load_dataset(tmp_path)

    assert len(valid_pairs) == 3
    assert summary["genrea"] == {"discovered": 2, "valid": 2, "skipped": 0}
    assert summary["genreb"] == {"discovered": 2, "valid": 1, "skipped": 1}


def test_load_dataset_empty_genre_has_zero_counts(tmp_path, write_sine_wav):
    _make_genre_tree(tmp_path, write_sine_wav)

    _, summary = load_dataset(tmp_path)

    assert summary["genrec"] == {"discovered": 0, "valid": 0, "skipped": 0}
