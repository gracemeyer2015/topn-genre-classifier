import json

import numpy as np
import pytest
import soundfile as sf

from build_dataset import (
    GENRES,
    apply_normalization,
    build_dataset,
    build_split_arrays,
    compute_band_stats,
)
from loader import load_dataset
from split import split_songs

SR = 22050
SONGS_PER_GENRE = 10  # -> 8/1/1 per genre at the default 80/10/10 split
SONG_DURATION_SEC = 6.0  # -> exactly 2 segments/song at segment_sec=3.0


def _write_varied_wav(path, frequency, duration_sec=SONG_DURATION_SEC, sr=SR):
    """A sine wav at a given frequency.

    Unlike conftest's write_sine_wav (always a fixed 440Hz), varying the
    frequency per file gives this synthetic dataset real inter-song mel-band
    variance. Without that, every segment would be a near-identical tone and
    per-band std would be ~0 everywhere, silently defeating the point of the
    normalization tests below (they'd all pass trivially against the
    NORM_EPSILON floor instead of against real computed statistics).
    """
    n_samples = int(duration_sec * sr)
    t = np.linspace(0, duration_sec, n_samples, endpoint=False)
    tone = (0.3 * np.sin(2 * np.pi * frequency * t)).astype(np.float32)
    sf.write(str(path), tone, sr)


def _make_genre_tree(root):
    """A tiny but real GTZAN-shaped tree: SONGS_PER_GENRE songs per genre,
    each at a different (deterministic, not random) frequency."""
    for genre_index, genre in enumerate(GENRES):
        genre_dir = root / genre
        genre_dir.mkdir(parents=True)
        for i in range(SONGS_PER_GENRE):
            frequency = 150 + (genre_index * 67 + i * 23) % 900
            _write_varied_wav(genre_dir / f"{genre}.{i:05d}.wav", frequency)


def test_build_split_arrays_shapes_and_dtypes(tmp_path):
    root = tmp_path / "genres"
    _make_genre_tree(root)
    pairs, _ = load_dataset(root)

    X, y = build_split_arrays(pairs, sr=SR, segment_sec=3.0)

    expected_segments = len(pairs) * 2  # 6s songs -> 2 segments each
    assert X.shape == (expected_segments, 1, 128, 130)
    assert X.dtype == np.float32
    assert y.shape == (expected_segments,)
    assert y.dtype == np.int64
    assert set(y.tolist()) == set(range(len(GENRES)))


def test_build_split_arrays_raises_on_unrecognized_genre(tmp_path):
    wav_path = tmp_path / "clip.wav"
    _write_varied_wav(wav_path, frequency=440)

    with pytest.raises(ValueError, match="Unrecognized genre"):
        build_split_arrays([(wav_path, "not_a_real_genre")])


def test_build_split_arrays_raises_on_empty_pairs():
    with pytest.raises(ValueError, match="empty"):
        build_split_arrays([])


def test_build_split_arrays_propagates_invalid_segment_sec(tmp_path):
    wav_path = tmp_path / "clip.wav"
    _write_varied_wav(wav_path, frequency=440)

    with pytest.raises(ValueError, match="segment_sec must be positive"):
        build_split_arrays([(wav_path, "blues")], segment_sec=0)


def test_build_split_arrays_propagates_invalid_sr(tmp_path):
    wav_path = tmp_path / "clip.wav"
    _write_varied_wav(wav_path, frequency=440)

    with pytest.raises(ValueError, match="sr must be positive"):
        build_split_arrays([(wav_path, "blues")], sr=0)


def test_build_split_arrays_raises_clearly_when_every_song_too_short(tmp_path):
    """A pairs list can be non-empty and every genre valid, yet still yield
    zero total segments if every song is shorter than segment_sec -- this
    must raise a clear error, not an opaque np.stack([]) crash."""
    wav_path = tmp_path / "clip.wav"
    _write_varied_wav(wav_path, frequency=440, duration_sec=1.0)  # < 3s

    with pytest.raises(ValueError, match="too short"):
        build_split_arrays([(wav_path, "blues")], sr=SR, segment_sec=3.0)


def test_compute_band_stats_matches_hand_computed_values(tmp_path):
    root = tmp_path / "genres"
    _make_genre_tree(root)
    pairs, _ = load_dataset(root)
    train_pairs, _val_pairs, _test_pairs = split_songs(pairs)
    X_train, _y_train = build_split_arrays(train_pairs, sr=SR, segment_sec=3.0)

    mean, std = compute_band_stats(X_train)

    hand_mean = X_train.mean(axis=(0, 1, 3))
    hand_std = X_train.std(axis=(0, 1, 3))
    np.testing.assert_allclose(mean, hand_mean, rtol=1e-4, atol=1e-4)
    # These varied-frequency tones give every band real variance, so the
    # NORM_EPSILON floor shouldn't engage here -- computed std should match
    # the hand-computed std directly, not the floor.
    np.testing.assert_allclose(std, hand_std, rtol=1e-4, atol=1e-4)


def test_apply_normalization_uses_given_stats_exactly(tmp_path):
    """Proves normalization uses the exact stats it's handed (i.e. train's),
    not stats recomputed from the array being normalized."""
    root = tmp_path / "genres"
    _make_genre_tree(root)
    pairs, _ = load_dataset(root)
    train_pairs, val_pairs, _test_pairs = split_songs(pairs)

    X_train, _y_train = build_split_arrays(train_pairs, sr=SR, segment_sec=3.0)
    X_val, _y_val = build_split_arrays(val_pairs, sr=SR, segment_sec=3.0)
    train_mean, train_std = compute_band_stats(X_train)

    X_val_original = X_val.copy()
    apply_normalization(X_val, train_mean, train_std)

    expected = (
        (X_val_original - train_mean[None, None, :, None])
        / train_std[None, None, :, None]
    )
    np.testing.assert_allclose(X_val, expected, rtol=1e-5, atol=1e-6)
    assert np.isfinite(X_val).all()


def test_build_dataset_writes_expected_artifacts(tmp_path):
    root = tmp_path / "genres"
    _make_genre_tree(root)
    output = tmp_path / "processed"

    splits, meta = build_dataset(root, output, sr=SR, segment_sec=3.0, seed=42)

    assert set(splits.keys()) == {"train", "val", "test"}
    for name in ("train", "val", "test"):
        with np.load(output / f"{name}.npz") as data:
            X, y = data["X"], data["y"]
            assert X.dtype == np.float32
            assert X.shape[1:] == (1, 128, 130)
            assert y.dtype == np.int64
            assert X.shape[0] == y.shape[0]
            assert np.isfinite(X).all()

    with np.load(output / "norm_stats.npz") as data:
        mean, std = data["mean"], data["std"]
        assert mean.shape == (128,)
        assert std.shape == (128,)
        assert mean.dtype == np.float32
        assert std.dtype == np.float32

    with open(output / "meta.json") as f:
        meta_on_disk = json.load(f)
    assert meta_on_disk == meta
    assert meta_on_disk["genres"] == list(GENRES)
    assert meta_on_disk["sr"] == SR
    assert meta_on_disk["segment_sec"] == 3.0

    # SONGS_PER_GENRE=10 * 10 genres = 100 songs, exact 80/10/10 -> 8/1/1 per genre
    assert meta_on_disk["split_sizes"]["songs"] == {"train": 80, "val": 10, "test": 10}
    # 6s songs -> 2 segments each
    assert meta_on_disk["split_sizes"]["segments"] == {
        "train": 160, "val": 20, "test": 20,
    }
