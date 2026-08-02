import json

import numpy as np
import pytest
import torch
from torch import nn

import evaluate.song_level as song_level_module
from evaluate.song_level import (
    aggregate_song_probabilities,
    build_model_from_config,
    run_model_on_split,
    segment_level_accuracy,
    song_level_accuracy,
    song_level_true_labels,
)


def test_aggregate_song_probabilities_averages_within_each_song():
    # 2 songs x 2 segments each, 3 classes.
    probabilities = np.array([
        [0.9, 0.05, 0.05],  # song 0, segment 0
        [0.7, 0.20, 0.10],  # song 0, segment 1
        [0.1, 0.10, 0.80],  # song 1, segment 0
        [0.2, 0.30, 0.50],  # song 1, segment 1
    ])
    song_ids = np.array([0, 0, 1, 1])

    unique_ids, avg = aggregate_song_probabilities(probabilities, song_ids)

    assert unique_ids.tolist() == [0, 1]
    np.testing.assert_allclose(avg[0], [0.80, 0.125, 0.075])
    np.testing.assert_allclose(avg[1], [0.15, 0.20, 0.65])


def test_aggregate_song_probabilities_handles_nondense_gapped_ids():
    """song_id can skip values entirely (a too-short song produced zero
    segments) -- grouping must work by actual unique values present, not by
    iterating range(max_id + 1)."""
    probabilities = np.array([
        [1.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 1.0],
    ])
    song_ids = np.array([0, 0, 5, 5])  # ids 1-4 never appear

    unique_ids, avg = aggregate_song_probabilities(probabilities, song_ids)

    assert unique_ids.tolist() == [0, 5]
    np.testing.assert_allclose(avg[0], [1.0, 0.0])
    np.testing.assert_allclose(avg[1], [0.0, 1.0])


def test_aggregate_song_probabilities_raises_on_length_mismatch():
    with pytest.raises(ValueError, match="must describe the same segments"):
        aggregate_song_probabilities(
            probabilities=np.zeros((3, 2)), song_ids=np.array([0, 0])
        )


def test_aggregate_song_probabilities_handles_empty_input():
    unique_ids, avg = aggregate_song_probabilities(
        probabilities=np.zeros((0, 3)), song_ids=np.array([], dtype=np.int64)
    )
    assert unique_ids.shape == (0,)
    assert avg.shape == (0, 3)


def test_song_level_true_labels_agrees_within_song():
    y = np.array([2, 2, 7, 7, 7])
    song_ids = np.array([0, 0, 1, 1, 1])

    unique_ids, labels = song_level_true_labels(y, song_ids)

    assert unique_ids.tolist() == [0, 1]
    assert labels.tolist() == [2, 7]


def test_song_level_true_labels_raises_on_length_mismatch():
    with pytest.raises(ValueError, match="must describe the same segments"):
        song_level_true_labels(y=np.array([0, 1, 2]), song_ids=np.array([0, 0]))


def test_song_level_true_labels_raises_on_mixed_genre_within_one_song():
    """A deliberately-broken fixture: song_id=0's segments disagree on y.
    This should never happen from a correctly-built dataset, but the
    assertion must actually fire rather than silently picking one label."""
    y = np.array([2, 9])  # song 0's two segments disagree
    song_ids = np.array([0, 0])

    with pytest.raises(ValueError, match="disagreeing on genre"):
        song_level_true_labels(y, song_ids)


def test_song_level_accuracy_lifts_a_song_whose_segments_split_the_vote():
    """Segment-level: song 0 gets one segment right, one wrong -> counts as
    half-wrong at the segment level. Song-level: averaging both segments'
    probabilities still favors the true class, so aggregation should fix
    the flipped segment and raise accuracy relative to what segment-level
    scoring alone would show for this song."""
    probabilities = np.array([
        [0.9, 0.1],  # song 0 segment 0: correctly predicts class 0
        [0.4, 0.6],  # song 0 segment 1: wrongly predicts class 1
        [0.2, 0.8],  # song 1 segment 0: correctly predicts class 1
        [0.1, 0.9],  # song 1 segment 1: correctly predicts class 1
    ])
    y = np.array([0, 0, 1, 1])
    song_ids = np.array([0, 0, 1, 1])

    seg_acc = segment_level_accuracy(probabilities, y)
    song_acc, song_ids_used = song_level_accuracy(probabilities, y, song_ids)

    assert seg_acc == 0.75  # 3/4 segments correct
    assert song_ids_used.tolist() == [0, 1]
    assert song_acc == 1.0  # both songs correct once aggregated


class _TinyModel(nn.Module):
    """Minimal stand-in for GenreCNN: same (N,1,128,130)->(N,10) contract,
    but tiny enough to construct/run instantly in a test."""

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(1 * 128 * 130, 10)

    def forward(self, x):
        return self.linear(self.flatten(x))


def test_run_model_on_split_raises_clearly_when_song_id_missing(tmp_path):
    """An .npz built before song_id was added should fail with a clear,
    actionable message, not a bare KeyError from inside np.load indexing."""
    npz_path = tmp_path / "val.npz"
    np.savez(
        npz_path,
        X=np.zeros((2, 1, 128, 130), dtype=np.float32),
        y=np.array([0, 1], dtype=np.int64),
    )

    with pytest.raises(ValueError, match="no 'song_id' key"):
        run_model_on_split(_TinyModel(), npz_path)


def test_run_model_on_split_handles_empty_split(tmp_path):
    npz_path = tmp_path / "val.npz"
    np.savez(
        npz_path,
        X=np.zeros((0, 1, 128, 130), dtype=np.float32),
        y=np.array([], dtype=np.int64),
        song_id=np.array([], dtype=np.int64),
    )

    probabilities, y, song_id = run_model_on_split(_TinyModel(), npz_path)

    assert probabilities.shape == (0, 10)
    assert y.shape == (0,)
    assert song_id.shape == (0,)


def test_run_model_on_split_returns_probabilities_that_sum_to_one(tmp_path):
    npz_path = tmp_path / "val.npz"
    X = np.random.randn(5, 1, 128, 130).astype(np.float32)
    np.savez(
        npz_path,
        X=X,
        y=np.array([0, 1, 2, 0, 1], dtype=np.int64),
        song_id=np.array([0, 0, 1, 1, 2], dtype=np.int64),
    )

    probabilities, y, song_id = run_model_on_split(_TinyModel(), npz_path, batch_size=2)

    assert probabilities.shape == (5, 10)
    np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(5), rtol=1e-5)
    assert y.tolist() == [0, 1, 2, 0, 1]
    assert song_id.tolist() == [0, 0, 1, 1, 2]


def test_run_model_on_split_matches_direct_model_call(tmp_path):
    """Batching (batch_size=2 over 5 rows) shouldn't change the result
    versus running the whole array through the model at once."""
    torch.manual_seed(0)
    npz_path = tmp_path / "val.npz"
    X = np.random.randn(5, 1, 128, 130).astype(np.float32)
    np.savez(
        npz_path,
        X=X,
        y=np.array([0, 1, 2, 0, 1], dtype=np.int64),
        song_id=np.array([0, 0, 1, 1, 2], dtype=np.int64),
    )
    model = _TinyModel()
    model.eval()

    probabilities, _y, _song_id = run_model_on_split(model, npz_path, batch_size=2)

    with torch.no_grad():
        expected = torch.softmax(model(torch.from_numpy(X)), dim=1).numpy()
    np.testing.assert_allclose(probabilities, expected, rtol=1e-5, atol=1e-6)


def test_run_model_on_split_raises_on_nonpositive_batch_size(tmp_path):
    npz_path = tmp_path / "val.npz"
    np.savez(
        npz_path,
        X=np.zeros((2, 1, 128, 130), dtype=np.float32),
        y=np.array([0, 1], dtype=np.int64),
        song_id=np.array([0, 1], dtype=np.int64),
    )

    with pytest.raises(ValueError, match="batch_size must be positive"):
        run_model_on_split(_TinyModel(), npz_path, batch_size=0)


def test_song_level_accuracy_raises_on_zero_songs():
    with pytest.raises(ValueError, match="zero songs"):
        song_level_accuracy(
            probabilities=np.zeros((0, 3)),
            y=np.array([], dtype=np.int64),
            song_ids=np.array([], dtype=np.int64),
        )


def test_segment_level_accuracy_raises_on_zero_segments():
    with pytest.raises(ValueError, match="zero segments"):
        segment_level_accuracy(probabilities=np.zeros((0, 3)), y=np.array([], dtype=np.int64))


class _NoArgGenreCNN(nn.Module):
    """Stand-in for main's GenreCNN() -- constructor takes no config kwargs."""

    def __init__(self):
        super().__init__()


class _ConfigurableGenreCNN(nn.Module):
    """Stand-in for train-real-pipeline-data's GenreCNN(dropout_rate,
    use_batchnorm) -- records what it was constructed with so the test can
    assert build_model_from_config actually passed the config values
    through, not just that construction didn't raise."""

    def __init__(self, dropout_rate=0.5, use_batchnorm=False):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.use_batchnorm = use_batchnorm


def _write_config(tmp_path, **kwargs):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"epochs": 200, "lr": 0.001, **kwargs}))
    return config_path


def test_build_model_from_config_omits_unaccepted_kwargs(tmp_path, monkeypatch):
    """Against a no-arg constructor (main's current GenreCNN), config keys
    the constructor doesn't accept must be silently dropped, not passed
    through and raise a TypeError."""
    monkeypatch.setattr(song_level_module, "GenreCNN", _NoArgGenreCNN)
    config_path = _write_config(tmp_path, dropout_rate=0.25, use_batchnorm=True)

    model = build_model_from_config(config_path)

    assert isinstance(model, _NoArgGenreCNN)


def test_build_model_from_config_passes_through_accepted_kwargs(tmp_path, monkeypatch):
    """Against a constructor that does accept dropout_rate/use_batchnorm
    (train-real-pipeline-data's GenreCNN), those config values must actually
    reach the constructor, not just avoid raising."""
    monkeypatch.setattr(song_level_module, "GenreCNN", _ConfigurableGenreCNN)
    config_path = _write_config(tmp_path, dropout_rate=0.25, use_batchnorm=True)

    model = build_model_from_config(config_path)

    assert isinstance(model, _ConfigurableGenreCNN)
    assert model.dropout_rate == 0.25
    assert model.use_batchnorm is True
