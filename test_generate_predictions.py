import pytest
import numpy as np
from cli.dummyModel import DummyModel
from evaluate.generate_predictions import generate_predictions


def make_fake_test_npz(tmp_path, n_samples=4):
    """Small fake .npz data matching tensor contract"""
    fake_X = np.random.randn(n_samples, 1,128,130).astype(np.float)
    fake_y = np.array(list(range(n_samples)))
    npz_path = tmp_path / "fake_test.npz"
    np.savez(npz_path, X=fake_X, y=fake_y)

    return str(npz_path)


def generate_predctions_returns_correct_length(tmp_path):
    """Is the full batch of given batch size being processed no skipped samples"""
    npz_path = make_fake_test_npz(tmp_path, n_samples=4)
    model = DummyModel()

    predicted, true = generate_predictions(model, npz_path)

    assert len(predicted) == 4
    assert len(true) == 4


def test_generate_predictions_return_strings(tmp_path):
    """makes sure that generate_predictions returns data type string """
    npz_path = make_fake_test_npz(tmp_path, n_samples=4)
    model = DummyModel()

    predicted, true = generate_predictions(model, npz_path)

    assert all(isinstance(g,str) for g in predicted)
    assert all(isinstance(g,str) for g in true)


def generate_predictions_maps_true_labels_correctly(tmp_path):
    """checks whether true labels are mapped to correctly"""
    npz_path = make_fake_test_npz(tmp_path, n_samples=4)
    model = DummyModel()

    predicted, true = generate_predictions(model, npz_path)

    assert true == ["blues", "classical", "country", "disco"]
    