# Written with assistance from Claude Code (Sonnet 5)

import csv

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from model.cnn import GenreCNN
from model.train import CSV_FIELDS, train, train_one_epoch, validate_one_epoch


def _make_loader(n, batch_size, shuffle=False):
    x = torch.randn(n, 1, 128, 130, dtype=torch.float32)
    y = torch.randint(0, 10, (n,))
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=shuffle)


def test_train_one_epoch_updates_weights():
    model = GenreCNN()
    loader = _make_loader(n=16, batch_size=8)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    before = [p.clone() for p in model.parameters()]
    train_one_epoch(model, loader, loss_fn, optimizer, torch.device("cpu"))
    after = list(model.parameters())

    assert any(not torch.equal(b, a) for b, a in zip(before, after))


def test_validate_one_epoch_does_not_update_weights():
    model = GenreCNN()
    loader = _make_loader(n=16, batch_size=8)
    loss_fn = nn.CrossEntropyLoss()

    before = [p.clone() for p in model.parameters()]
    validate_one_epoch(model, loader, loss_fn, torch.device("cpu"))
    after = list(model.parameters())

    assert all(torch.equal(b, a) for b, a in zip(before, after))


def test_validate_one_epoch_runs_without_grad():
    model = GenreCNN()
    loader = _make_loader(n=8, batch_size=4)
    loss_fn = nn.CrossEntropyLoss()

    validate_one_epoch(model, loader, loss_fn, torch.device("cpu"))
    assert not model.training


def test_uneven_final_batch():
    # 10 samples, batch_size=4 -> batches of 4, 4, 2
    model = GenreCNN()
    loader = _make_loader(n=10, batch_size=4)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    loss, acc = train_one_epoch(model, loader, loss_fn, optimizer, torch.device("cpu"))
    assert loss == loss  # not NaN
    assert 0.0 <= acc <= 1.0


def test_batch_size_one():
    # GenreCNN has no BatchNorm layers, so a batch of 1 should not error.
    model = GenreCNN()
    loader = _make_loader(n=3, batch_size=1)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_one_epoch(model, loader, loss_fn, optimizer, torch.device("cpu"))
    validate_one_epoch(model, loader, loss_fn, torch.device("cpu"))


def test_train_writes_expected_csv_columns(tmp_path):
    model = GenreCNN()
    train_loader = _make_loader(n=12, batch_size=4)
    val_loader = _make_loader(n=8, batch_size=4)
    csv_path = tmp_path / "training_log.csv"

    train(
        model,
        train_loader,
        val_loader,
        epochs=2,
        lr=1e-3,
        device=torch.device("cpu"),
        csv_path=csv_path,
    )

    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    assert rows[0].keys() == set(CSV_FIELDS)
    assert len(rows) == 2
    assert [r["epoch"] for r in rows] == ["1", "2"]
