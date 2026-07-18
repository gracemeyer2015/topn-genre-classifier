import csv
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from model.cnn import GenreCNN

# Constants: logged once per epoch to CSV.
CSV_FIELDS = ["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy"]


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(inputs)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += inputs.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        logits = model(inputs)
        loss = loss_fn(logits, labels)

        total_loss += loss.item() * inputs.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += inputs.size(0)

    return total_loss / total, correct / total


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
    csv_path: str | Path,
) -> Path:
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)

    csv_path = Path(csv_path)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, loss_fn, optimizer, device
            )
            val_loss, val_acc = validate_one_epoch(
                model, val_loader, loss_fn, device
            )

            writer.writerow({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
            })
            f.flush()

            print(
                f"epoch {epoch}/{epochs}  "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            )

    return csv_path


def _make_dummy_loaders(
    n_train: int = 30, n_val: int = 16, batch_size: int = 8
) -> tuple[DataLoader, DataLoader]:
    """Stand in: Random tensors matching the tensor contract"""
    train_x = torch.randn(n_train, 1, 128, 130, dtype=torch.float32)
    train_y = torch.randint(0, 10, (n_train,))
    val_x = torch.randn(n_val, 1, 128, 130, dtype=torch.float32)
    val_y = torch.randint(0, 10, (n_val,))

    train_loader = DataLoader(
        TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(val_x, val_y), batch_size=batch_size, shuffle=False
    )
    return train_loader, val_loader


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GenreCNN()
    train_loader, val_loader = _make_dummy_loaders()
    train(
        model,
        train_loader,
        val_loader,
        epochs=3,
        lr=1e-3,
        device=device,
        csv_path="training_log.csv",
    )
