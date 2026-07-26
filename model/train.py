# Real-data loading (_load_dataloaders), dropout_rate/weight_decay/batch-norm
# CLI wiring, checkpoint saving (best val_loss) in train(), and the
# experiments/ tracking system (_new_experiment_dir, _write_config,
# _write_notes_template) written with assistance from Claude Code (Sonnet 5)

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from model.cnn import GenreCNN
from model.plot_training import plot_training_curves

# Constants: logged once per epoch to CSV.
CSV_FIELDS = ["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy"]

# Where build_dataset.py writes train.npz/val.npz/test.npz by default.
DEFAULT_DATA_DIR = Path("data/processed")

# Each run under __main__ gets its own experiments/<run_id>/ folder
# config.json + notes.md so past runs stay comparable instead of overwriting
# training_log.csv from run to run.
EXPERIMENTS_DIR = Path("experiments")

NOTES_TEMPLATE = """# Experiment notes

## Hypothesis
What are you trying in this run, and why?

## Result summary
Fill in after training: final train/val loss & accuracy, and how the curve looked
(see curves.png).

## Interpretation
Why do you think it turned out this way?

## Next experiment
What will you try next, and why?
"""


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """
    Runs one training pass over dataloader. Returns a tuple containing
    the average loss and accuracy.
    """
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
    """
    Run a single no-grad evaulation pass returning tuple (average loss and accuracy)
    weights unchanged.
    """
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
    weight_decay: float = 0.0,
    checkpoint_path: str | Path | None = None,
) -> Path:
    """
    Train - logs loss and accuracy to CSV per epoch. If checkpoint_path is
    given, saves the model's weights there whenever val_loss reaches a new
    best -- not just whatever epoch training happens to end on, since the
    best epoch is often a few epochs before the last one.
    """
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.to(device)

    best_val_loss = float("inf")

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

            # Matches the {"model_state_dict": ...} wrapper cli/inference.py
            # already expects (torch.load(...)["model_state_dict"]).
            if checkpoint_path is not None and val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)

            print(
                f"epoch {epoch}/{epochs}  "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            )

    return csv_path


def _load_dataloaders(
    data_dir: str | Path, batch_size: int = 32
) -> tuple[DataLoader, DataLoader]:
    """
    Loads the train/val splits written by build_dataset.py (see
    docs/tensor-contract.md) into DataLoaders. X is already segmented and
    per-band normalized. This wraps the arrays, no further processing.
    """
    data_dir = Path(data_dir)

    def _dataset(split: str) -> TensorDataset:
        with np.load(data_dir / f"{split}.npz") as npz:
            X = torch.from_numpy(npz["X"])
            y = torch.from_numpy(npz["y"])
        return TensorDataset(X, y)

    train_loader = DataLoader(_dataset("train"), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(_dataset("val"), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def _new_experiment_dir(label: str = "") -> Path:
    """Creates experiments/<timestamp>[_label]/ and returns its path."""
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    if label:
        run_id = f"{run_id}_{label}"
    run_dir = EXPERIMENTS_DIR / run_id
    run_dir.mkdir(parents=True)
    return run_dir


def _write_config(run_dir: Path, args: argparse.Namespace, model: nn.Module) -> None:
    """Records the hyperparameters and model architecture used for this run,
    so past experiments/ runs stay comparable instead of relying on memory."""
    config = {
        "label": args.label,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "dropout_rate": args.dropout_rate,
        "weight_decay": args.weight_decay,
        "use_batchnorm": args.batch_norm,
        "data_dir": str(args.data_dir),
        "model_architecture": str(model),
        "n_params": sum(p.numel() for p in model.parameters()),
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)


def _write_notes_template(run_dir: Path) -> None:
    (run_dir / "notes.md").write_text(NOTES_TEMPLATE)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train GenreCNN on the dataset written by build_dataset.py."
    )
    parser.add_argument(
        "--data-dir", type=Path, default=DEFAULT_DATA_DIR,
        help="Directory with train.npz/val.npz from build_dataset.py (default: %(default)s)",
    )
    parser.add_argument(
        "--epochs", type=int, default=10,
        help="Number of training epochs (default: %(default)s)",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="Adam learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Training/validation batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--dropout-rate", type=float, default=0.5,
        help="Dropout probability before the final Linear (default: %(default)s)",
    )
    parser.add_argument(
        "--weight-decay", type=float, default=0.0,
        help="L2 weight decay for the Adam optimizer (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-norm", action="store_true",
        help="Add BatchNorm2d after each conv layer (default: off)",
    )
    parser.add_argument(
        "--label", type=str, default="",
        help="Short name appended to this run's experiments/ folder, "
             "e.g. --label baseline (default: none, just a timestamp)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GenreCNN(dropout_rate=args.dropout_rate, use_batchnorm=args.batch_norm)
    train_loader, val_loader = _load_dataloaders(args.data_dir, batch_size=args.batch_size)

    run_dir = _new_experiment_dir(args.label)
    _write_config(run_dir, args, model)

    csv_path = train(
        model,
        train_loader,
        val_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        csv_path=run_dir / "training_log.csv",
        weight_decay=args.weight_decay,
        checkpoint_path=run_dir / "checkpoint.pt",
    )

    plot_training_curves(csv_path, run_dir / "curves.png")
    _write_notes_template(run_dir)

    print(f"Experiment logged to {run_dir}/ - fill in notes.md with your findings.")
