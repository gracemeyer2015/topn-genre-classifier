# Real-data loading, batch-norm/weight-decay CLI wiring, checkpoint saving,
# early stopping, and the experiments/ tracking system written with
# assistance from Claude Code (Sonnet 5).

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

# Each run gets its own experiments/<run_id>/ folder, so runs stay
# comparable instead of overwriting one shared log.
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
    Run one training pass over dataloader, updating model weights.

    Args:
        model (nn.Module): Updated in place via optimizer.step().
        dataloader (DataLoader): Yields (inputs, labels) batches.
        loss_fn (nn.Module): e.g. nn.CrossEntropyLoss().
        optimizer (torch.optim.Optimizer): Wraps model's parameters.
        device (torch.device): Device to train on.

    Returns:
        tuple: (average_loss, accuracy) for this epoch.
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
    Run one no-grad evaluation pass, leaving model weights unchanged.

    Args:
        model (nn.Module): Not updated by this function.
        dataloader (DataLoader): Yields (inputs, labels) batches.
        loss_fn (nn.Module): e.g. nn.CrossEntropyLoss().
        device (torch.device): Device to evaluate on.

    Returns:
        tuple: (average_loss, accuracy).
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
    patience: int | None = None,
) -> Path:
    """
    Run the full training loop, logging loss and accuracy to CSV per
    epoch.

    If checkpoint_path is given, saves weights on each new best val_loss,
    not just whatever epoch training ends on. If patience is given, stops
    early once val_loss hasn't improved for that many epochs, since long
    runs otherwise keep training well past their peak for no benefit.

    Args:
        model (nn.Module): Trained in place.
        train_loader (DataLoader): Training batches.
        val_loader (DataLoader): Validation batches, evaluated once per
            epoch.
        epochs (int): Number of epochs to run.
        lr (float): Adam learning rate.
        device (torch.device): Device to train on.
        csv_path (str | Path): Where to write the per-epoch metrics CSV.
        weight_decay (float): L2 weight decay for Adam. Defaults to 0.0.
        checkpoint_path (str | Path | None): Where to save the
            best-val_loss checkpoint. Defaults to None (disabled).
        patience (int | None): Stop early after this many epochs without
            improvement. Defaults to None (disabled).

    Returns:
        Path: The csv_path written to.
    """
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.to(device)

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_since_improvement = 0

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

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                epochs_since_improvement = 0
                # Matches the {"model_state_dict": ...} wrapper cli/inference.py expects.
                if checkpoint_path is not None:
                    torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)
            else:
                epochs_since_improvement += 1

            print(
                f"epoch {epoch}/{epochs}  "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            )

            if patience is not None and epochs_since_improvement >= patience:
                print(
                    f"No improvement in val_loss for {patience} epochs "
                    f"(best was epoch {best_epoch}, val_loss={best_val_loss:.4f}). "
                    f"Stopping early at epoch {epoch}/{epochs}."
                )
                break

    return csv_path


def _load_dataloaders(
    data_dir: str | Path, batch_size: int = 32
) -> tuple[DataLoader, DataLoader]:
    """
    Load the train/val splits from build_dataset.py into DataLoaders.

    X is already segmented and per-band normalized; this just wraps the
    arrays.

    Args:
        data_dir (str | Path): Directory with train.npz/val.npz.
        batch_size (int): Batch size for both loaders. Defaults to 32.

    Returns:
        tuple: (train_loader, val_loader).
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
    """
    Create experiments/<timestamp>[_label]/ and return its path.

    Args:
        label (str): Optional suffix for the folder name. Defaults to "".

    Returns:
        Path: The newly created run directory.
    """
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    if label:
        run_id = f"{run_id}_{label}"
    run_dir = EXPERIMENTS_DIR / run_id
    run_dir.mkdir(parents=True)
    return run_dir


def _write_config(run_dir: Path, args: argparse.Namespace, model: nn.Module) -> None:
    """
    Record this run's hyperparameters and architecture, so past runs stay
    comparable instead of relying on memory.

    Args:
        run_dir (Path): The experiments/<run>/ directory to write into.
        args (argparse.Namespace): Parsed CLI arguments for this run.
        model (nn.Module): Used to record its architecture and param count.

    Returns:
        None
    """
    config = {
        "label": args.label,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "dropout_rate": args.dropout_rate,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "use_batchnorm": args.batch_norm,
        "data_dir": str(args.data_dir),
        "model_architecture": str(model),
        "n_params": sum(p.numel() for p in model.parameters()),
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)


def _write_notes_template(run_dir: Path) -> None:
    """
    Write the blank notes.md template into run_dir.

    Args:
        run_dir (Path): The experiments/<run>/ directory to write into.

    Returns:
        None
    """
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
        "--patience", type=int, default=None,
        help="Stop early if val_loss hasn't improved for this many epochs "
             "in a row (default: disabled, always runs the full --epochs)",
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
        patience=args.patience,
    )

    plot_training_curves(csv_path, run_dir / "curves.png")
    _write_notes_template(run_dir)

    print(f"Experiment logged to {run_dir}/. Fill in notes.md with your findings.")
