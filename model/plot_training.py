"""Plot train/val loss and accuracy curves from a train.py training_log.csv.

Usage: python -m model.plot_training experiments/<run>/training_log.csv
"""

# Written with assistance from Claude Code (Sonnet 5)

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def plot_training_curves(csv_path: str | Path, output_path: str | Path) -> None:
    """Reads a train.py CSV log and saves a loss/accuracy-vs-epoch plot."""
    epochs, train_loss, train_acc, val_loss, val_acc = [], [], [], [], []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            epochs.append(int(row["epoch"]))
            train_loss.append(float(row["train_loss"]))
            train_acc.append(float(row["train_accuracy"]))
            val_loss.append(float(row["val_loss"]))
            val_acc.append(float(row["val_accuracy"]))

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(10, 4))

    ax_loss.plot(epochs, train_loss, marker="o", label="train")
    ax_loss.plot(epochs, val_loss, marker="o", label="val")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("loss")
    ax_loss.set_title("Loss")
    ax_loss.legend()

    ax_acc.plot(epochs, train_acc, marker="o", label="train")
    ax_acc.plot(epochs, val_acc, marker="o", label="val")
    ax_acc.set_xlabel("epoch")
    ax_acc.set_ylabel("accuracy")
    ax_acc.set_title("Accuracy")
    ax_acc.legend()

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot train/val loss and accuracy curves from a training_log.csv."
    )
    parser.add_argument(
        "csv_path", type=Path, help="Path to a training_log.csv written by train.py"
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Where to save the plot (default: <csv_path's directory>/curves.png)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    output = args.output or args.csv_path.parent / "curves.png"
    plot_training_curves(args.csv_path, output)
    print(f"Wrote {output}")
