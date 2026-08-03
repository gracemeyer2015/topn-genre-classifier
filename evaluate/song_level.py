"""Song-level accuracy evaluation.

build_dataset.py currently reports segment-level accuracy: each ~30s song
becomes ~10 independent 3-second segments, each scored on its own. This
module aggregates a song's segment predictions (mean of softmax
probabilities) into one prediction per song, and reports song-level
accuracy alongside segment-level accuracy on the same data -- see the
song-level-accuracy-eval plan for the motivation.

Standalone rather than built on evaluate/generate_predictions.py or
evaluate/metrics.py: those exist only on the unmerged
origin/cli-pr2-integration branch, work with argmaxed label strings rather
than probabilities, and still wire up DummyModel -- extending them now would
mean restructuring that code anyway, not just adding a song_id parameter.
Revisit consolidating into one shared prediction-generation path once that
branch merges.
"""

import argparse
import inspect
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from model.cnn import GenreCNN


def aggregate_song_probabilities(probabilities, song_ids):
    """Average per-segment class probabilities within each song.

    Args:
        probabilities: array-like, shape (N, num_classes) -- softmax
            probabilities per segment (not raw logits).
        song_ids: array-like, shape (N,) -- song_id per segment, as written
            by build_dataset.py. Not required to be dense or sorted --
            grouped by the actual unique values present, since a too-short
            song can leave a gap in the sequence (see
            docs/tensor-contract.md).

    Returns:
        (unique_song_ids, avg_probabilities): unique_song_ids is a sorted
        1D array of the distinct song_id values found (empty if there are no
        segments at all). avg_probabilities is shape (len(unique_song_ids),
        num_classes), row i being the mean of every segment's probability
        vector sharing unique_song_ids[i].

    Raises:
        ValueError: if probabilities and song_ids don't describe the same
            number of segments.
    """
    probabilities = np.asarray(probabilities)
    song_ids = np.asarray(song_ids)
    if probabilities.shape[0] != song_ids.shape[0]:
        raise ValueError(
            f"probabilities has {probabilities.shape[0]} rows but song_ids "
            f"has {song_ids.shape[0]} -- they must describe the same segments."
        )

    unique_ids = np.unique(song_ids)
    # Not reachable via build_dataset.py today (build_split_arrays raises
    # before an empty split is ever written to disk), but np.stack([]) on a
    # genuinely empty input raises an opaque "need at least one array to
    # stack" -- this guard keeps that edge case self-explanatory instead.
    if unique_ids.shape[0] == 0:
        return unique_ids, np.empty((0, probabilities.shape[1]), dtype=probabilities.dtype)
    avg = np.stack([
        probabilities[song_ids == song_id].mean(axis=0) for song_id in unique_ids
    ])
    return unique_ids, avg


def song_level_true_labels(y, song_ids):
    """One true label per song, asserting every segment of a song agrees.

    Args:
        y: array-like, shape (N,) -- per-segment genre index.
        song_ids: array-like, shape (N,) -- matching song_id per segment.

    Returns:
        (unique_song_ids, labels): labels[i] is the single genre index
        shared by every segment of unique_song_ids[i].

    Raises:
        ValueError: if y and song_ids don't describe the same number of
            segments, or if any song's segments disagree on y. A song can't
            have two genres -- this is a real data-integrity check on the
            song_id/y pairing, not a defensive style preference, so it's a
            hard failure rather than e.g. taking a majority vote.
    """
    y = np.asarray(y)
    song_ids = np.asarray(song_ids)
    if y.shape[0] != song_ids.shape[0]:
        raise ValueError(
            f"y has {y.shape[0]} elements but song_ids has {song_ids.shape[0]} "
            "-- they must describe the same segments."
        )
    unique_ids = np.unique(song_ids)
    labels = []
    for song_id in unique_ids:
        song_labels = np.unique(y[song_ids == song_id])
        if song_labels.shape[0] != 1:
            raise ValueError(
                f"song_id={song_id} has segments disagreeing on genre "
                f"{song_labels.tolist()} -- a song can't have two genres; "
                "this indicates a corrupted song_id/y pairing."
            )
        labels.append(song_labels[0])
    return unique_ids, np.array(labels, dtype=y.dtype)


def song_level_accuracy(probabilities, y, song_ids):
    """Song-level accuracy: aggregate each song's segments, then score.

    Averages each song's segment probabilities (aggregate_song_probabilities),
    argmaxes to one predicted genre per song, and compares against that
    song's true label (song_level_true_labels). Ties (two classes with an
    exactly equal averaged probability) resolve to the lower class index,
    matching numpy/torch argmax's default tie-breaking -- not a bug, just
    worth knowing.

    Returns:
        (accuracy, song_ids_used): accuracy is a float in [0, 1].
        song_ids_used is the sorted unique song_id array the accuracy was
        computed over, for cross-referencing against segment-level results
        computed on the same underlying data.

    Raises:
        ValueError: if there are zero songs to score. Accuracy over zero
            songs is undefined, not 0.0 -- without this guard, `.mean()` on
            the empty comparison array silently returns NaN with a numpy
            RuntimeWarning instead of a clear failure.
    """
    unique_ids, avg_probs = aggregate_song_probabilities(probabilities, song_ids)
    _, true_labels = song_level_true_labels(y, song_ids)
    if unique_ids.shape[0] == 0:
        raise ValueError("song_level_accuracy got zero songs -- accuracy is undefined.")
    predictions = avg_probs.argmax(axis=1)
    accuracy = float((predictions == true_labels).mean())
    return accuracy, unique_ids


def segment_level_accuracy(probabilities, y):
    """Plain per-segment accuracy, computed from the same predictions used
    for song_level_accuracy -- the point of this module is to see the delta
    between the two on identical underlying data, not two separately-derived
    numbers.

    Raises:
        ValueError: if there are zero segments to score, for the same reason
            song_level_accuracy raises on zero songs -- see its docstring.
    """
    probabilities = np.asarray(probabilities)
    y = np.asarray(y)
    if probabilities.shape[0] == 0:
        raise ValueError("segment_level_accuracy got zero segments -- accuracy is undefined.")
    predictions = probabilities.argmax(axis=1)
    return float((predictions == y).mean())


@torch.no_grad()
def run_model_on_split(model, npz_path, batch_size=32, device="cpu"):
    """Run `model` over every segment in a build_dataset.py split .npz file.

    Args:
        model: an already-constructed, already-loaded (state_dict applied)
            PyTorch model. This function doesn't build or load weights
            itself, so it works with any GenreCNN variant regardless of
            constructor signature -- see build_model_from_config for why
            that matters here.
        npz_path: path to a train/val/test.npz from build_dataset.py,
            expected to contain "X", "y", and "song_id".
        batch_size: inference batch size.
        device: torch device string.

    Returns:
        (probabilities, y, song_id): probabilities is float32 (N,
        num_classes) softmax output; y and song_id are the arrays read
        straight from the .npz file.

    Raises:
        ValueError: if the .npz has no "song_id" key -- it was built before
            song-level provenance was added; rerun build_dataset.py. Also
            raised for batch_size <= 0 (directly CLI-controlled via
            --batch-size): 0 makes range() below a no-op and negative values
            reach np.concatenate([]), both opaque failures otherwise.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")
    with np.load(npz_path) as data:
        if "song_id" not in data:
            raise ValueError(
                f"{npz_path} has no 'song_id' key -- it was built before "
                "song-level provenance was added. Rerun build_dataset.py to "
                "regenerate it."
            )
        X, y, song_id = data["X"], data["y"], data["song_id"]

    model = model.to(device)
    model.eval()

    if X.shape[0] == 0:
        # Not reachable via build_dataset.py today (an empty split is never
        # written), but np.concatenate([]) on a genuinely empty split raises
        # an opaque "need at least one array to concatenate" -- this keeps
        # that edge case self-explanatory instead. 10 matches the fixed
        # genre count in the tensor contract (docs/tensor-contract.md), not
        # something inferred from a model output we never got to compute.
        return np.empty((0, 10), dtype=np.float32), y, song_id

    all_probs = []
    for start in range(0, X.shape[0], batch_size):
        batch = torch.from_numpy(X[start:start + batch_size]).to(device)
        logits = model(batch)
        all_probs.append(F.softmax(logits, dim=1).cpu().numpy())

    probabilities = np.concatenate(all_probs, axis=0).astype(np.float32)
    return probabilities, y, song_id


def build_model_from_config(config_path):
    """Construct GenreCNN from a config.json, tolerant of architecture drift.

    train.py's experiments/<run>/config.json records dropout_rate/
    use_batchnorm per run, but model/cnn.py's constructor signature isn't
    the same on every branch (main's GenreCNN() currently takes no
    arguments; train-real-pipeline-data's takes dropout_rate/use_batchnorm).
    Rather than hardcoding one signature and breaking on the other, this
    inspects the live GenreCNN.__init__ and passes through only the config
    keys it actually accepts -- so this script runs against whichever
    cnn.py happens to be checked out, but can only reproduce the settings
    that architecture supports. See the song-level-accuracy-eval plan's
    "Critical fix" note: evaluating a checkpoint trained with a different
    cnn.py than what's currently checked out requires checking out that
    version of model/cnn.py first (e.g. `git checkout
    train-real-pipeline-data -- model/cnn.py`) -- this function can't paper
    over a genuinely different architecture, only a matching one with
    dropped/added constructor kwargs.

    Args:
        config_path: path to an experiments/<run>/config.json written by
            train.py's _write_config.

    Returns:
        An un-loaded GenreCNN instance -- caller still applies the
        checkpoint's state_dict (see load_checkpoint).
    """
    with open(config_path) as f:
        config = json.load(f)

    accepted = set(inspect.signature(GenreCNN.__init__).parameters) - {"self"}
    kwargs = {
        key: config[key]
        for key in ("dropout_rate", "use_batchnorm")
        if key in config and key in accepted
    }
    return GenreCNN(**kwargs)


def load_checkpoint(model, checkpoint_path, device="cpu"):
    """Load a train.py-style {"model_state_dict": ...} checkpoint into model."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare segment-level vs. song-level accuracy for a trained "
            "GenreCNN checkpoint on a build_dataset.py val/test split. Run "
            "as a module from the repo root (`python -m evaluate.song_level "
            "...`) -- `python evaluate/song_level.py` fails with "
            "ModuleNotFoundError since it doesn't put the repo root, and "
            "therefore the model package, on sys.path."
        )
    )
    parser.add_argument("checkpoint", type=Path, help="Path to a train.py checkpoint.pt")
    parser.add_argument("config", type=Path, help="Path to that run's config.json")
    parser.add_argument(
        "--data", type=Path, default=Path("data/processed/val.npz"),
        help="Path to the split .npz to evaluate (default: %(default)s)",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def main():
    args = _parse_args()
    model = build_model_from_config(args.config)
    model = load_checkpoint(model, args.checkpoint)

    probabilities, y, song_id = run_model_on_split(
        model, args.data, batch_size=args.batch_size
    )

    seg_acc = segment_level_accuracy(probabilities, y)
    song_acc, song_ids_used = song_level_accuracy(probabilities, y, song_id)

    print(f"Segments: {probabilities.shape[0]}, songs: {len(song_ids_used)}")
    print(f"Segment-level accuracy: {seg_acc:.4f}")
    print(f"Song-level accuracy:    {song_acc:.4f}")


if __name__ == "__main__":
    main()
