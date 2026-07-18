"""Build the train/val/test .npz dataset from raw GTZAN audio.

Pipeline: loader.load_dataset (discover + validate) -> split.split_songs
(song-level, stratified by genre) -> per-song segmentation + mel-conversion
(preprocess.segment_audio / preprocess.melspectrogram_from_audio) -> per-band
normalization (stats from train only) -> serialize.

Team handoff contract: writes <output>/{train,val,test}.npz (keys "X" float32
shape (N, 1, 128, 130), "y" int64 shape (N,)), <output>/norm_stats.npz (keys
"mean"/"std", float32 shape (128,) -- the per-mel-band normalization already
applied to X in the three .npz files above), and <output>/meta.json (the
preprocessing parameters needed to reproduce this exact pipeline on a new
clip, e.g. for live inference in the CLI). See docs/tensor-contract.md.
"""

import argparse
import json
from pathlib import Path

import librosa
import numpy as np

from loader import load_dataset
from preprocess import (
    DEFAULT_HOP_LENGTH,
    DEFAULT_SR,
    melspectrogram_from_audio,
    segment_audio,
)
from split import split_songs

# Alphabetical order -- must match docs/tensor-contract.md and the mapping
# already hardcoded in cli/inference.py. Not serialized as a separate array
# per sample (see the plan discussion): it's a fixed, documented contract,
# not per-run data, so it's recorded once in meta.json instead.
GENRES = (
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock",
)
GENRE_TO_INDEX = {genre: index for index, genre in enumerate(GENRES)}

# Floor applied to per-band std before dividing by it in apply_normalization,
# so a band with (near) zero variance in the training data can't produce a
# division blowing up to inf/NaN.
NORM_EPSILON = 1e-8

DEFAULT_ROOT = Path("data/genres_original")
DEFAULT_OUTPUT = Path("data/processed")
DEFAULT_N_MELS = 128
DEFAULT_N_FFT = 2048


def build_split_arrays(
    pairs, sr=DEFAULT_SR, segment_sec=3.0,
    n_mels=DEFAULT_N_MELS, n_fft=DEFAULT_N_FFT, hop_length=DEFAULT_HOP_LENGTH,
):
    """Convert a list of (path, genre) song pairs into stacked model-ready arrays.

    For each song: load the audio once, slice it into segment_sec-length
    segments (preprocess.segment_audio), and convert each segment to a
    log-mel spectrogram (preprocess.melspectrogram_from_audio). Every segment
    becomes one row of the returned arrays -- a 30-second song contributes
    roughly ten rows, not one.

    Args:
        pairs: [(path, genre_str), ...] -- e.g. one split's output from
            split.split_songs. Must not be empty.
        sr, segment_sec, n_mels, n_fft, hop_length: preprocessing parameters,
            passed through to librosa.load / segment_audio /
            melspectrogram_from_audio.

    Returns:
        (X, y): X is float32, shape (N, 1, n_mels, time_frames), N being the
        total segment count across every song in `pairs`. y is int64, shape
        (N,), each entry the alphabetical genre index (GENRES) of the song
        that segment came from.

    Raises:
        ValueError: if `pairs` is empty, `sr`/`segment_sec` is not positive
            (validated here, not just inside preprocess.segment_audio -- see
            the comment below), a song's genre isn't one of the 10 known
            GENRES, or every song in `pairs` was too short to yield even one
            segment. An unrecognized genre is a hard stop rather than a skip:
            a silently-dropped or mis-encoded genre would corrupt the class
            balance with no visible symptom until training produces confusing
            results.
    """
    if not pairs:
        raise ValueError("build_split_arrays got an empty pairs list -- nothing to build.")
    # Validated here too, not just inside segment_audio: librosa.load runs
    # before segment_audio ever sees these values, and librosa raises its own
    # differently-worded error for a bad sr, so without this check the error
    # message a caller sees would depend on which library happened to
    # validate first rather than being consistent.
    if sr <= 0:
        raise ValueError(f"sr must be positive, got {sr}.")
    if segment_sec <= 0:
        raise ValueError(f"segment_sec must be positive, got {segment_sec}.")

    specs = []
    labels = []
    for path, genre in pairs:
        if genre not in GENRE_TO_INDEX:
            raise ValueError(
                f"Unrecognized genre {genre!r} for {path} -- expected one of {GENRES}."
            )
        label = GENRE_TO_INDEX[genre]

        y, loaded_sr = librosa.load(path, sr=sr, mono=True)
        for segment in segment_audio(y, loaded_sr, segment_sec=segment_sec):
            mel_spec = melspectrogram_from_audio(
                segment, loaded_sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
            )
            specs.append(mel_spec)
            labels.append(label)

    if not specs:
        raise ValueError(
            f"None of the {len(pairs)} song(s) in pairs produced a single "
            f"segment_sec={segment_sec}s segment -- every song was too short. "
            "Check segment_sec, sr, or the audio files themselves."
        )

    # copy=False: specs are already float32 (melspectrogram_from_audio casts),
    # so this is a no-op cast rather than a second full-size copy of X -- the
    # same "don't allocate a redundant copy" reasoning as apply_normalization.
    X = np.stack(specs).astype(np.float32, copy=False)[:, np.newaxis, :, :]
    y_arr = np.array(labels, dtype=np.int64)
    return X, y_arr


def compute_band_stats(X_train):
    """Per-mel-band mean/std over the training array, for normalization.

    Computed over axes (0, 1, 3) -- every sample, the single channel, and
    every time frame -- leaving one mean and one std per mel band (axis 2).
    Per-band rather than a single global scalar because low and high mel
    bands carry very different absolute dB ranges; a global scalar would
    leave that imbalance in the data for the model to work around instead of
    normalizing it away.

    The intermediate mean/std computation runs in float64 (numpy's `dtype=`
    kwarg) before casting back to float32, to avoid accumulating rounding
    error across tens of thousands of float32 values. std is floored at
    NORM_EPSILON before returning.

    Args:
        X_train: float32 array, shape (N, 1, n_mels, time_frames). Must come
            from the *training* split only -- see apply_normalization.

    Returns:
        (mean, std): both float32, shape (n_mels,).
    """
    mean = X_train.mean(axis=(0, 1, 3), dtype=np.float64).astype(np.float32)
    std = X_train.std(axis=(0, 1, 3), dtype=np.float64).astype(np.float32)
    std = np.maximum(std, NORM_EPSILON)
    return mean, std


def apply_normalization(X, mean, std):
    """Normalize X in place to per-band mean 0 / std 1, using the given stats.

    In place (X -= ...; X /= ...) rather than X = (X - mean) / std, to avoid
    allocating a second full-size temporary copy of X -- with several
    thousand training segments that's hundreds of MB, so an out-of-place
    version would briefly double peak memory for no benefit.

    `mean`/`std` must come from compute_band_stats(X_train) -- the *training*
    split only -- and those exact same values must be applied to train, val,
    and test. Recomputing stats from val/test themselves would leak
    information about those sets into the normalization the model trains on.

    Args:
        X: float32 array, shape (N, 1, n_mels, time_frames). Modified in place.
        mean, std: float32 arrays, shape (n_mels,), from compute_band_stats.

    Raises:
        AssertionError: if normalization produces a non-finite value. This
            would only happen from an unfloored zero/near-zero in std, i.e.
            a bug in compute_band_stats, not something a caller did wrong.
    """
    X -= mean[None, None, :, None]
    X /= std[None, None, :, None]
    assert np.isfinite(X).all(), "Normalization produced a non-finite value."


def build_dataset(
    root, output, sr=DEFAULT_SR, segment_sec=3.0,
    train_frac=0.8, val_frac=0.1, test_frac=0.1, seed=42,
):
    """Run the full pipeline and write the handoff artifacts to `output`.

    discover (loader.load_dataset) -> split (split.split_songs) -> segment +
    mel-convert each split (build_split_arrays) -> compute normalization
    stats from train only (compute_band_stats) -> apply those stats to all
    three splits in place (apply_normalization) -> serialize.

    Args:
        root: GTZAN root folder, <root>/<genre>/*.wav.
        output: directory to write train.npz/val.npz/test.npz/
            norm_stats.npz/meta.json into. Created if it doesn't exist.
        sr, segment_sec: preprocessing parameters (see build_split_arrays).
        train_frac, val_frac, test_frac, seed: passed to split.split_songs.

    Returns:
        (splits, meta): splits is {"train": (X, y), "val": (X, y),
        "test": (X, y)} (the same arrays written to disk); meta is the dict
        written to meta.json.
    """
    pairs, _load_summary = load_dataset(root)
    train_pairs, val_pairs, test_pairs = split_songs(
        pairs, train_frac=train_frac, val_frac=val_frac, test_frac=test_frac, seed=seed
    )
    song_counts = {"train": len(train_pairs), "val": len(val_pairs), "test": len(test_pairs)}

    splits = {}
    for name, split_pairs in (
        ("train", train_pairs), ("val", val_pairs), ("test", test_pairs)
    ):
        splits[name] = build_split_arrays(split_pairs, sr=sr, segment_sec=segment_sec)

    X_train, _y_train = splits["train"]
    mean, std = compute_band_stats(X_train)
    for X, _y in splits.values():
        apply_normalization(X, mean, std)

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for name, (X, y) in splits.items():
        np.savez(output / f"{name}.npz", X=X, y=y)
    np.savez(output / "norm_stats.npz", mean=mean, std=std)

    meta = {
        "sr": sr,
        "segment_sec": segment_sec,
        "n_mels": DEFAULT_N_MELS,
        "n_fft": DEFAULT_N_FFT,
        "hop_length": DEFAULT_HOP_LENGTH,
        "db_ref": "max",
        "genres": list(GENRES),
        "norm_epsilon": NORM_EPSILON,
        "seed": seed,
        "split_ratios": {"train": train_frac, "val": val_frac, "test": test_frac},
        "split_sizes": {
            "songs": song_counts,
            "segments": {name: int(y.shape[0]) for name, (_X, y) in splits.items()},
        },
    }
    with open(output / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return splits, meta


def main():
    """CLI entry point: parse arguments, run build_dataset, print a summary.

    Note for anyone changing --sr/--segment-sec from their defaults: the
    (N,1,128,130) shape documented in docs/tensor-contract.md is what the
    default 22050 Hz / 3.0s settings produce. Changing either changes the
    time-frame dimension (the 128 mel bands stay fixed -- n_mels isn't
    exposed as a CLI flag). meta.json always records the actual parameters a
    given run used, so it's the source of truth for a non-default run.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Build the train/val/test .npz dataset: discover GTZAN audio, "
            "split by song, segment + mel-convert, normalize per mel band, "
            "and write the handoff artifacts."
        )
    )
    parser.add_argument(
        "root", type=Path, nargs="?", default=DEFAULT_ROOT,
        help="Root folder containing <genre>/*.wav subfolders (default: %(default)s)",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help="Directory to write the .npz/.json artifacts into (default: %(default)s)",
    )
    parser.add_argument(
        "--sr", type=int, default=DEFAULT_SR,
        help="Resample audio to this rate in Hz before processing (default: %(default)s)",
    )
    parser.add_argument(
        "--segment-sec", type=float, default=3.0,
        help="Segment length in seconds; changes the output time-frame "
             "dimension away from the default-documented 130 (default: %(default)s)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for the train/val/test split (default: %(default)s)",
    )
    parser.add_argument(
        "--train-frac", type=float, default=0.8,
        help="Fraction of songs (by genre-stratified count) in train (default: %(default)s)",
    )
    parser.add_argument(
        "--val-frac", type=float, default=0.1,
        help="Fraction of songs in val (default: %(default)s)",
    )
    parser.add_argument(
        "--test-frac", type=float, default=0.1,
        help="Fraction of songs in test (default: %(default)s)",
    )
    args = parser.parse_args()

    splits, meta = build_dataset(
        args.root, args.output, sr=args.sr, segment_sec=args.segment_sec,
        train_frac=args.train_frac, val_frac=args.val_frac,
        test_frac=args.test_frac, seed=args.seed,
    )

    print(f"Wrote arrays to {args.output}/")
    for name in ("train", "val", "test"):
        X, y = splits[name]
        songs = meta["split_sizes"]["songs"][name]
        print(f"  {name:<6} {songs:>4} songs -> X {X.shape} {X.dtype}, "
              f"y {y.shape} {y.dtype}")


if __name__ == "__main__":
    main()
