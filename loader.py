"""Discover and validate GTZAN-style audio files, labeled by genre folder."""

import argparse
import logging
from pathlib import Path

import soundfile as sf

logger = logging.getLogger(__name__)

ROW_FMT = "{:<12} {:>10} {:>7} {:>8}"


def discover_audio(root):
    """Walk root/<genre>/*.wav and return [(path, genre_label), ...].

    Labels are the lowercased genre folder name, as strings; integer
    encoding is left to downstream consumers.
    Matching is extension-case-insensitive (.wav/.WAV) so results don't vary
    across filesystems. Non-.wav files are ignored. An empty genre folder
    logs a warning but does not raise.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(
            f"Audio root not found or not a directory: {root}. Expected layout: "
            "<root>/<genre>/*.wav. The dataset is not tracked in git and must be "
            "downloaded separately."
        )
    pairs = []
    for genre_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        genre = genre_dir.name.lower()
        wavs = sorted(
            p for p in genre_dir.iterdir()
            if p.is_file() and p.suffix.lower() == ".wav"
        )
        if not wavs:
            logger.warning("No .wav files found in genre folder: %s", genre_dir)
            continue
        pairs.extend((wav, genre) for wav in wavs)
    return pairs


def validate_audio(path):
    """Lightweight decode check: can soundfile read this file's header?

    Header-only by design: decoding every file fully is far slower, so a
    file with a valid header but corrupt audio data can still pass. Returns
    True if readable, False if corrupt/unrecognized. Only catches
    soundfile's documented decode error, not arbitrary exceptions.
    """
    try:
        sf.info(str(path))
        return True
    except sf.LibsndfileError as exc:
        logger.warning("Skipping unreadable file %s: %s", path, exc)
        return False


def load_dataset(root):
    """Discover + validate all audio under root; return only the valid pairs.

    Also returns a per-genre summary dict of discovered/valid/skipped counts
    (empty genre folders appear with zero counts), used for the CLI report.
    """
    root = Path(root)
    all_pairs = discover_audio(root)

    summary = {
        p.name.lower(): {"discovered": 0, "valid": 0, "skipped": 0}
        for p in root.iterdir() if p.is_dir()
    }
    valid_pairs = []
    for path, genre in all_pairs:
        counts = summary[genre]
        counts["discovered"] += 1
        if validate_audio(path):
            valid_pairs.append((path, genre))
            counts["valid"] += 1
        else:
            counts["skipped"] += 1
    return valid_pairs, summary


def main():
    parser = argparse.ArgumentParser(
        description="Discover and validate GTZAN-style audio files."
    )
    parser.add_argument(
        "root", type=Path, nargs="?",
        default=Path("data/genres_original"),
        help="Root folder containing <genre>/*.wav subfolders",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        valid_pairs, summary = load_dataset(args.root)
    except FileNotFoundError as exc:
        parser.error(str(exc))

    totals = {
        key: sum(s[key] for s in summary.values())
        for key in ("discovered", "valid", "skipped")
    }
    print(ROW_FMT.format("genre", "discovered", "valid", "skipped"))
    for genre in sorted(summary):
        s = summary[genre]
        print(ROW_FMT.format(genre, s["discovered"], s["valid"], s["skipped"]))
    print(ROW_FMT.format("TOTAL", totals["discovered"], totals["valid"],
                         totals["skipped"]))


if __name__ == "__main__":
    main()
