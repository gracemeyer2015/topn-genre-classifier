"""Song-level train/validation/test split, stratified by genre.

Team handoff contract: this split must run *before* segmentation
(see preprocess.segment_audio). Splitting after segmentation would let 3-second
segments from the same song land in more than one set, so the model would see
near-duplicate data at both train and test time (data leakage), producing an
inflated, meaningless accuracy score.
"""

from sklearn.model_selection import train_test_split


def split_songs(pairs, train_frac=0.8, val_frac=0.1, test_frac=0.1, seed=42):
    """Split (path, genre) pairs into train/val/test sets, stratified by genre.

    Each GTZAN file is one full song, so splitting `pairs` directly *is* the
    song-level split described in the project plan -- there's no need to group
    multiple rows into one song first. (Caveat: GTZAN is known to contain a
    handful of duplicate/re-released recordings across its files; this function
    treats every file as a distinct song and does not detect or correct for that.)

    The split happens in two stratified steps so genre ratios stay equal across
    all three sets: first train vs. (val+test), then val vs. test out of that
    remainder. The two calls use different random_state values (seed, seed + 1)
    so the second split isn't correlated with the first by reusing the same seed.

    Args:
        pairs: [(path, genre_str), ...], e.g. as returned by loader.load_dataset.
        train_frac, val_frac, test_frac: must each be positive and sum to ~1.0
            (default 80/10/10, matching the project plan's target split).
        seed: random_state for the first split; the second split uses seed + 1.

    Returns:
        (train_pairs, val_pairs, test_pairs), each a list of (path, genre) tuples
        in the same format as `pairs`.

    Raises:
        ValueError: if `pairs` is empty, the fractions are invalid, a resulting
            split would be empty, or a genre has too few songs for sklearn to
            stratify at the requested ratios.
    """
    if not pairs:
        raise ValueError("split_songs got an empty pairs list -- nothing to split.")

    for name, frac in (
        ("train_frac", train_frac), ("val_frac", val_frac), ("test_frac", test_frac)
    ):
        if frac <= 0:
            raise ValueError(f"{name} must be positive, got {frac}.")
    total = train_frac + val_frac + test_frac
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"train_frac + val_frac + test_frac must sum to 1.0, got {total}."
        )

    genres = [genre for _, genre in pairs]
    try:
        train_pairs, remainder = train_test_split(
            pairs, train_size=train_frac, stratify=genres, random_state=seed
        )
    except ValueError as exc:
        raise ValueError(
            "Could not stratify the train vs. (val+test) split -- likely a genre "
            f"with too few songs for train_frac={train_frac}. Original error: {exc}"
        ) from exc

    remainder_genres = [genre for _, genre in remainder]
    val_of_remainder = val_frac / (val_frac + test_frac)
    try:
        val_pairs, test_pairs = train_test_split(
            remainder, train_size=val_of_remainder, stratify=remainder_genres,
            random_state=seed + 1,
        )
    except ValueError as exc:
        raise ValueError(
            "Could not stratify the val vs. test split -- likely a genre with too "
            f"few remaining songs for val_frac={val_frac}/test_frac={test_frac}. "
            f"Original error: {exc}"
        ) from exc

    for name, split in (("train", train_pairs), ("val", val_pairs), ("test", test_pairs)):
        if not split:
            raise ValueError(f"{name} split is empty -- check your ratios/sample size.")

    return train_pairs, val_pairs, test_pairs
