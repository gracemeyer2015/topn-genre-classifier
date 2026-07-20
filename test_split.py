import pytest

from split import split_songs

# Deliberately not imported from build_dataset.GENRES: split_songs is generic
# over any genre labels (it never hardcodes a genre list, see split.py), so
# these tests use their own list to prove that genericity rather than
# incidentally testing against build_dataset's specific 10 genres.
GENRES = [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock",
]


def _make_pairs(per_genre=12):
    """Fake (path, genre) pairs. split_songs never opens the file, so a bare
    filename string stands in for a real audio path here."""
    return [
        (f"{genre}.{i:05d}.wav", genre)
        for genre in GENRES
        for i in range(per_genre)
    ]


def test_split_is_disjoint_at_song_level():
    pairs = _make_pairs()
    train, val, test = split_songs(pairs)

    train_paths = {p for p, _ in train}
    val_paths = {p for p, _ in val}
    test_paths = {p for p, _ in test}

    assert not (train_paths & val_paths)
    assert not (train_paths & test_paths)
    assert not (val_paths & test_paths)
    assert len(train) + len(val) + len(test) == len(pairs)


def test_split_ratios_are_approximately_80_10_10():
    pairs = _make_pairs(per_genre=50)
    train, val, test = split_songs(pairs)

    total = len(pairs)
    assert abs(len(train) / total - 0.8) < 0.03
    assert abs(len(val) / total - 0.1) < 0.03
    assert abs(len(test) / total - 0.1) < 0.03


def test_split_keeps_genre_ratios_balanced():
    train, val, test = split_songs(_make_pairs(per_genre=20))

    for split in (train, val, test):
        counts = {genre: 0 for genre in GENRES}
        for _, genre in split:
            counts[genre] += 1
        assert min(counts.values()) > 0
        assert max(counts.values()) - min(counts.values()) <= 1


def test_split_is_deterministic_for_same_seed():
    pairs = _make_pairs()

    train1, val1, test1 = split_songs(pairs, seed=7)
    train2, val2, test2 = split_songs(pairs, seed=7)

    assert train1 == train2
    assert val1 == val2
    assert test1 == test2


def test_split_changes_with_different_seed():
    pairs = _make_pairs()

    train_a, _, _ = split_songs(pairs, seed=1)
    train_b, _, _ = split_songs(pairs, seed=2)

    assert train_a != train_b


def test_split_raises_on_empty_pairs():
    with pytest.raises(ValueError, match="empty"):
        split_songs([])


def test_split_raises_on_ratios_not_summing_to_one():
    with pytest.raises(ValueError, match="sum to 1.0"):
        split_songs(_make_pairs(), train_frac=0.8, val_frac=0.1, test_frac=0.2)


def test_split_raises_on_negative_ratio():
    with pytest.raises(ValueError, match="must be positive"):
        split_songs(_make_pairs(), train_frac=1.1, val_frac=0.1, test_frac=-0.2)


def test_split_raises_on_too_few_samples_per_genre():
    pairs = [("blues.0.wav", "blues"), ("rock.0.wav", "rock")]
    with pytest.raises(ValueError):
        split_songs(pairs)
