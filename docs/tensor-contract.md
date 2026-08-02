# Tensor Contract

**Pipeline -> Model input:** `(N, 1, 128, 130)`, float32
- N = batch size
- 1 = single channel
- 128 = mel frequency bands
- 130 = time frames per 3-second clip

**Model output:** `(N, 10)`, raw logits (no softmax - CLI applies that later)

**Labels:** genre index (0-9):

| Index | Genre | Index | Genre |
|---|---|---|---|
| 0 | blues | 5 | jazz |
| 1 | classical | 6 | metal |
| 2 | country | 7 | pop |
| 3 | disco | 8 | reggae |
| 4 | hiphop | 9 | rock |

## Pipeline output artifacts

`build_dataset.py` writes these to `data/processed/` (not tracked in git --
`data/` and `*.npy`/`*.npz` are gitignored; rerun `build_dataset.py` locally
to regenerate them):

| File | Keys | Shape | Dtype | Notes |
|---|---|---|---|---|
| `train.npz` | `X`, `y`, `song_id` | `(N,1,128,T)`, `(N,)`, `(N,)` | float32, int64, int64 | Normalized (see below); `song_id` unique only within this file, see below |
| `val.npz` | `X`, `y`, `song_id` | `(N,1,128,T)`, `(N,)`, `(N,)` | float32, int64, int64 | Normalized with **train's** stats |
| `test.npz` | `X`, `y`, `song_id` | `(N,1,128,T)`, `(N,)`, `(N,)` | float32, int64, int64 | Normalized with **train's** stats |
| `norm_stats.npz` | `mean`, `std` | `(128,)`, `(128,)` | float32, float32 | Per-mel-band, computed from train only |
| `meta.json` | -- | -- | -- | Preprocessing params: `sr`, `segment_sec`, `n_mels`, `n_fft`, `hop_length`, `db_ref`, `genres`, `norm_epsilon`, `seed`, `split_ratios`, `split_sizes` |

`T` (time frames) is **130 for the default settings** (`sr=22050`,
`segment_sec=3.0`, `hop_length=512`) -- `build_dataset.py --sr`/`--segment-sec`
can change it for a given run. `meta.json` records the exact parameters that
run actually used; treat it, not this table, as the source of truth if you're
consuming a non-default run's output.

`X` is already normalized (per-mel-band mean 0 / std 1, stats from the
training split only) -- consumers don't need to re-normalize. The CLI's live
preprocessing (`cli/predict.py`'s `preprocess_audio_file`, currently a stub)
will need `norm_stats.npz` plus `meta.json`'s parameters to reproduce this
exact chain on a user-submitted clip.

**Caveats:**
- **Song-level split == file-level split for GTZAN**, since each file is one
  song. GTZAN is known to contain some duplicate/re-released recordings
  across its files; the split does not detect or correct for that.
- **Segment counts per song are not always exactly 10.** Real GTZAN clip
  lengths vary by a handful of samples around 30s; a song slightly under 30s
  yields fewer than 10 segments rather than an error (its trailing partial
  segment is dropped). Confirmed on the real dataset: 799 train songs
  produced 7981 segments, not a flat 7990.
- **`song_id` is unique only *within* one split's own `.npz` file, not
  globally.** `song_id=5` in `val.npz` and `song_id=5` in `train.npz` are
  unrelated songs -- don't concatenate across splits and group by `song_id`
  expecting it to mean anything. Assigned as the 0-based index of the song
  within that split's song list, in the order `build_split_arrays` processed
  them; a song that's too short to yield any segments leaves a gap in the
  sequence (see below) rather than shifting later songs' ids down.
- **Segments sharing a `song_id` are always contiguous rows** in `X`/`y`/
  `song_id` -- `build_split_arrays` appends every segment of one song before
  moving to the next, so a segment's position within its song (if needed)
  can be recovered by counting the offset from that run's first row, rather
  than needing its own serialized array.
- **`song_id` is not guaranteed dense.** A too-short song (see the segment-
  count caveat above) contributes zero rows, so its index is skipped
  entirely rather than reused or backfilled -- e.g. songs `[ok, too_short,
  ok]` produce `song_id` values `[0, 0, .., 2, 2, ..]`, never `1`. Group by
  the actual unique values present (e.g. `np.unique`), not by iterating
  `range(max(song_id) + 1)`, and validate song counts via
  `len(np.unique(song_id))` rather than `max(song_id) + 1`. Note this can
  diverge from `meta.json`'s `split_sizes.songs` count: that field counts
  every song *assigned* to the split, while `len(np.unique(song_id))` only
  counts songs that actually yielded ≥1 segment. They're equal today only
  because no song in the real GTZAN splits is fully dropped (799/100/100
  songs -> 7981/1000/1000 segments, none at zero) -- don't assume they'll
  always match if a future dataset variant does drop a song entirely.
