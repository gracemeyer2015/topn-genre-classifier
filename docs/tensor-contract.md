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
| `train.npz` | `X`, `y` | `(N,1,128,T)`, `(N,)` | float32, int64 | Normalized (see below) |
| `val.npz` | `X`, `y` | `(N,1,128,T)`, `(N,)` | float32, int64 | Normalized with **train's** stats |
| `test.npz` | `X`, `y` | `(N,1,128,T)`, `(N,)` | float32, int64 | Normalized with **train's** stats |
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
- **Follow-up, not yet implemented:** per-sample `song_id`/`segment_index`
  provenance isn't serialized. Without it, aggregating a song's ~10 segment
  predictions into one song-level prediction/accuracy isn't possible from
  these artifacts alone -- only segment-level metrics are. Add this if/when
  song-level evaluation is needed.
