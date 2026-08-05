# Top-N Genre Classifier

A CNN that predicts a song's genre from its audio. Trained on the GTZAN
dataset to recognize 10 genres (blues, classical, country, disco, hiphop,
jazz, metal, pop, reggae, rock), and available both as a command-line tool
and a web app.

Built by Thomas Kiss, Grace Meyer, and Elizabeth Peyton for CS467.

## Try it

- **Web app:** `streamlit run app.py`, then upload a song in the browser.
- **CLI:** `python -m cli.predict path/to/song.mp3`

## Setup

```
pip install -r requirements.txt
```

Also requires `ffmpeg` on your system (used as a fallback audio decoder for
files `soundfile` can't handle directly):

```
brew install ffmpeg        # macOS
apt-get install ffmpeg     # Debian/Ubuntu
winget install ffmpeg      # Windows
```

## Rebuilding the dataset and model

The trained checkpoint is already committed under `experiments/`, so this
step is only needed to retrain from scratch. Requires the GTZAN dataset at
`data/genres_original/<genre>/*.wav`.

```
python build_dataset.py
python -m model.train --dropout-rate 0.25 --weight-decay 0.0001 --batch-norm --epochs 200 --patience 25 --seed 42
```

## How it works

Audio is split into 3-second segments, each converted to a log-mel
spectrogram, and scored independently; a song's segment predictions are
averaged into one final ranked list of genres. See `pages/1_Model_Details.py`
(in the web app) or `docs/hyperparameter-tuning.md` for the full
architecture, training methodology, and the statistics behind the final
settings.

## Project layout

| Path | What's in it |
|---|---|
| `build_dataset.py`, `preprocess.py`, `loader.py`, `split.py` | Raw audio -> train/val/test `.npz` arrays |
| `model/` | CNN architecture (`cnn.py`) and training loop (`train.py`) |
| `cli/` | Command-line genre prediction |
| `evaluate/` | Test-set accuracy, confusion matrix, song-level aggregation |
| `experiments/` | Every training run's config, logs, and checkpoint |
| `docs/` | Tensor shape contract and the hyperparameter tuning writeup |
| `app.py`, `pages/` | The Streamlit web app |
