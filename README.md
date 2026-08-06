# Top-N Genre Classifier

A Convolutional Neural Network that predicts a song's top 5 genres, each
with a confidence percentage. Trained on the
[GTZAN dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)
to recognize 10 genres (blues, classical, country, disco, hiphop, jazz,
metal, pop, reggae, rock). Available both as a command-line tool and a web app.

## Try it

- **Live web app:** [cnn-topn-genre-classifier.streamlit.app](https://cnn-topn-genre-classifier.streamlit.app/)
- **Run it locally:** `streamlit run app.py`, then upload a song in the browser.
- **CLI:** `python -m cli.predict path/to/song.mp3`

## Setup

```
pip install -r requirements.txt
```

Use a different file if this doesn't match your machine:
- **Intel Mac:** `pip install -r requirements-intel-mac.txt` (PyTorch dropped Intel-Mac wheel builds after 2.2.x, so this pins an older compatible set)
- **Windows/Linux with an NVIDIA GPU:** `pip install -r requirements-cuda.txt --extra-index-url https://download.pytorch.org/whl/cu126`

Also requires `ffmpeg` on your system (used as a fallback audio decoder for
files `soundfile` can't handle directly):

```
brew install ffmpeg        # macOS
apt-get install ffmpeg     # Debian/Ubuntu
winget install ffmpeg      # Windows
```

## Rebuilding the dataset and model

The trained checkpoint is already committed under `experiments/`, so this
is only needed to retrain from scratch.

Download the [GTZAN dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)
and move `genres_original/` into the project as `data/genres_original/`.

Then process it into training arrays and train:

```
python build_dataset.py
python -m model.train --dropout-rate 0.25 --weight-decay 0.0001 --batch-norm --epochs 200 --patience 25 --seed 42
```

## Testing

CI (`.github/workflows/ci.yml`) runs both a lint check and the test suite
on every push and pull request:

```
flake8 .
pytest
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

## Authors

Oregon State University CS467 Capstone Project.

- [Thomas Kiss](https://github.com/thomas-kiss)
- [Grace Meyer](https://github.com/gracemeyer2015)
- [Elizabeth Peyton](https://github.com/eapeyton205)
