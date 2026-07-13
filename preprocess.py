"""Convert audio clips to log-mel spectrograms.

Team handoff contract: audio_to_melspectrogram returns a float32 numpy array
of shape (n_mels, time_frames) -- e.g. (128, 130) for a 3-second segment at
sr=22050, hop_length=512. Segmentation into 3-second, [N, 1, 128, 130] batches
is PR#2 scope; this module only does single-clip conversion.
"""

import argparse
from pathlib import Path

import librosa
import numpy as np

DEFAULT_SR = 22050
DEFAULT_HOP_LENGTH = 512


def audio_to_melspectrogram(
    path, sr=DEFAULT_SR, n_mels=128, n_fft=2048, hop_length=DEFAULT_HOP_LENGTH
):
    """Load an audio file and convert it to a log-mel spectrogram.

    librosa.load resamples to `sr`, downmixes to mono, and returns float32
    samples (nominally in [-1, 1]) regardless of the source file's format.

    power_to_db uses ref=np.max deliberately: each clip is scaled relative to
    its own peak, so downstream models see spectral shape rather than absolute
    loudness (which varies with recording/mastering, not genre).
    """
    y, sr = librosa.load(path, sr=sr, mono=True)
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db.astype(np.float32)


def save_spectrogram_plot(mel_spec_db, sr, hop_length, output_path, title=None):
    """Render a mel spectrogram to a PNG using librosa's display helper."""
    # Imported here so importing this module (tests, PR#2 segmentation code)
    # doesn't pay matplotlib's startup cost for a plot-only code path.
    import librosa.display
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    img = librosa.display.specshow(
        mel_spec_db, sr=sr, hop_length=hop_length,
        x_axis="time", y_axis="mel", ax=ax
    )
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Convert an audio clip to a log-mel spectrogram."
    )
    parser.add_argument("audio_path", type=Path, help="Path to a .wav file")
    parser.add_argument(
        "--plot", action="store_true",
        help="Save a PNG visualization to --output",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("figures"),
        help="A .png file path, or a directory to save <clip>.png in "
             "(default: figures/)",
    )
    args = parser.parse_args()

    mel_spec_db = audio_to_melspectrogram(args.audio_path)
    print(f"{args.audio_path}: shape={mel_spec_db.shape}, "
          f"dtype={mel_spec_db.dtype}, "
          f"range=[{mel_spec_db.min():.1f}, {mel_spec_db.max():.1f}] dB")

    if args.plot:
        if args.output.suffix.lower() == ".png":
            png_path = args.output
        else:
            png_path = args.output / f"{args.audio_path.stem}.png"
        save_spectrogram_plot(
            mel_spec_db, sr=DEFAULT_SR, hop_length=DEFAULT_HOP_LENGTH,
            output_path=png_path, title=args.audio_path.stem,
        )
        print(f"Saved plot to {png_path}")


if __name__ == "__main__":
    main()
