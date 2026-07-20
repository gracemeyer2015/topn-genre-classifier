"""Convert audio clips to log-mel spectrograms, and split songs into fixed-length
segments before conversion.

Team handoff contract: melspectrogram_from_audio/audio_to_melspectrogram return a
float32 numpy array of shape (n_mels, time_frames) -- e.g. (128, 130) for a
3-second segment at sr=22050, hop_length=512. segment_audio splits a full song's
waveform into 3-second chunks *before* mel-conversion; see build_dataset.py for
the full song -> segments -> stacked [N, 1, 128, 130] pipeline.
"""

import argparse
from pathlib import Path

import librosa
import numpy as np

DEFAULT_SR = 22050
DEFAULT_HOP_LENGTH = 512


def melspectrogram_from_audio(
    y, sr, n_mels=128, n_fft=2048, hop_length=DEFAULT_HOP_LENGTH
):
    """Convert an already-loaded waveform to a log-mel spectrogram.

    This is the array-in/array-out half of audio_to_melspectrogram, split out
    so that converting many 3-second segments of one song (see segment_audio)
    doesn't require re-reading and re-decoding the file from disk once per
    segment -- the caller loads the file exactly once, then calls this
    function once per segment.

    power_to_db uses ref=np.max deliberately: each chunk passed in is scaled
    relative to its own peak, so downstream models see spectral shape rather
    than absolute loudness (which varies with recording/mastering, not genre).
    One consequence worth knowing: because ref=np.max is computed per call,
    converting a whole song at once and converting that same song's 3-second
    segments separately will *not*, in general, produce byte-identical
    results segment-by-segment -- each segment gets normalized against its
    own peak, not the whole song's peak.

    Args:
        y: 1-D waveform samples (as returned by librosa.load).
        sr: sample rate of `y`, in Hz.
        n_mels, n_fft, hop_length: passed through to librosa.feature.melspectrogram.

    Returns:
        float32 numpy array, shape (n_mels, time_frames).
    """
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db.astype(np.float32)


def audio_to_melspectrogram(
    path, sr=DEFAULT_SR, n_mels=128, n_fft=2048, hop_length=DEFAULT_HOP_LENGTH
):
    """Load an audio file and convert it to a log-mel spectrogram.

    librosa.load resamples to `sr`, downmixes to mono, and returns float32
    samples (nominally in [-1, 1]) regardless of the source file's format.
    See melspectrogram_from_audio for the conversion itself and the ref=np.max
    normalization note.

    Args:
        path: path to an audio file librosa can decode.
        sr, n_mels, n_fft, hop_length: passed through to librosa.load /
            melspectrogram_from_audio.

    Returns:
        float32 numpy array, shape (n_mels, time_frames).
    """
    y, sr = librosa.load(path, sr=sr, mono=True)
    return melspectrogram_from_audio(
        y, sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
    )


def segment_audio(y, sr, segment_sec=3.0):
    """Split a waveform into fixed-length, non-overlapping segments.

    Returns a generator of 1-D float32 arrays, each exactly
    `segment_sec * sr` samples, in order from the start of `y`. A trailing
    partial segment (shorter than a full segment_sec) is dropped rather than
    padded or raised on: real GTZAN clip lengths vary by a handful of samples
    around 30s, so treating "doesn't divide evenly" as an error would fail on
    real data more often than it would catch an actual bug. A clip shorter
    than one full segment yields nothing -- zero segments from a song is not
    an error at this level; it's a caller decision whether that's a problem.

    Argument validation happens here, in this outer function, rather than in
    the generator itself: a generator's body doesn't run at all until you
    start iterating it, so if the validation lived inside the `yield`-ing
    function, segment_audio(y, sr=0) would return successfully and only raise
    later, whenever something finally iterates the result -- a confusing
    place to discover a bad argument. Validating eagerly here means a bad
    call fails immediately, at the call site.

    Args:
        y: 1-D waveform samples, as returned by librosa.load.
        sr: sample rate of `y`, in Hz. Must be positive.
        segment_sec: segment length in seconds. Must be positive.

    Returns:
        A generator yielding 1-D arrays of length int(segment_sec * sr), same
        dtype as `y` (float32 for librosa-loaded audio -- this function only
        slices `y`, it doesn't cast, so a non-float32 input comes out as
        whatever dtype it went in as).

    Raises:
        ValueError: if sr or segment_sec is not positive.
    """
    if sr <= 0:
        raise ValueError(f"sr must be positive, got {sr}.")
    if segment_sec <= 0:
        raise ValueError(f"segment_sec must be positive, got {segment_sec}.")
    return _iter_segments(y, sr, segment_sec)


def _iter_segments(y, sr, segment_sec):
    segment_len = int(segment_sec * sr)
    n_segments = len(y) // segment_len
    for i in range(n_segments):
        start = i * segment_len
        yield y[start:start + segment_len]


def save_spectrogram_plot(mel_spec_db, sr, hop_length, output_path, title=None):
    """Render a mel spectrogram to a PNG using librosa's display helper."""
    # Imported here so importing this module doesn't pay matplotlib's
    # startup cost when no plotting happens.
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
