import librosa
import numpy as np
import pytest

from preprocess import (
    DEFAULT_SR,
    audio_to_melspectrogram,
    melspectrogram_from_audio,
    segment_audio,
)


def test_mono_three_second_clip_shape(tmp_path, write_sine_wav):
    wav_path = tmp_path / "mono_3s.wav"
    write_sine_wav(wav_path, duration_sec=3.0)

    mel_spec_db = audio_to_melspectrogram(wav_path)

    assert mel_spec_db.dtype == np.float32
    assert mel_spec_db.shape == (128, 130)
    assert np.isfinite(mel_spec_db).all()


def test_stereo_clip_is_downmixed(tmp_path, write_sine_wav):
    wav_path = tmp_path / "stereo_3s.wav"
    write_sine_wav(wav_path, duration_sec=3.0, channels=2)

    mel_spec_db = audio_to_melspectrogram(wav_path)

    assert mel_spec_db.shape[0] == 128
    assert np.isfinite(mel_spec_db).all()


def test_short_clip_has_fewer_time_frames(tmp_path, write_sine_wav):
    wav_path = tmp_path / "short_1s.wav"
    write_sine_wav(wav_path, duration_sec=1.0)

    mel_spec_db = audio_to_melspectrogram(wav_path)

    assert mel_spec_db.shape[0] == 128
    assert mel_spec_db.shape[1] < 130


def test_n_mels_parameter_sets_output_height(tmp_path, write_sine_wav):
    wav_path = tmp_path / "mono_1s.wav"
    write_sine_wav(wav_path, duration_sec=1.0)

    mel_spec_db = audio_to_melspectrogram(wav_path, n_mels=64)

    assert mel_spec_db.shape[0] == 64


def test_melspectrogram_from_audio_matches_audio_to_melspectrogram(
    tmp_path, write_sine_wav
):
    """The array-in entry point and the file-in entry point must agree.

    Loads the same wav both ways: once via audio_to_melspectrogram (path in),
    and once by loading it ourselves and calling melspectrogram_from_audio
    (array in). Compares against librosa.load's *output*, not the pre-encode
    sine wave -- comparing to the pre-encode array would fail on wav
    quantization noise that has nothing to do with a real bug.
    """
    wav_path = tmp_path / "clip.wav"
    write_sine_wav(wav_path, duration_sec=3.0)

    y, sr = librosa.load(wav_path, sr=DEFAULT_SR, mono=True)

    from_array = melspectrogram_from_audio(y, sr)
    from_file = audio_to_melspectrogram(wav_path, sr=DEFAULT_SR)

    np.testing.assert_array_equal(from_array, from_file)


def test_segment_audio_exact_multiple_yields_expected_count():
    sr = DEFAULT_SR
    y = np.zeros(sr * 6, dtype=np.float32)  # 6s -> two clean 3s segments

    segments = list(segment_audio(y, sr, segment_sec=3.0))

    assert len(segments) == 2
    assert all(len(seg) == sr * 3 for seg in segments)


def test_segment_audio_drops_trailing_partial_segment():
    sr = DEFAULT_SR
    y = np.zeros(sr * 7, dtype=np.float32)  # 7s -> two 3s segments, 1s dropped

    segments = list(segment_audio(y, sr, segment_sec=3.0))

    assert len(segments) == 2


def test_segment_audio_shorter_than_one_segment_yields_nothing():
    sr = DEFAULT_SR
    y = np.zeros(sr * 1, dtype=np.float32)  # 1s, shorter than the 3s default

    segments = list(segment_audio(y, sr, segment_sec=3.0))

    assert segments == []


def test_segment_audio_segments_are_contiguous_and_in_order():
    sr = DEFAULT_SR
    y = np.arange(sr * 6, dtype=np.float32)  # a ramp so content is checkable

    segments = list(segment_audio(y, sr, segment_sec=3.0))

    np.testing.assert_array_equal(segments[0], y[: sr * 3])
    np.testing.assert_array_equal(segments[1], y[sr * 3: sr * 6])


def test_segment_audio_raises_immediately_on_nonpositive_segment_sec():
    # Not wrapped in list(...): validation must fire at the call itself,
    # before any iteration -- see segment_audio's docstring for why.
    with pytest.raises(ValueError, match="segment_sec must be positive"):
        segment_audio(np.zeros(1000, dtype=np.float32), DEFAULT_SR, segment_sec=0)


def test_segment_audio_raises_immediately_on_nonpositive_sr():
    with pytest.raises(ValueError, match="sr must be positive"):
        segment_audio(np.zeros(1000, dtype=np.float32), sr=0)


def test_segment_then_convert_matches_tensor_contract(tmp_path, write_sine_wav):
    """End-to-end sanity check for the segment -> mel-convert chain that
    build_dataset.py will run: 3s @ 22050Hz with hop_length=512 must produce
    exactly 130 time frames per segment, matching docs/tensor-contract.md."""
    wav_path = tmp_path / "clip_6s.wav"
    write_sine_wav(wav_path, duration_sec=6.0)
    y, sr = librosa.load(wav_path, sr=DEFAULT_SR, mono=True)

    mel_specs = [
        melspectrogram_from_audio(segment, sr)
        for segment in segment_audio(y, sr, segment_sec=3.0)
    ]

    assert len(mel_specs) == 2
    for mel_spec in mel_specs:
        assert mel_spec.shape == (128, 130)
        assert mel_spec.dtype == np.float32
        assert np.isfinite(mel_spec).all()
