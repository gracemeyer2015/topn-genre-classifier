import numpy as np

from preprocess import audio_to_melspectrogram


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
