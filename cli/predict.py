# entry: argparse full flow of CLI process
from cli.validation import validate_audio_file
from cli.inference import load_model, predict_genre
from preprocess import melspectrogram_from_audio, segment_audio, DEFAULT_SR
from build_dataset import apply_normalization
import argparse
import librosa
import torch
import numpy as np


def preprocess_audio_file(audio_file):
    """
    Take user audio file creates a segmented 3 second mel spectrograms
    turns them into the right tensor shape

    Args:
        audio_file: a file path name passed in on the command line from user

    Returns:
        tensors: a list of tensors, 3 second segments of mel spectrogram data
        in tensor shape
    """
    y, sr = librosa.load(audio_file, sr=DEFAULT_SR)
    norm_stats = np.load("data/processed/norm_stats.npz")
    mean = norm_stats["mean"]
    std = norm_stats["std"]

    tensors = []
    for segment in segment_audio(y, sr, segment_sec=3.0):
        mel_spec = melspectrogram_from_audio(segment, sr)

        mel_spec_seg = mel_spec[np.newaxis, np.newaxis, :, :]
        apply_normalization(mel_spec_seg, mean, std)
        tensor = torch.from_numpy(mel_spec).unsqueeze(0).unsqueeze(0)
        tensors.append(tensor)
    return tensors


def parse_arguments():
    """
    Parse command line arguments for the genre prediction script.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="Find the top-n closest music genre matches "
                                     "for your audio file, ranked by confidence.")

    parser.add_argument("audio_file", type=str, help="Path to the audio file for genre prediction.")

    # is the number n predefined or is it a user input? If user input, add argument for n
    parser.add_argument("-n", "--top_n", type=int, default=5,
                        help="Number of top genre matches to return (default: 5).")

    args = parser.parse_args()

    return args


def main():

    args = parse_arguments()

    try:
        validate_audio_file(args.audio_file)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        return

    # temporary prints for progress #1 tracking
    print(f"Validated audio file: {args.audio_file}")

    tensors = preprocess_audio_file(args.audio_file)
    model, label_mapping = load_model()
    results = predict_genre(model, tensors, label_mapping, args.top_n)

    for genre, conf in results:
        print(f"Genre: {genre} Confidence Level: {conf:.1f}%")


if __name__ == "__main__":
    main()
