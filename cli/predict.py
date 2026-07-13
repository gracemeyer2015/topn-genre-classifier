# entry: argparse full flow of CLI process
from cli.validation import validate_audio_file
from cli.inference import load_model, predict_genre
import argparse
# import os
import torch


def preprocess_audio_file(audio_file):
    return torch.rand(1, 1, 128, 130)


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

    tensor = preprocess_audio_file(args.audio_file)
    model, label_mapping = load_model()
    results = predict_genre(model, tensor, label_mapping, args.top_n)

    print(results)


if __name__ == "__main__":
    main()
