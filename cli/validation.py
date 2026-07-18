# file existence extension check librosa load check
from pathlib import Path
import librosa


def validate_audio_file(file_path):
    """"
    Check that the filepath exists, and has a supported audio file
    extension for librosa to preprocess

    Args:
        file_path (str): Path to the audio file.

    Returns:
        None: Raises an exception if the file does not exist or has an unsupported extension.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    extension = file_path.suffix.lower()
    supported_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.aiff',
                            '.aif', '.aifc', '.au', '.caf']

    exists = extension in supported_extensions
    if not exists:
        raise ValueError(f"The file {file_path} has an unsupported extension."
                         f"supported extensions are: {supported_extensions}")

    try:
        # Attempt to load the audio file with librosa to check if it's valid
        librosa.load(file_path, sr=None)
    except Exception as e:
        raise ValueError(f"The file {file_path} could not be loaded by librosa. Error: {e}")
