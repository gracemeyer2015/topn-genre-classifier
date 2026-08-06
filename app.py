import mutagen
import streamlit as st
import tempfile

from pathlib import Path
from cli.validation import validate_audio_file
from cli.predict import preprocess_audio_file
from cli.inference import load_model, predict_genre

CHECKPOINT_PATH = "experiments/dr0.25-wd1e-4-bnT-200ep/20260803-164349_final/checkpoint.pt"


# Whole script reruns on every upload. cached so the checkpoint loads once
@st.cache_resource
def get_model():
    return load_model(CHECKPOINT_PATH)


def get_song_info(path, fallback_name):
    """Read artist/title tags off the file; fall back to the filename if
    tags are missing (easy=True gives a plain dict across mp3/flac/ogg/etc,
    instead of a different tag-frame API per format)."""
    try:
        tags = mutagen.File(path, easy=True)
        artist = tags.get("artist", [None])[0] if tags else None
        title = tags.get("title", [None])[0] if tags else None
    except mutagen.MutagenError:
        artist = title = None
    return artist or "Unknown artist", title or fallback_name


st.title("Top-n Music Genre Classification Neural Network")
st.write("Upload a song to see the top 5 predicted genres.")
# Native in-app navigation link, rather than a raw URL, since the target is
# another page in this same app
st.page_link("pages/1_Model_Details.py", label="How the model works")

# Renders drag and drop upload widget and returns what user uploads
uploaded_file = st.file_uploader(
    "Choose an audio file",
    # UI convenience, restricts what browser file picker will accept
    # validate_audio_file is the real check later
    type=["wav", "mp3", "flac", "ogg", "aiff"],
)

if uploaded_file is not None:
    suffix = Path(uploaded_file.name).suffix
    # Creates real file on disk in OS temp folder and gives path to it
    # Suffix keeps extension so librosa can get format
    # delete=False persists file beyond with block so librosa.load can open it
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        # First prediction is slow (checkpoint load + inference); this covers
        # that wait so it doesn't look like the app has stalled
        with st.spinner("Analyzing..."):
            # Same pipeline cli/predict.py's main() runs
            validate_audio_file(tmp_path)
            # Splits audio into 3-second segments, returns one tensor per segment
            tensors = preprocess_audio_file(tmp_path)
            model, label_mapping = get_model()
            # Averages predictions across all segments, returns top 5 (genre, confidence%)
            results = predict_genre(model, tensors, label_mapping, top_n=5)

        artist, title = get_song_info(tmp_path, fallback_name=uploaded_file.name)
        st.subheader("Predictions")
        st.write(f"**{title}** by **{artist}**")
        for genre, confidence in results:
            st.write(f"{genre}: {confidence:.1f}%")

    # Same exceptions cli/predict.py's main() catches (invalid file format, clip < 3 sec)
    except (FileNotFoundError, ValueError) as e:
        st.error(str(e))
    finally:
        Path(tmp_path).unlink(missing_ok=True)

st.divider()
st.caption(
    "Built by Thomas Kiss, Grace Meyer, and Elizabeth Peyton. "
    "[View on GitHub](https://github.com/gracemeyer2015/topn-genre-classifier)"
)
