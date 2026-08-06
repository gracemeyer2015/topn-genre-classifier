"""Second page: architecture, training methodology, and final settings.
Condensed from docs/hyperparameter-tuning.md -- see that doc for the full
tables, statistical tests, and source citations."""

import streamlit as st

REPO_URL = "https://github.com/gracemeyer2015/topn-genre-classifier"

st.title("Model Architecture & Training")
st.write(
    "Final settings: `dropout_rate=0.25`, `weight_decay=0.0001`, "
    "`batch_norm=True`, GAP architecture. Chosen from 138 training runs, "
    f"see [the full writeup]({REPO_URL}/blob/main/docs/hyperparameter-tuning.md) "
    "for the statistics and reasoning behind each choice."
)

st.header("Pipeline: audio to spectrogram")
st.write(
    "Each uploaded song is loaded at 22050 Hz and split into non-"
    "overlapping 3-second segments, a 3-4 minute song becomes dozens of "
    "independent segments, each scored separately."
)
st.write(
    "Every segment is converted to a log-mel spectrogram: 128 mel "
    "frequency bands, a 2048-sample FFT window, 512-sample hop length, "
    "producing a 128 x 130 image where one axis is frequency and the "
    "other is time."
)
st.image(
    "docs/figures/sample-melspectrogram.png",
    caption="One 3-second segment from an actual training clip (blues.00000.wav), "
    "brighter = louder at that frequency and moment",
)
st.write(
    "Before scoring, each spectrogram is normalized per frequency band. "
    "The mean and std come from the training data only. Val and test "
    "songs reuse those same numbers, not their own."
)

st.header("Layer by layer")
st.write(
    "Three convolutional blocks, each Conv2d -> BatchNorm2d -> ReLU -> "
    "MaxPool2d, followed by global average pooling, dropout, and one "
    "linear layer down to the 10 genres. 24,170 parameters total."
)
st.table(
    {
        "Layer": [
            "Input", "Conv block 1 (1->16 ch)", "Conv block 2 (16->32 ch)",
            "Conv block 3 (32->64 ch)", "Global avg pool", "Dropout (0.25)",
            "Linear (64->10)",
        ],
        "Output shape": [
            "1 x 128 x 130", "16 x 64 x 65", "32 x 32 x 32", "64 x 16 x 16",
            "64 x 1 x 1", "64", "10",
        ],
        "Params": ["-", "192", "4,704", "18,624", "0", "0", "650"],
    }
)

st.header("Architecture: GAP, not flatten")
st.write(
    "The final layer uses global average pooling (GAP) instead of "
    "flattening. Flattening keeps every spatial position as a separate, "
    "easily-memorized feature, and drove train accuracy to ~99% while val "
    "accuracy stalled near 70%. GAP averages each feature map to one "
    "number instead, shrinking the final layer from 163,850 to 650 "
    "parameters and reducing overfitting."
)
st.image(
    "docs/figures/eliminated-flatten-architecture.png",
    caption="Flatten architecture: severe train/val overfitting gap",
)

st.header("What each setting does")
st.markdown(
    "- **Dropout:** randomly zeroes activations during training so the "
    "network can't rely on any single feature too heavily.\n"
    "- **Weight decay:** an L2 penalty that shrinks weight magnitudes.\n"
    "- **Batch norm:** normalizes each layer's activations using the "
    "current batch's statistics, stabilizing and speeding up training."
)

st.header("Dropout: 0.25")
st.write(
    "0.5 regularized too hard, and 0.0 overfit despite a competitive "
    "raw val_loss. 0.25 won on val_loss, val_accuracy, and train/val gap, "
    "with or without batch norm."
)
st.image("docs/figures/dropout-comparison.png")

st.header("Weight decay & batch norm")
st.write(
    "Batch norm was the strongest finding, confirmed at two epoch "
    "budgets and all three weight-decay values, every comparison was "
    "statistically significant. Weight decay was the closest call: "
    "0.0001 had the best val_loss, val_accuracy, and train/val gap, "
    "though not by a statistically significant margin over 0.001."
)
st.image("docs/figures/wd-bn-grid.png")

st.header("Final settings")
st.table(
    {
        "Setting": ["Architecture", "dropout_rate", "weight_decay", "batch_norm", "seed"],
        "Value": ["GAP", "0.25", "0.0001", "True", "42"],
    }
)
st.write(
    "The final checkpoint was trained once using a fixed seed, not "
    "picked as the best of many runs."
)
st.image("docs/figures/final-config-sample-curve.png")

st.header("From checkpoint to prediction")
st.write(
    "This app and the project's command-line tool (`cli/predict.py`) "
    "both run the same inference path: load the checkpoint's saved "
    "weights into the architecture above, then run each segment through "
    "it in evaluation mode (dropout off, batch norm using its stored "
    "running statistics rather than the current batch's). Each segment "
    "gets its own probability over the 10 genres; those are averaged "
    "together into one set of probabilities for the whole song, and the "
    "top 5 are shown."
)
