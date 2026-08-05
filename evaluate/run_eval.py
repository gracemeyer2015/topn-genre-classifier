from collections import Counter
from cli.inference import load_model
from evaluate.generate_predictions import generate_predictions
from evaluate.metrics import confusion_matrix, total_accuracy, per_genre
from evaluate.visualize import plot_confusion_matrix, plot_per_genre_accuracy
from build_dataset import GENRES
from evaluate.song_level import (
    aggregate_song_probabilities, song_level_true_labels, run_model_on_split
)


CHECKPOINT_PATH = "experiments/dr0.25-wd1e-4-bnT-200ep/20260803-164349_final/checkpoint.pt"
CONFIG_PATH = "experiments/dr0.25-wd1e-4-bnT-200ep/20260803-164349_final/config.json"
TEST_DATA_PATH = "data/processed/test.npz"

model, label_mapping = load_model(CHECKPOINT_PATH)

# ---------------------------------------------------------------------------------------------
# Segment Evaluation
# ---------------------------------------------------------------------------------------------
predicted_labels, true_labels = generate_predictions(model, TEST_DATA_PATH)

print(f"Total Segments Evaluated: {len(true_labels)}")
print(f"Test Set Genre Distribution: {Counter(true_labels)}")

accuracy = total_accuracy(true_labels, predicted_labels)
genre_accuracy = per_genre(true_labels, predicted_labels)
matrix_dict = confusion_matrix(true_labels, predicted_labels)

print(f"Segment-Level Accuracy: {accuracy:.4f}")
print(f"Segment-Level Per-Genre Accuracy: {genre_accuracy}")

plot_confusion_matrix(matrix_dict, output_path="docs/figures/confusion_matrix.png")
plot_per_genre_accuracy(genre_accuracy, output_path="docs/figures/per_genre_accuracy.png")


# ---------------------------------------------------------------------------------------------
# Song-level Evaluation
# ---------------------------------------------------------------------------------------------
probabilities, y, song_id = run_model_on_split(model, TEST_DATA_PATH)

unique_song_ids, avg_probs = aggregate_song_probabilities(probabilities, song_id)
_, true_label_indices = song_level_true_labels(y, song_id)
predicted_indices = avg_probs.argmax(axis=1)

song_true_labels = [GENRES[i] for i in true_label_indices]
song_pred_labels = [GENRES[i] for i in predicted_indices]

song_accuracy = total_accuracy(song_true_labels, song_pred_labels)
song_genre_accuracy = per_genre(song_true_labels, song_pred_labels)
song_matrix_dict = confusion_matrix(song_true_labels, song_pred_labels)


print(f"Song-Level Accuracy: {song_accuracy:.4f}")
print(f"Song-Level Per-Genre Accuracy: {song_genre_accuracy}")

plot_confusion_matrix(song_matrix_dict, output_path="docs/figures/song_level_confusion_matrix.png")
plot_per_genre_accuracy(song_genre_accuracy, output_path="docs/figures/"
                                                         "song_level_per_genre_accuracy.png")

# ---------------------------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------------------------

print(f"Segment-Level Accuracy: {accuracy:.4f}")
print(f"Song-Level Accuracy: {song_accuracy:.4f}")
print(f"Difference: {song_accuracy - accuracy:+.4f}")
