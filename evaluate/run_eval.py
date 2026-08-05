from cli.inference import load_model
from evaluate.generate_predictions import generate_predictions
from evaluate.metrics import confusion_matrix, total_accuracy, per_genre
from evaluate.visualize import plot_confusion_matrix, plot_per_genre_accuracy


model, label_mapping = load_model("experiments/dr0.25-wd1e-4-bnT-200ep/"
                                  "20260803-164349_final/checkpoint.pt")

predicted_labels, true_labels = generate_predictions(model,
                                                     "data/processed/test.npz")

print(f"Total segments evaluated: {len(true_labels)}")

accuracy = total_accuracy(true_labels, predicted_labels)
genre_accuracy = per_genre(true_labels, predicted_labels)
matrix_dict = confusion_matrix(true_labels, predicted_labels)

print(f"Overall accuracy: {accuracy:.4f}")
print(f"Per-genre accuracy: {genre_accuracy}")

plot_confusion_matrix(matrix_dict, output_path="docs/figures/confusion_matrix.png")
plot_per_genre_accuracy(genre_accuracy, output_path="docs/figures/per_genre_accuracy.png")
