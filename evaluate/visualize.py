import matplotlib.pyplot as plt
from build_dataset import GENRES
# graph visual polish help from Claude (Sonnet 5) prompting inspired by the below article
# https://practicaldatascience.org/notebooks/class_5/week_1/2.2.2_making_plots_pretty_2.html


plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["xtick.labelsize"] = 8
plt.rcParams["ytick.labelsize"] = 8
plt.rcParams["axes.titlepad"] = 20


def plot_confusion_matrix(confusion_matrix, genres=GENRES, output_path="confusion_matrix.png"):
    """
    Plots the confusion matrix as a heatmap and saves it

    Converts the nested dict confusion_matrix into a 2D grid,
    using a fixed genre order to line up true/predicted genre
    pairs plots a grid as a color coded heatmap using
    matplotlib gives the count per each true predicted genre
    mapping

    Args:
        confusion_matrix (dict[str, dict[str, int]]): nested dict returned by
        confusion_matrix within metrics.py
        genres (tuple[str]): full ordered list of genres imported from
        build_dataset.py for consistent row/column order
        output_path (str): path file where the figure is saved to

    Returns:
        None side effect of saving figure to file
    """
    matrix = [[confusion_matrix.get(true_genre, {}).get(pred_genre, 0)
              for pred_genre in genres] for true_genre in genres]

    # Used later for text color changes
    max_count = max(max(r) for r in matrix)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matrix, cmap="Blues")
    cbar = fig.colorbar(im, ax=ax, label="Count", pad=0.02)
    cbar.outline.set_visible(False)

    ax.set_xticks(range(len(genres)))
    ax.set_xticklabels(genres, rotation=45, ha='right')
    ax.set_yticks(range(len(genres)))
    ax.set_yticklabels(genres)

    ax.set_title("Genre Confusion Matrix Per Segment", fontweight="bold")
    ax.set_xlabel("Predicted Genre", labelpad=12)
    ax.set_ylabel("True Genre", labelpad=12)

    # Removes black box around grid on all sides
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_xticks([x - 0.5 for x in range(1, len(genres))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(genres))], minor=True)
    ax.grid(which="minor", color="white", linewidth=2.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    for row in range(len(genres)):
        for col in range(len(genres)):
            count = matrix[row][col]
            color = "white" if count >= (max_count / 1.3) else "#222"
            weight = "bold" if row == col else "normal"
            ax.text(col, row, count, ha='center', va='center', color=color, fontweight=weight)

    fig.tight_layout(pad=5.0)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_per_genre_accuracy(per_genre_accuracy, genres=GENRES,
                            output_path="per_genre_accuracy.png"):
    """
    Plots a per genre test accuracy and saves it to
    per_genre_accuracy.png

    Args:
        per_genre_accuracy (dict[str, float])
        genres (tuple[str]): full ordered list of genres imported from
        build_dataset.py for consistent row/column order
        output_path (str): path file where the figure is saved to

    Returns:
        None saves the plot as a png with output_path
    """
    accuracies = [per_genre_accuracy.get(genre, 0) for genre in genres]
    fig, ax = plt.subplots(figsize=(10, 6.5))
    bars = ax.bar(genres, accuracies, color="#3088BF", width=0.6)

    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
                f"{acc:.0%}", ha="center", va="bottom", fontsize=10)

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    ax.set_xlabel("Genre", labelpad=16)
    ax.set_ylabel("Accuracy", labelpad=12)
    ax.set_title("Per-Genre Test Accuracy", fontweight="bold")
    ax.set_xticks(range(len(genres)))
    ax.set_xticklabels(genres, rotation=45, ha='right', rotation_mode="anchor")
    ax.set_xticklabels(genres, rotation=45, ha='right', rotation_mode="anchor")
    ax.tick_params(axis='y')

    fig.tight_layout(pad=2.0)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
