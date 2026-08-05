import matplotlib.pyplot as plt
from build_dataset import GENRES


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

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(len(genres)))
    ax.set_xticklabels(genres, rotation=45, ha='right')
    ax.set_yticks(range(len(genres)))
    ax.set_yticklabels(genres)
    ax.set_title("Genre Confusion Matrix Per Segment", fontsize=22, pad=24)
    ax.set_xlabel("Predicted Genre", fontsize=14, labelpad=12)
    ax.set_ylabel("True Genre", fontsize=14, labelpad=12)

    for row in range(len(genres)):
        for col in range(len(genres)):
            count = matrix[row][col]
            ax.text(col, row, count, ha='center', va='center')

    fig.tight_layout()
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
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(genres, accuracies)

    ax.set_xlabel("Genres", fontsize=14, labelpad=16)
    ax.set_ylabel("Accuracy", fontsize=14, labelpad=12)
    ax.set_title("Per Genre Accuracy", fontsize=22, pad=24)
    ax.set_xticklabels(genres, rotation=45, ha='right', fontsize=11)
    ax.tick_params(axis='y', labelsize=11)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
