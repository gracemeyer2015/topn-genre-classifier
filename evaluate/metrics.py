from sklearn.metrics import accuracy_score, classification_report
import numpy as np


def total_accuracy(true_labels, predicted_labels):
    """
    Compares each predicted genre label with the corresponding true
    genre label and returns the proportion of correct predictions

    Args:
        true_label (list[str]): the actual genres from the test data
        predicted_labels (list[str]): the genres predicted by the model

    Returns:
        accuracy (float): The overall accuracy, the number of correct
        predictions divided by the total number of predictions
        range 0.0-1.0
    """
    count = 0
    for genre in range(len(true_labels)):
        if true_labels[genre] == predicted_labels[genre]:
            count += 1

    accuracy = count/len(true_labels)
    return accuracy


def per_genre(true_labels, predicted_labels):
    """
    Calculates the accuracy of classification per each genre

    Compares the predicted genre labels with true genre labels
    computes the accuracy for each genre independently

    Args:
        true_label (list[str]): the actual genres from the test data
        predicted_labels (list[str]): the genres predicted by the model

    Returns:
        per_genre_accuracy (dict[str, float]): maps each genre label to
        its accuracy correct predictions / total predictions for that
        genre, range 0.0-1.0
    """
    genre_counts = {}
    per_genre_accuracy = {}

    for i in range(len(true_labels)):
        genre = true_labels[i]
        if genre not in genre_counts:
            genre_counts[genre] = {"total": 0, "correct": 0}
        if genre == predicted_labels[i]:
            genre_counts[genre]["correct"] += 1

        genre_counts[genre]["total"] += 1

    for genre in genre_counts:
        total = genre_counts[genre]["total"]
        correct = genre_counts[genre]["correct"]

        accuracy = (correct/total)
        per_genre_accuracy[genre] = accuracy

    return per_genre_accuracy


def confusion_matrix(true_labels, predicted_labels):
    """
    Builds a confusion matrix comparing true vs.
    predicted genre labels

    For every prediction, counts number of times
    the classified predicted genre matches the
    true genre label

    Args:
        true_label (list[str]): the actual genres from the test data
        predicted_labels (list[str]): the genres predicted by the model

    Returns:
        confusion_matrix (dict[str, dict[str, int]]): A nested dict
        the outer dict key gives the segments true genre label, and the
        inner dict maps each predicted genre to the number of times that
        genre was predicted. This gives the genre mix-ups at the segment
        level
    """
    confusion_matrix = {}

    for i in range(len(true_labels)):
        genre = true_labels[i]
        predicted = predicted_labels[i]
        if genre not in confusion_matrix:
            confusion_matrix[genre] = {}
        if predicted not in confusion_matrix[genre]:
            confusion_matrix[genre][predicted] = 1
        else:
            confusion_matrix[genre][predicted] += 1

    return confusion_matrix


def top_k_accuracy(probabilities, true_label_indices, k=2):
    """
    Fraction of examples where the true label is given in
    the top-k highest probability predictions

    Args:
        probabilties: numpy array shape (N, 10) N = # segments, 10 = # genres
        true_label_indices: array shape (N) gives the true genre index
        k: int the number of top predictions

    Returns:
        Float fraction of the number of segments or songs that had the
        true label within their top k predictions
    """
    sorted_indices = np.argsort(probabilities, axis=1)
    top_k_indices = sorted_indices[:, -k:]
    N = probabilities.shape[0]
    count = 0

    for i in range(N):
        if true_label_indices[i] in top_k_indices[i]:
            count += 1

    return count / N


def main():
    """
    Gives a manual test of the metrics functions using fake labels,
    separate from any given model predictions or test data
    """
    true_labels = ["rock", "jazz", "rock", "pop", "jazz", "rock"]
    predicted_labels = ["rock", "jazz", "pop", "pop", "rock", "rock"]

    my_accuracy = total_accuracy(true_labels, predicted_labels)
    print(f"My accuracy: {my_accuracy}")

    sk_accuracy = accuracy_score(true_labels, predicted_labels)
    print(f"Sklearn accuracy: {sk_accuracy}")

    print(classification_report(true_labels, predicted_labels))

    print("Per-genre accuracy:", per_genre(true_labels, predicted_labels))
    print("confusion matrix", confusion_matrix(true_labels, predicted_labels))


if __name__ == "__main__":
    main()
