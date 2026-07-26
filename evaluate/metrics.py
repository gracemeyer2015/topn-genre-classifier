from sklearn.metrics import accuracy_score, classification_report
from evaluate.generate_predictions import generate_predictions
from model.cnn import GenreCNN


def total_accuracy(true_labels, predicted_labels):
    """
    """
    count = 0
    for genre in range(len(true_labels)):
        if true_labels[genre] == predicted_labels[genre]:
            count += 1

    accuracy = count/len(true_labels)
    return accuracy


def per_genre(true_labels, predicted_labels):
    """
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


def main():
    test_data_file_path = "data/processed/test.npz"
    model = GenreCNN()
    predicted_labels, true_labels = generate_predictions(model, test_data_file_path)
    
    my_accuracy = total_accuracy(true_labels, predicted_labels)
    print(f"My accuracy: {my_accuracy}")

    sk_accuracy = accuracy_score(true_labels, predicted_labels)
    print(f"Sklearn accuracy: {sk_accuracy}")

    print(classification_report(true_labels, predicted_labels))
    

if __name__ == "__main__":
    main()