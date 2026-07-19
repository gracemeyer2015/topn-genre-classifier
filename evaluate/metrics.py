import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, classification_report

true_labels =       ["jazz", "rock", "jazz", "blues", "rock", "jazz", "blues", "rock", "blues", "jazz"]
predicted_labels = ["jazz", "rock", "blues", "blues", "rock", "blues", "blues", "jazz", "blues", "jazz"]


print(classification_report(true_labels, predicted_labels))


def total_accuracy(true_labels, predicted_labels):
    count = 0
    for genre in range(len(true_labels)):
        if true_labels[genre] == predicted_labels[genre]:
            count += 1
    
    accuracy = count/len(true_labels)
    return accuracy


def per_genre(true_labels, predicted_labels):

    genre_counts = {}
    per_genre_accuracy = {}

    for i in range(len(true_labels)):
        genre = true_labels[i]
        if genre not in genre_counts:
            genre_counts[genre] = {"total": 0, "correct":0}
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

     
def validation_train_loss(csv_file, output_path = "loss_curve.png"):
    df = pd.read_csv(csv_file)

    plt.plot(df["epoch"], df["train_loss"], label = "Train Loss")
    plt.plot(df["epoch"], df["val_loss"], label = "Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(output_path)
    plt.close()





my_accuracy = total_accuracy(true_labels, predicted_labels)
print(f"My acccuracy: {my_accuracy}")

sk_accuracy = accuracy_score(true_labels, predicted_labels)
print(f"Sklearn accuracy: {sk_accuracy}")