import matplotlib.pyplot as plt
import pandas as pd

def validation_train_loss(csv_file, output_path="loss_curve.png"):
    """
    """
    df = pd.read_csv(csv_file)

    plt.plot(df["epoch"], df["train_loss"], label="Train Loss")
    plt.plot(df["epoch"], df["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(output_path)
    plt.close()

