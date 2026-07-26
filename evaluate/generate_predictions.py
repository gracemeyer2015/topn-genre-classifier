from cli.inference import load_model
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

label_mapping = {0: "blues", 1: "classical", 2: "country", 3: "disco", 4:
                     "hiphop", 5: "jazz", 6: "metal", 7: "pop",
                     8: "reggae", 9: "rock"}

def generate_predictions(model,test_data_file_path,):
    """
    Runs the trained model against the serialized testing set
    returns a list of true/predicted genre labels, strings, 
    for evaluation

    Args:
        model: A trained (currently, untrained) PyTorch model
        test_data_file_path (str): .npz file containing the X specs
        y genre labels as integers matching format of build_dataset.py
    
    Returns:
        tuple: (predicted_labels, true_labels) each a list of genre strings, 
        in matching order
    """
    data = np.load(test_data_file_path)
    spec = torch.tensor(data["X"])
    label = torch.tensor(data["y"])
    test_loader = DataLoader(TensorDataset(spec, label),batch_size=8, shuffle=False)
    predicted_labels = []
    true_labels = []
    for inputs, labels in test_loader:
        logits = model(inputs)
        predicted_indices = logits.argmax(dim=1)
        for i in range(len(predicted_indices)):
            pred = predicted_indices[i].item()
            true = labels[i].item()
            predicted_labels.append(label_mapping[pred])
            true_labels.append(label_mapping[true])
    
    return predicted_labels, true_labels

