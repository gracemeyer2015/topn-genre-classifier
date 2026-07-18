import torch
from cli.dummyModel import DummyModel  # Replace with your actual model class


def load_model(PATH_TO_MODEL=None):
    """
    Load the model and its state dictionary from specificed path.

    Temporary stub version: no real model exists cannot yet use torch.load or
    model.load_state_dict this content is commented out

    Args:
        state_dict (dict): The state dictionary of the model.
        PATH_TO_MODEL (str): Path to the model file.

    Returns: Tuple containing:
        model: The loaded model with the state dictionary applied.
        label_mapping: A dictionary mapping class indices to genre labels provided by data pipeline
    """
    # loading the model using state_dict
    # model_contents = torch.load(PATH_TO_MODEL)
    model = DummyModel()  # Replace with your actual model class
    # model.load_state_dict(model_contents['model_state_dict'])
    model.eval()   # Set the model to evaluation mode

    # explicitly written for now loaded from data pipeline (hot one encoded) later
    label_mapping = {0: "blues", 1: "classical", 2: "country", 3: "disco", 4:
                     "hiphop", 5: "jazz", 6: "metal", 7: "pop",
                     8: "reggae", 9: "rock"}

    return model, label_mapping


def predict_genre(model, tensor, label_mapping, top_n=5):
    """
    Uses the model to predict the genre of a preprocessed audio tensor and
    returns printed top-n closest music genre matches ranked by confidence.

    Args:
        model: The model and its state dictionary.
        tensor: The preprocessed audio tensor.
        label_mapping: A dictionary mapping class indices to genre labels.
        top_n (int): The number of top genre matches to return (default: 5).

    Returns:
        None: Prints the top-n closest music genre matches ranked by confidence.
    """
    output = model(tensor)
    # convert to probabilites
    probabilities = torch.softmax(output, dim=1)
    top_prob, top_indices = torch.topk(probabilities, top_n)

    results = []
    for i in range(len(top_prob[0])):
        prob = top_prob[0][i].item()
        indx = top_indices[0][i].item()
        genre = label_mapping[indx]

        results.append((genre, prob*100))

    return results
