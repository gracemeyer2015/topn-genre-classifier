import torch
from build_dataset import GENRES
from evaluate.song_level import build_model_from_config
from pathlib import Path


def load_model(PATH_TO_MODEL=None):
    """
    Load the model and its state dictionary from specificed checkpoint path,
    reading the checkpoint's config.json to build the right architecture
    (dropout_rate, use_batchnorm)

    Temporary stub version: no real model exists cannot yet use torch.load or
    model.load_state_dict this content is commented out

    Args:
        PATH_TO_MODEL (str): Path to the checkpoint.pt file
            The checkpoint's config is found within the same folder

    Returns: Tuple containing:
        model: The loaded model with the state dictionary applied.
        label_mapping: A dictionary mapping class indices to genre labels.
    """
    if PATH_TO_MODEL is None:
        raise ValueError("load_model requires a checkpoint path")

    config_path = Path(PATH_TO_MODEL).parent/"config.json"
    model = build_model_from_config(config_path)
    checkpoint = torch.load(PATH_TO_MODEL)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    index_to_genre = {i: genre for i, genre in enumerate(GENRES)}

    return model, index_to_genre


def predict_genre(model, tensors, label_mapping, top_n=5):
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
    all_probabilities = []
    for tensor in tensors:
        output = model(tensor)
        # convert to probabilites
        probabilities = torch.softmax(output, dim=1)
        all_probabilities.append(probabilities)

    stacked = torch.stack(all_probabilities)
    avg_probabilities = stacked.mean(dim=0)

    top_prob, top_indices = torch.topk(avg_probabilities, top_n)

    results = []
    for i in range(len(top_prob[0])):
        prob = top_prob[0][i].item()
        indx = top_indices[0][i].item()
        genre = label_mapping[indx]

        results.append((genre, prob*100))

    return results
