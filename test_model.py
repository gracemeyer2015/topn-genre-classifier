import torch

from model.cnn import GenreCNN


def test_forward_pass_shape():
    model = GenreCNN()
    batch = torch.randn(4, 1, 128, 130, dtype=torch.float32)
    out = model(batch)
    assert out.shape == (4, 10)
