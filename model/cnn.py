import torch
from torch import nn

# Input/output shapes below follow docs/tensor-contract.md.
#
# Base pattern adapted from Valerio Velardo's "PyTorch for Audio + Music
# Processing" series (Lesson 08, CNNNetwork):
#
# https://github.com/musikalkemist/pytorchforaudio/blob/main/08%20Implementing%20a%20CNN%20network/cnn.py
#
# Changed: 3 blocks not 4 (plan spec), padding=1 not 2 (predictable shapes),
# raw logits not softmax (CrossEntropyLoss applies log-softmax internally).
#
# TODO: no hidden layer before the final Linear, unlike PyTorch's "Learn the
# Basics" tutorial:
#
# https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
#
# GTZAN is small; likely need a hidden layer + dropout if validation
# curves show overfitting.


class GenreCNN(nn.Module):
    """CNN genre classifier. See docs/tensor-contract.md for I/O shapes."""

    def __init__(self) -> None:
        super().__init__()
        # block 1: 1 input channel (mel spectrogram) -> 16 feature maps
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        # block 2: 16 -> 32 channels, spatial dims halved again
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        # block 3: 32 -> 64 channels, spatial dims halved again
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.flatten = nn.Flatten()
        # tensor contract: (N, 1, 128, 130) Each block halves H and W
        # (128->64->32->16, 130->65->32->16) = 64 x 16 x 16.
        self.linear = nn.Linear(64 * 16 * 16, 10)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        logits = self.linear(x)
        return logits


if __name__ == "__main__":
    cnn = GenreCNN()
    batch = torch.randn(4, 1, 128, 130, dtype=torch.float32)
    out = cnn(batch)
    assert out.shape == (4, 10), f"expected (4, 10), got {tuple(out.shape)}"
    n_params = sum(p.numel() for p in cnn.parameters())
    print(f"GenreCNN OK - output shape {tuple(out.shape)}, {n_params:,} parameters")
