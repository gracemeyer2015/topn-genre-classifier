import torch
from torch import nn
from torchsummary import summary

# Input/output shapes below follow docs/tensor-contract.md.
#
# Base pattern adapted from Valerio Velardo's "PyTorch for Audio + Music
# Processing" series (Lesson 08, CNNNetwork):
#
# https://github.com/musikalkemist/pytorchforaudio/blob/main/08%20Implementing%20a%20CNN%20network/cnn.py
#
# Changed: 3 blocks, padding=1, raw logits - no softmax

#
# Status: validation curves showed clear overfitting on GTZAN (train_acc ->
# ~99%, val_loss turning upward after epoch 2-3, val_acc stuck ~70% See
# experiments/). Dropout alone (experiments/*_dropout-0.5) helped but didn't
# close the train/val gap. The final Linear still had 163,850 of the
# model's ~187k total parameters (flatten was 64*16*16=16,384-dim). Now
# replacing flatten with global average pooling to actually shrink that
# layer (16,384 -> 64 features -> Linear(64,10) = 650 params) instead of
# just making the oversized layer harder to use via dropout alone.


class GenreCNN(nn.Module):
    """CNN genre classifier. See docs/tensor-contract.md for I/O shapes."""

    def __init__(self, dropout_rate: float = 0.5) -> None:
        super().__init__()
        # 3 conv blocks -> flatten -> linear
        # block 1: 1 input channel (mel spectrogram) -> 16 feature maps
        # Sequential = containter: PyTorch will process the layers sequentially
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
        # Global average pooling: average each of conv3's 64 channel feature
        # maps down to a single value, regardless of their spatial size
        # (16x16 per the tensor contract). Output is (N, 64, 1, 1) This replaces
        # flattening the full 64*16*16=16,384-dim map, which was the
        # overwhelming majority of the model's parameters once fed into a
        # Linear (163,850 of ~187k total.
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()

        # Dense layer: 64 pooled channel averages -> 10 genre logits.
        self.linear = nn.Linear(64, 10)

        # Randomly zeroes activations during training only (no-op in eval
        # mode) - extra regularization on top of the now much smaller Linear.
        self.dropout = nn.Dropout(p=dropout_rate)

    # Define method to pass data from one layer to the next
    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.global_avg_pool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        logits = self.linear(x)
        return logits


if __name__ == "__main__":
    cnn = GenreCNN()
    batch = torch.randn(4, 1, 128, 130, dtype=torch.float32)
    out = cnn(batch)
    assert out.shape == (4, 10), f"expected (4, 10), got {tuple(out.shape)}"
    n_params = sum(p.numel() for p in cnn.parameters())
    print(f"GenreCNN OK - output shape {tuple(out.shape)}, {n_params:,} parameters")

    # Layer-by-layer breakdown (output shape + params per layer), additive to
    # the assert above - summary() only prints, it doesn't verify anything.
    # input_size excludes the batch dim; torchsummary adds its own batch of 2.
    summary(cnn, (1, 128, 130))
