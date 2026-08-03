# Written with assistance from Claude Code (Sonnet 5)

import torch
from torch import nn
from torchsummary import summary

# Shapes follow docs/tensor-contract.md. Base pattern adapted from Valerio
# Velardo's "PyTorch for Audio + Music Processing" series (Lesson 08,
# CNNNetwork), changed to 3 blocks, padding=1, raw logits, no softmax:
# https://github.com/musikalkemist/pytorchforaudio/blob/main/08%20Implementing%20a%20CNN%20network/cnn.py

# GAP replaces flatten to fix overfitting seen in early experiments (see
# experiments/): flatten fed 16,384 features into the final Linear (163,850
# of ~187k params). GAP shrinks that to 64 features, cutting the Linear to
# 650 params.


class GenreCNN(nn.Module):
    """CNN genre classifier. See docs/tensor-contract.md for I/O shapes."""

    def __init__(self, dropout_rate: float = 0.5, use_batchnorm: bool = False) -> None:
        """
        Build the 3-block conv, global average pool, linear architecture.

        Args:
            dropout_rate (float): Dropout probability before the final
                Linear. Defaults to 0.5.
            use_batchnorm (bool): Add BatchNorm2d after each Conv2d.
                Defaults to False.

        Returns:
            None
        """
        super().__init__()

        # Optional BatchNorm2d per conv block: stabilizes/speeds up training
        # and mildly regularizes, distinct from dropout. Off by default so
        # it doesn't change existing experiments.
        def conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
            """
            Build one Conv2d, optional BatchNorm2d, ReLU, MaxPool2d block.

            Args:
                in_channels (int): Input channels.
                out_channels (int): Output channels.

            Returns:
                nn.Sequential: The assembled block.
            """
            layers = [
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            ]
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers += [nn.ReLU(), nn.MaxPool2d(kernel_size=2)]
            return nn.Sequential(*layers)

        # 3 conv blocks: 1 -> 16 -> 32 -> 64 channels, each halving H/W.
        self.conv1 = conv_block(1, 16)
        self.conv2 = conv_block(16, 32)
        self.conv3 = conv_block(32, 64)

        # GAP averages each of conv3's 64 feature maps to one value
        # (16x16 -> 1x1), replacing flatten's 16,384-dim output.
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()

        self.linear = nn.Linear(64, 10)  # 64 pooled channels -> 10 genres

        # Zeroes activations during training only (no-op in eval mode).
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """
        Run the forward pass.

        Args:
            input_data (torch.Tensor): Input batch, shape (N, 1, 128, 130),
                float32.

        Returns:
            torch.Tensor: Raw logits, shape (N, 10); softmax happens in
                the CLI, not here.
        """
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

    # summary() only prints; the assert above is the real check. input_size
    # excludes the batch dim (torchsummary adds its own batch of 2).
    summary(cnn, (1, 128, 130))
