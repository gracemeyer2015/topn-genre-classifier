import torch.nn as nn


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(1*128*130, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        return x
