from torch import nn


class SimpleNet(nn.Module):
    def __init__(self, in_shape, out_shape, dtype):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(in_shape, 64, dtype=dtype),
            nn.ReLU(),
            nn.Linear(64, 32, dtype=dtype),
            nn.ReLU(),
            nn.Linear(32, out_shape, dtype=dtype),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits
