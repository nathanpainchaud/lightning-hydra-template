# Script to reproduce an issue with using `self.log` in compiled PyTorch Lightning module
# This minimal example reproduces the issue, whose root cause is an interaction between dynamo and torchmetrics
# Link to the issue: https://github.com/Lightning-AI/pytorch-lightning/issues/18123

# Disable ruff entirely for this file
# flake8: noqa

import torch
import torch.nn as nn
from torchmetrics import Accuracy


class DummyModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(2, 2)

    def forward(self, x):
        # y = DummyModule() # <-- replacing the torchmetric with a dummy module fixes the issue
        y = Accuracy(task="binary")  # <-- this fails
        return self.layer(x)


def overwrite_torch_functions():
    module_set_attr_orig = torch.nn.Module.__setattr__

    def wrap_set_attr(self, name, value):
        if isinstance(value, torch.nn.Module):
            print(value)  # <-- calls `__repr__` on the module
        module_set_attr_orig(self, name, value)

    torch.nn.Module.__setattr__ = wrap_set_attr


if __name__ == "__main__":
    overwrite_torch_functions()

    model = Model()
    model = torch.compile(model)
    model(torch.rand(2, 2))
