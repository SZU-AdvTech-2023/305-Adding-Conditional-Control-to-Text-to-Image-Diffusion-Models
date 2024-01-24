import importlib
from torch import nn

def disabled_train(self: nn.Module) -> nn.Module:
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


def frozen_module(module: nn.Module) -> None:
    module.eval()
    module.train = disabled_train
    for p in module.parameters():
        p.requires_grad = False

def unfrozen_module(module: nn.Module) -> None:
    module.train()
    for p in module.parameters():
        p.requires_grad = True