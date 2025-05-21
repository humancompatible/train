from typing import Callable, Iterable

from torch.nn import Module
from torch.utils.data import Dataset

from src.algorithms.c_utils.constraint import FairnessConstraint


class Algorithm:
    def __init__(
        self,
        net: Module,
        data: Dataset,
        loss: Callable,
        constraints: Iterable[FairnessConstraint],
    ):
        self.net = net
        self.constraints = constraints
        self.loss_fn = loss
        self.dataset = data

        self.history = {"loss": [], "constr": [], "w": [], "time": [], "n_samples": []}

    def optimize(self, max_runtime: float = None, max_iter: int = None):
        pass
