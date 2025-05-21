from itertools import cycle
from typing import Callable, Iterable

import numpy as np
import torch


def _dataloader_from_subset(dataset, indices, *args, **kwargs):
    data_s = torch.utils.data.Subset(dataset, indices)
    loader_s = torch.utils.data.DataLoader(data_s, *args, **kwargs)
    return loader_s


class FairnessConstraint:
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        group_indices: Iterable[Iterable[int]],
        fn: Callable,
        batch_size: int = None,
        use_dataloaders=True,
        seed=None,
    ):
        self.dataset = dataset
        self.group_sets = [
            torch.utils.data.Subset(dataset, idx) for idx in group_indices
        ]
        self.fn = fn
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        if batch_size is not None:
            self.batch_size = batch_size
            if use_dataloaders:
                g = torch.Generator()
                if seed is not None:
                    g.manual_seed(seed)
                self.dataloaders = [
                    cycle(
                        _dataloader_from_subset(
                            dataset,
                            idx,
                            batch_size=batch_size,
                            shuffle=True,
                            generator=g,
                        )
                    )
                    for idx in group_indices
                ]

    def group_sizes(self):
        return [len(group) for group in self.group_sets]

    def eval(self, net, sample, **kwargs):
        return self.fn(net, sample, **kwargs)

    def sample_loader(self):
        return [next(l) for l in self.dataloaders]

    def sample_dataset(
        self, N, rng: np.random.Generator = None, indices=None, return_indices=False
    ):
        if rng is None:
            rng = self.rng

        if indices is None:
            indices = []
            # returns len(group) points if N > len(group)
            for group in self.group_sets:
                indices.append(
                    rng.choice(group.indices, N)
                    if N < len(group)
                    else rng.choice(group.indices, len(group))
                )

        sample = [self.dataset[indices[i]] for i, _ in enumerate(self.group_sets)]

        if return_indices:
            return sample, indices
        else:
            return sample
