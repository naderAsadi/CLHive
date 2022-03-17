from typing import Any, Dict, List, Optional, Tuple
from copy import deepcopy

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from torchcl.data.datasets import get_dataset
from torchcl.data.datasets.base import BaseDataset
from torchcl.data.transforms import BaseTransform, get_transform
from torchcl.data.samplers import ContinualSampler


def make_val_from_train(dataset: BaseDataset, split: float = 0.9):
    """Create a validation set from training set.

    Args:
        dataset ([BaseDataset]): [description]
        split (float, optional): [description]. Defaults to .9.

    Returns:
        [type]: [description]
    """

    train_ds, val_ds = deepcopy(dataset), deepcopy(dataset)

    train_idx, val_idx = [], []
    for label in np.unique(dataset.targets):
        label_idx = np.squeeze(np.argwhere(dataset.targets == label))
        split_idx = int(label_idx.shape[0] * split)
        train_idx += [label_idx[:split_idx]]
        val_idx   += [label_idx[split_idx:]]

    train_idx = np.concatenate(train_idx)
    val_idx   = np.concatenate(val_idx)

    train_ds = Subset(dataset, train_idx)
    val_ds   = Subset(dataset, val_idx)

    return train_ds, val_ds


def get_loaders_and_transforms(config: Dict[str, Any]):
    """[summary]

    Args:
        config (Dict[str, Any]): [description]

    Returns:
        [type]: [description]
    """

    base_transform = BaseTransform()
    train_transform = get_transform(transform_config=config.data.transform)

    val_set = val_loader = None
    train_set = get_dataset(config.data, train = True)
    test_set = get_dataset(config.data, train = False)
    # if config.validation:
    #     train_set, val_set = make_val_from_train(trainval_ds)

    train_sampler = ContinualSampler(dataset = train_set, n_tasks = config.data.n_tasks)
    train_loader  = DataLoader(
        train_set,
        num_workers=config.data.n_workers,
        sampler=train_sampler,
        batch_size=config.train.batch_size,
        pin_memory=True
    )

    test_sampler  = ContinualSampler(dataset = test_set,  n_tasks = config.data.n_tasks)
    test_loader = DataLoader(
        test_set,
        num_workers=config.data.n_workers,
        batch_size=config.eval.batch_size,
        sampler=test_sampler,
        pin_memory=True
    )

    if val_set is not None:
        val_sampler = ContinualSampler(dataset = val_set, n_tasks = config.data.n_tasks)
        val_loader  = DataLoader(
            val_set,
            num_workers=config.data.n_workers,
            batch_size=config.eval.batch_size,
            sampler=val_sampler,
            pin_memory=True
        )
    
    if config.data.n_tasks == -1:
        config.data.n_tasks = train_set._DEFAULT_N_TASKS
    config.data.image_size = (3, train_set._IMAGE_SIZE, train_set._IMAGE_SIZE)
    config.data.n_classes = train_sampler.n_classes
    config.data.n_classes_per_task = config.data.n_classes // config.data.n_tasks

    return train_transform, train_loader, val_loader, test_loader
