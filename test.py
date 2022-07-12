from collections import OrderedDict
from collections.abc import Iterable
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from clhive.data import SplitCIFAR100
from clhive.scenarios import ClassIncremental, TaskIncremental
from clhive import Trainer, ReplayBuffer


class ReplayBuffer(nn.Module):
    def __init__(self, capacity: int):
        super(ReplayBuffer, self).__init__()

        self.capacity = capacity
        self.current_index = 0
        self.n_seen_so_far = 0

        self.registered_buffers = []

        # defaults
        self.add = self.add_reservoir
        self.sample = self.sample_random

        # self.register_buffer(
        #     name="bx",
        #     tensor=torch.empty(size=(100, 3, 32 ,32,), dtype=torch.float32)
        # )
        # self.register_buffer(
        #     name="by",
        #     tensor=torch.empty(size=(100,), dtype=torch.int64)
        # )
        # self.register_buffer(
        #     name="bt",
        #     tensor=torch.empty(size=(100,), dtype=torch.int64)
        # )
        self.add_buffer(name="x", dtype=torch.float32, size=(100, 3, 32, 32))
        self.add_buffer(name="y", dtype=torch.int64, size=(100,))
        self.add_buffer(name="t", dtype=torch.int64, size=(100,))

    def __len__(self):
        return self.current_index

    def add_buffer(self, name: str, dtype: torch.dtype, size: int) -> None:
        """ used to add extra containers (e.g. for logit storage) """

        self.register_buffer(
            name=f"b{name}",
            tensor=torch.empty(size=(self.capacity,) + size, dtype=dtype),
        )
        self.registered_buffers += [f"b{name}"]

    def _create_buffers(self, batch):
        created = 0

        for name, tensor in batch.items():
            bname = f"b{name}"
            if bname not in self.registered_buffers:

                if not type(tensor) == torch.Tensor:
                    tensor = torch.from_numpy(np.array([tensor]))

                self.add_buffer(name, tensor.dtype, tensor.shape[1:])
                created += 1

                if self.capacity > 0:  # Print buffer created only if mem_size > 0
                    print(f"created buffer {name}\t {tensor.dtype}, {tensor.shape[1:]}")

        assert created in [0, len(batch)], "not all buffers created at the same time"

    def add_reservoir(self, batch):
        self._create_buffers(batch)

        n_elem = batch["x"].size(0)

        place_left = max(0, self.capacity - self.current_index)

        indices = torch.FloatTensor(n_elem)  # .to(self.device)
        indices = indices.uniform_(0, self.n_seen_so_far).long()

        if place_left > 0:
            upper_bound = min(place_left, n_elem)
            indices[:upper_bound] = torch.arange(upper_bound) + self.current_index

        valid_indices = (indices < self.capacity).long()
        idx_new_data = valid_indices.nonzero().squeeze(-1)
        idx_buffer = indices[idx_new_data]

        self.n_seen_so_far += n_elem
        self.current_index = min(self.n_seen_so_far, self.capacity)

        if idx_buffer.numel() == 0:
            return

        # perform overwrite op
        for name, data in batch.items():
            buffer = getattr(self, f"b{name}")

            if isinstance(data, Iterable):
                buffer[idx_buffer] = data[idx_new_data]
            else:
                buffer[idx_buffer] = data

    def sample_random(self, amt, task_id=None, exclude_task=None, **kwargs):
        buffers = OrderedDict()

        if exclude_task is not None:
            assert hasattr(self, "bt")
            valid_indices = torch.where(self.bt != exclude_task)[0]
            valid_indices = valid_indices[valid_indices < self.current_index]
            for buffer_name in self.registered_buffers:
                buffers[buffer_name[1:]] = getattr(self, buffer_name)[valid_indices]
        elif task_id is not None:
            valid_indices = torch.where(self.bt == task_id)[0]
            valid_indices = valid_indices[valid_indices < self.current_index]
            for buffer_name in self.registered_buffers:
                buffers[buffer_name[1:]] = getattr(self, buffer_name)[valid_indices]
        else:
            for buffer_name in self.registered_buffers:
                buffers[buffer_name[1:]] = getattr(self, buffer_name)[
                    : self.current_index
                ]

        n_selected = buffers["x"].size(0)
        if n_selected <= amt:
            assert n_selected > 0
            return buffers
        else:
            idx_np = np.random.choice(buffers["x"].size(0), amt, replace=False)
            indices = torch.from_numpy(idx_np)  # .to(self.bx.device)

            return OrderedDict({k: v[indices] for (k, v) in buffers.items()})


dataset = SplitCIFAR100(root="../cl-datasets/data/")
scenario = ClassIncremental(dataset=dataset, n_tasks=10, batch_size=128, n_workers=6)

print(f"Number of tasks: {scenario.n_tasks} | Number of classes: {scenario.n_classes}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

buffer = ReplayBuffer(capacity=20).to(device)

for task_id, train_loader in enumerate(scenario):
    for x, y, t in train_loader:
        x, y = x.to(device), y.to(device)

        # Do your cool stuff here
        if len(buffer) > 0:
            re_data = buffer.sample(amt=20)
            print(
                f"Replay Data --> x: {re_data['x'].shape} y: {re_data['y'].shape} Device: {buffer.bx.device}",
                end="\r",
            )

        buffer.add({"t": task_id, "x": x, "y": y})
