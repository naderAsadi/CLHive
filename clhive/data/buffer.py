from typing import Dict, Optional, Union
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
from collections.abc import Iterable

from ..scenarios import ClassIncremental, TaskIncremental
from ..utils.generic import *


class ReplayBuffer(nn.Module):
    def __init__(
        self, capacity: int, input_size: int, input_n_channels: int,
    ) -> "ReplayBuffer":
        super(ReplayBuffer, self).__init__()

        self.capacity = capacity
        self.current_index = 0
        self.n_seen_so_far = 0

        self.registered_buffers = ["data_buffer", "targets_buffer", "task_ids_buffer"]

        shape = (self.capacity, input_n_channels, input_size, input_size)
        self._create_buffers(
            batch={
                "data_buffer": torch.empty(*shape, dtype=torch.float32),
                "targets_buffer": torch.empty(1, dtype=torch.int64),
                "task_ids_buffer": torch.empty(1, dtype=torch.int64),
            }
        )

        # defaults
        self.add = self.add_reservoir
        self.sample = self.sample_random

    @property
    def num_bits(self) -> int:
        total = 0

        for name in self.registered_buffers:
            buffer = getattr(self, name)

            if buffer.dtype == torch.float32:
                bits_per_item = 8 if name == "data_buffer" else 32
            elif buffer.dtype == torch.int64:
                bits_per_item = buffer.max().float().log2().clamp_(min=1).int().item()

            total += bits_per_item * buffer.numel()

        return total

    def __len__(self) -> int:
        return self.current_index

    def _get_batch(
        self,
        data: torch.FloatTensor,
        targets: torch.FloatTensor,
        task_ids: torch.FloatTensor,
    ) -> Dict[str, torch.FloatTensor]:
        return {
            "data_buffer": data,
            "targets_buffer": targets,
            "task_ids_buffer": task_ids,
        }

    def _create_buffers(self, batch: Dict[str, torch.FloatTensor]) -> None:
        created = 0

        for name, tensor in batch.items():
            if not type(tensor) == torch.Tensor:
                tensor = torch.from_numpy(np.array([tensor]))

            self.add_buffer(name, tensor.dtype, tensor.shape[1:])
            created += 1

            if self.capacity > 0:  # Print buffer created only if mem_size > 0
                print(f"created buffer {name}\t {tensor.dtype}, {tensor.shape[1:]}")

        assert created in [0, len(batch)], "not all buffers created at the same time"

    def add_buffer(self, name: str, dtype: torch.dtype, size: int) -> None:
        """ used to add extra containers (e.g. for logit storage) """

        self.register_buffer(
            name=name, tensor=torch.zeros(size=(self.capacity,) + size, dtype=dtype),
        )

    def add_reservoir(
        self,
        data: torch.FloatTensor,
        targets: torch.FloatTensor,
        task_ids: torch.FloatTensor,
    ) -> None:
        batch = self._get_batch(data, targets, task_ids)

        n_elem = batch["data_buffer"].size(0)

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

        # perform overwrite operation
        for name, data in batch.items():
            buffer = getattr(self, name)

            if isinstance(data, Iterable):
                buffer[idx_buffer] = data[idx_new_data]
            else:
                buffer[idx_buffer] = data

    def add_balanced(
        self,
        data: torch.FloatTensor,
        targets: torch.FloatTensor,
        task_ids: torch.FloatTensor,
    ) -> None:
        batch = self._get_batch(data, targets, task_ids)

        n_elem = batch["data_buffer"].size(0)

        # increment first
        self.n_seen_so_far += n_elem
        self.current_index = min(self.n_seen_so_far, self.capacity)

        # first thing is we just add all the data
        for name, data in batch.items():
            buffer = getattr(self, name)

            if not isinstance(data, Iterable):
                data = buffer.new(size=(n_elem, *buffer.shape[1:])).fill_(data)

            buffer = torch.cat((data, buffer))[: self.n_seen_so_far]
            setattr(self, name, buffer)

        n_samples_over = buffer.size(0) - self.capacity

        # no samples to remove
        if n_samples_over <= 0:
            return

        # remove samples from the most common classes
        class_count = self.by.bincount()
        rem_per_class = torch.zeros_like(class_count)

        while rem_per_class.sum() < n_samples_over:
            max_idx = class_count.argmax()
            rem_per_class[max_idx] += 1
            class_count[max_idx] -= 1

        # always remove the oldest samples for each class
        classes_trimmed = rem_per_class.nonzero().flatten()
        idx_remove = []

        for cls in classes_trimmed:
            cls_idx = (self.by == cls).nonzero().view(-1)
            idx_remove += [cls_idx[-rem_per_class[cls] :]]

        idx_remove = torch.cat(idx_remove)
        idx_mask = torch.BoolTensor(buffer.size(0))  # .to(self.device)
        idx_mask.fill_(0)
        idx_mask[idx_remove] = 1

        # perform overwrite op
        for name, data in batch.items():
            buffer = getattr(self, name)
            buffer = buffer[~idx_mask]
            setattr(self, name, buffer)

    def add_queue(
        self,
        data: torch.FloatTensor,
        targets: torch.FloatTensor,
        task_ids: torch.FloatTensor,
    ) -> None:
        batch = self._get_batch(data, targets, task_ids)

        if not hasattr(self, "queue_ptr"):
            self.queue_ptr = 0

        start_idx = self.queue_ptr
        end_idx = (start_idx + batch["data_buffer"].size(0)) % self.capacity

        for name, data in batch.items():
            buffer = getattr(self, name)
            buffer[start_idx:end_idx] = data

    def sample_random(
        self,
        n_samples: int,
        task_id: Optional[int] = None,
        exclude_task_id: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, torch.FloatTensor]:
        buffers = OrderedDict()

        if exclude_task_id is not None:
            assert hasattr(self, "task_ids_buffer")

            valid_indices = torch.where(
                getattr(self, "task_ids_buffer") != exclude_task_id
            )[0]
            valid_indices = valid_indices[valid_indices < self.current_index]
            for buffer_name in self.registered_buffers:
                buffers[buffer_name] = getattr(self, buffer_name)[valid_indices]

        elif task_id is not None:
            valid_indices = torch.where(getattr(self, "task_ids_buffer") == task_id)[0]
            valid_indices = valid_indices[valid_indices < self.current_index]
            for buffer_name in self.registered_buffers:
                buffers[buffer_name] = getattr(self, buffer_name)[valid_indices]

        else:
            for buffer_name in self.registered_buffers:
                buffers[buffer_name] = getattr(self, buffer_name)[: self.current_index]

        n_selected = buffers["data_buffer"].size(0)
        if n_selected <= n_samples:
            assert n_selected > 0
            return buffers
        else:
            idx_np = np.random.choice(
                buffers["data_buffer"].size(0), n_samples, replace=False
            )
            indices = torch.from_numpy(idx_np)  # .to(self.bx.device)

            return OrderedDict({k: v[indices] for (k, v) in buffers.items()})

    def sample_balanced(
        self,
        n_samples: int,
        task_id: Optional[int] = None,
        exclude_task_id: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, torch.FloatTensor]:
        buffers = OrderedDict()

        if exclude_task_id is not None:
            assert hasattr(self, "task_ids_buffer")
            valid_indices = (
                (getattr(self, "task_ids_buffer") != exclude_task_id)
                .nonzero()
                .squeeze()
            )
            for buffer_name in self.registered_buffers:
                buffers[buffer_name] = getattr(self, buffer_name)[valid_indices]
        else:
            for buffer_name in self.registered_buffers:
                buffers[buffer_name] = getattr(self, buffer_name)[: self.current_index]

        class_count = buffers["targets_buffer"].bincount()

        # a sample's prob. of being sample is inv. prop to its class abundance
        class_sample_p = 1.0 / class_count.float() / class_count.size(0)
        per_sample_p = class_sample_p.gather(0, buffers["targets_buffer"])
        indices = torch.multinomial(per_sample_p, n_samples)

        return OrderedDict({k: v[indices] for (k, v) in buffers.items()})

    def sample_pos_neg(
        self,
        data: torch.FloatTensor,
        targets: torch.FloatTensor,
        task_ids: torch.FloatTensor,
        task_free: Optional[bool] = True,
        same_task_neg: Optional[bool] = True,
    ) -> Dict[str, torch.FloatTensor]:
        x = data
        label = targets
        task = task_ids

        # we need to create an "augmented" buffer containing the incoming data
        bx = torch.cat((getattr(self, "data_buffer")[: self.current_index], x))
        by = torch.cat((getattr(self, "targets_buffer")[: self.current_index], label))
        bt = torch.cat((getattr(self, "task_ids_buffer"), task))
        bidx = torch.arange(bx.size(0))  # .to(bx.device)

        # buf_size x label_size
        same_label = label.view(1, -1) == by.view(-1, 1)
        same_task = task.view(1, -1) == bt.view(-1, 1)
        same_ex = bidx[-x.size(0) :].view(1, -1) == bidx.view(-1, 1)

        task_labels = label.unique()
        real_same_task = same_task

        # TASK FREE METHOD : instead of using the task ID, we'll use labels in
        # the current batch to mimic task
        if task_free:
            same_task = torch.zeros_like(same_task)

            for label_ in task_labels:
                label_exp = label_.view(1, -1).expand_as(same_task)
                same_task = same_task | (label_exp == by.view(-1, 1))

        valid_pos = same_label & ~same_ex

        if same_task_neg:
            valid_neg = ~same_label & same_task
        else:
            valid_neg = ~same_label

        # remove points which don't have pos, neg from same and diff t
        has_valid_pos = valid_pos.sum(0) > 0
        has_valid_neg = valid_neg.sum(0) > 0

        invalid_idx = ~has_valid_pos | ~has_valid_neg

        if invalid_idx.sum() > 0:
            # so the fetching operation won't fail
            valid_pos[:, invalid_idx] = 1
            valid_neg[:, invalid_idx] = 1

        # easier if invalid_idx is a binary tensor
        is_invalid = torch.zeros_like(label).bool()
        is_invalid[invalid_idx] = 1

        # fetch positive samples
        pos_idx = torch.multinomial(valid_pos.float().T, 1).squeeze(1)
        neg_idx = torch.multinomial(valid_neg.float().T, 1).squeeze(1)

        # n_fwd = torch.stack((pos_idx, neg_idx), 1)[~invalid_idx].unique().size(0)

        return {
            "data_pos": bx[pos_idx],
            "data_neg": bx[neg_idx],
            "targets_pos": by[pos_idx],
            "targets_neg": by[neg_idx],
        }
