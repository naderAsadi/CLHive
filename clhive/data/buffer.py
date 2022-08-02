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
    """_summary_

    Args:
        nn (_type_): _description_
    """

    def __init__(
        self, capacity: int, device: Optional[torch.device] = None,
    ) -> "ReplayBuffer":
        """_summary_

        Args:
            capacity (int): _description_
            device (Optional[torch.device], optional): _description_. Defaults to None.

        Returns:
            ReplayBuffer: _description_
        """

        super(ReplayBuffer, self).__init__()

        self.registered_buffers = []

        self.capacity = capacity
        self.current_index = 0
        self.n_seen_so_far = 0

        if device is None:
            device = torch.device("cpu")
        self.device = device

        # defaults
        self.add = self.add_reservoir
        self.sample = self.sample_random

    @property
    def num_bits(self) -> int:
        total = 0

        for name in self.buffers:
            buffer = getattr(self, name)

            if buffer.dtype == torch.float32:
                bits_per_item = 8 if name == "x" else 32
            elif buffer.dtype == torch.int64:
                bits_per_item = buffer.max().float().log2().clamp_(min=1).int().item()

            total += bits_per_item * buffer.numel()

        return total

    def __len__(self) -> int:
        return self.current_index

    def add_buffer(self, name: str, dtype: torch.dtype, size: int) -> None:
        """Used to add extra containers (e.g. for logit storage)

        Args:
            name (str): _description_
            dtype (torch.dtype): _description_
            size (int): _description_
        """

        tmp = torch.zeros(size=(self.capacity,) + size, dtype=dtype).to(self.device)
        self.register_buffer(name, tmp)
        self.registered_buffers += [name]

    def _create_buffers(self, batch: Dict[str, torch.FloatTensor]) -> None:
        created = 0

        for name, tensor in batch.items():
            if name not in self.registered_buffers:

                if not type(tensor) == torch.Tensor:
                    tensor = torch.from_numpy(np.array([tensor]))

                self.add_buffer(name, tensor.dtype, tensor.shape[1:])
                created += 1

                print(f"created buffer {name}\t {tensor.dtype}, {tensor.shape[1:]}")

        assert created in [0, len(batch)], "not all buffers created at the same time"

    def add_reservoir(self, batch: Dict[str, torch.FloatTensor]) -> None:
        """_summary_

        Args:
            batch (Dict[str, torch.FloatTensor]): _description_
        """

        self._create_buffers(batch)

        n_elem = batch["x"].size(0)

        place_left = max(0, self.capacity - self.current_index)

        indices = torch.FloatTensor(n_elem).to(self.device)
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
            buffer = getattr(self, name)

            if isinstance(data, Iterable):
                buffer[idx_buffer] = data[idx_new_data]
            else:
                buffer[idx_buffer] = data

    def add_balanced(self, batch: Dict[str, torch.FloatTensor]) -> None:
        self._create_buffers(batch)

        n_elem = batch["x"].size(0)

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
        class_count = self.y.bincount()
        rem_per_class = torch.zeros_like(class_count)

        while rem_per_class.sum() < n_samples_over:
            max_idx = class_count.argmax()
            rem_per_class[max_idx] += 1
            class_count[max_idx] -= 1

        # always remove the oldest samples for each class
        classes_trimmed = rem_per_class.nonzero().flatten()
        idx_remove = []

        for cls in classes_trimmed:
            cls_idx = (self.y == cls).nonzero().view(-1)
            idx_remove += [cls_idx[-rem_per_class[cls] :]]

        idx_remove = torch.cat(idx_remove)
        idx_mask = torch.BoolTensor(buffer.size(0)).to(self.device)
        idx_mask.fill_(0)
        idx_mask[idx_remove] = 1

        # perform overwrite op
        for name, data in batch.items():
            buffer = getattr(self, name)
            buffer = buffer[~idx_mask]
            setattr(self, name, buffer)

    def add_queue(self, batch: Dict[str, torch.FloatTensor]) -> None:
        self._create_buffers(batch)

        if not hasattr(self, "queue_ptr"):
            self.queue_ptr = 0

        start_idx = self.queue_ptr
        end_idx = (start_idx + batch["x"].size(0)) % self.capacity

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
        """_summary_

        Args:
            n_samples (int): _description_
            task_id (Optional[int], optional): _description_. Defaults to None.
            exclude_task_id (Optional[int], optional): _description_. Defaults to None.

        Returns:
            Dict[str, torch.FloatTensor]: _description_
        """

        buffers = OrderedDict()

        if task_id is not None:
            assert hasattr(self, "t"), "`t` is not a registered buffer."

            valid_indices = torch.where(getattr(self, "t") == task_id)[0]
            valid_indices = valid_indices[valid_indices < self.current_index]
            for buffer_name in self.registered_buffers:
                buffers[buffer_name] = getattr(self, buffer_name)[valid_indices]

        elif exclude_task_id is not None:
            assert hasattr(self, "t"), "`t` is not a registered buffer."

            valid_indices = torch.where(self.t != exclude_task_id)[0]
            valid_indices = valid_indices[valid_indices < self.current_index]
            for buffer_name in self.registered_buffers:
                buffers[buffer_name] = getattr(self, buffer_name)[valid_indices]

        else:
            for buffer_name in self.registered_buffers:
                buffers[buffer_name] = getattr(self, buffer_name)[: self.current_index]

        n_selected = buffers["x"].size(0)
        if n_selected <= n_samples:
            assert n_selected > 0
            return buffers
        else:
            idx_np = np.random.choice(buffers["x"].size(0), n_samples, replace=False)
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

        if task_id is not None:
            assert hasattr(self, "t"), "`t` is not a registered buffer."

            valid_indices = (getattr(self, "t") == task_id).nonzero().squeeze()
            for buffer_name in self.registered_buffers:
                buffers[buffer_name] = getattr(self, buffer_name)[valid_indices]

        elif exclude_task_id is not None:
            assert hasattr(self, "t"), "`t` is not a registered buffer."

            valid_indices = (getattr(self, "t") != exclude_task_id).nonzero().squeeze()
            for buffer_name in self.registered_buffers:
                buffers[buffer_name] = getattr(self, buffer_name)[valid_indices]

        else:
            for buffer_name in self.registered_buffers:
                buffers[buffer_name] = getattr(self, buffer_name)[: self.current_index]

        class_count = buffers["y"].bincount()

        # a sample's prob. of being sample is inv. prop to its class abundance
        class_sample_p = 1.0 / class_count.float() / class_count.size(0)
        per_sample_p = class_sample_p.gather(0, buffers["y"])
        indices = torch.multinomial(per_sample_p, n_samples)

        return OrderedDict({k: v[indices] for (k, v) in buffers.items()})

    def sample_mir(
        self,
        n_samples: int,
        subsample: int,
        model: torch.nn.Module,
        exclude_task_id: Optional[int] = None,
        lr: Optional[float] = 0.1,
        head_only: Optional[bool] = False,
        **kwargs,
    ) -> Dict[str, torch.FloatTensor]:
        subsample = self.sample_random(subsample, exclude_task_id=exclude_task_id)

        if not hasattr(model, "grad_dims"):
            model.mir_grad_dims = []
            if head_only:
                for param in model.linear.parameters():
                    model.mir_grad_dims += [param.data.numel()]
            else:
                for param in model.parameters():
                    model.mir_grad_dims += [param.data.numel()]

        if head_only:
            grad_vector = get_grad_vector(
                list(model.linear.parameters()), model.mir_grad_dims
            )
            model_temp = get_future_step_parameters(
                model.linear, grad_vector, model.mir_grad_dims, lr=lr
            )
        else:
            grad_vector = get_grad_vector(list(model.parameters()), model.mir_grad_dims)
            model_temp = get_future_step_parameters(
                model, grad_vector, model.mir_grad_dims, lr=lr
            )

        with torch.no_grad():
            hidden_pre = model.return_hidden(subsample["x"])
            logits_pre = model.linear(hidden_pre)

            if head_only:
                logits_post = model_temp(hidden_pre)
            else:
                logits_post = model_temp(subsample["x"])

            pre_loss = F.cross_entropy(logits_pre, subsample["y"], reduction="none")
            post_loss = F.cross_entropy(logits_post, subsample["y"], reduction="none")

            scores = post_loss - pre_loss
            indices = scores.sort(descending=True)[1][:n_samples]

        return OrderedDict({k: v[indices] for (k, v) in subsample.items()})
