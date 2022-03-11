import copy
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchcl.methods import BaseMethod
from torchcl.models import ModelWrapper
from torchcl.models.heads import LinearClassifier
from torchcl.utils.evaluator import Evaluator


class ProbeEvaluator(Evaluator):

    def __init__(
        self,
        method: BaseMethod,
        train_loader: DataLoader,
        test_loader: DataLoader,
        config: Dict[str, Any],
        logger,
    ) -> None:

        super().__init__(method, test_loader, config, logger)

        self.train_loader = train_loader
        self.linear_heads = {}

    def _train_linear_heads(self, task_list):

        if self.config.eval.scenario == 'multi_head':
            probe_n_classes = self.config.data.n_classes_per_task * (task_list[0] + 1)
            sample_all_seen_tasks = False
        else:
            probe_n_classes = self.config.data.n_classes
            sample_all_seen_tasks = True

        self.linear_heads = {
            str(t) : LinearClassifier(
                feature_dim = int(self.agent.model._model.last_hid),
                n_classes=probe_n_classes
            ).to(self.device) for t in task_list
        }
        self.agent.eval()

        for task_t in task_list:
            self.train_loader.sampler.set_task(task_t, sample_all_seen_tasks=sample_all_seen_tasks)
            optim = torch.optim.SGD(self.linear_heads[str(task_t)].parameters(), 
                                    lr=self.config.train.optim.lr,
                                    momentum=0.9,
                                    weight_decay=0.)
            
            for epoch in range(self.config.eval.n_epochs):
                for _, (x, y) in enumerate(self.train_loader):
                    x, y = x.to(self.device), y.to(self.device)

                    if self.config.eval.scenario == 'multi_head':
                        y = y - task_t * self.config.data.n_classes_per_task

                    with torch.no_grad():
                        features = self.agent.model.forward_model(x)
                    logits = self.linear_heads[str(task_t)](features.detach())
                    loss = F.cross_entropy(logits, y)

                    optim.zero_grad()
                    loss.backward()
                    optim.step()

                    print(f"Linear head {task_t} | Epoch: {epoch + 1} / {self.config.eval.n_epochs} - Training loss: {loss}", end='\r')

    def fit(self, task_id: int):
        if self.config.eval.scenario == 'multi_head':
            task_list = [*range(task)]
        else:
            task_list = [task_id]

        self._train_linear_heads(task_list)
        self._evaluate(task_id)
