import copy
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ...methods import BaseMethod
from ...models import ModelWrapper
from ...models.heads import LinearClassifier
from ...utils.evaluators import Evaluator


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

    def _train_linear_heads(self, task_id):

        if self.config.eval.scenario == 'multi_head':
            probe_n_classes = self.config.data.n_classes_per_task * (task_id + 1)
            sample_all_seen_tasks = False
            task_list = [*range(task_id + 1)]
        else:
            probe_n_classes = self.config.data.n_classes
            sample_all_seen_tasks = True
            task_list = [task_id]

        self.linear_heads = {
            str(t) : LinearClassifier(
                feature_dim = int(self.agent.model._model.last_hid),
                n_classes=probe_n_classes
            ).to(self.device) for t in task_list
        }
        self.agent.eval()

        for task_t in task_list:
            self.train_loader.sampler.set_task(task_t, sample_all_seen_tasks=sample_all_seen_tasks)
            optim = torch.optim.SGD(
                self.linear_heads[str(task_t)].parameters(), 
                lr=self.config.optim.lr,
                momentum=0.9,
                weight_decay=0.
            )
            
            for epoch in range(self.config.eval.n_epochs):
                for _, (x, y, t) in enumerate(self.train_loader):
                    x, y = x.to(self.device), y.to(self.device)

                    if self.config.eval.scenario == 'multi_head':
                        y -= y.min()

                    with torch.no_grad():
                        features = self.agent.model.forward_model(x)
                    logits = self.linear_heads[str(task_t)](features.detach())
                    loss = F.cross_entropy(logits, y)

                    optim.zero_grad()
                    loss.backward()
                    optim.step()

                    print(f"Linear head {task_t} | Epoch: {epoch + 1} / {self.config.eval.n_epochs} - Training loss: {loss}", end='\r')

    @torch.no_grad()
    def _evaluate(self, task_id: int):
        self.agent.eval()

        accs = np.zeros(shape=(self.config.data.n_tasks,))

        for task_t in range(task_id + 1):

            n_ok, n_total = 0, 0
            self.test_loader.sampler.set_task(task_t)

            # iterate over samples from task
            for i, (data, target, task) in enumerate(self.test_loader):

                data, target = data.to(self.device), target.to(self.device)
                
                probe_id = task_id
                if self.config.eval.scenario == 'multi_head':
                    target = target - task_t * self.config.data.n_classes_per_task
                    probe_id = task_t
               
                features = self.agent.model.forward_model(data)
                logits = self.linear_heads[str(probe_id)](features)

                n_total += data.size(0)
                if logits is not None:
                    pred   = logits.max(1)[1]
                    n_ok   += pred.eq(target).sum().item()

            accs[task_t] = (n_ok / n_total) * 100
        
        avg_acc = np.mean(accs[:task_id + 1])
        print('\n', '\t'.join([str(int(x)) for x in accs]), f'\tAvg Acc: {avg_acc:.2f}')


    def fit(self, task_id: int = None):            
        
        if task_id is None:
            task_id = self.config.data.n_tasks - 1
        self._train_linear_heads(task_id)
        self._evaluate(task_id)
