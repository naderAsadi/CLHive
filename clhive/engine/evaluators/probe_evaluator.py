import copy
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from .base import BaseEvaluator
from ...loggers import BaseLogger, Logger
from ...methods import BaseMethod
from ...models import ContinualModel, LinearClassifier
from ...scenarios import ClassIncremental, TaskIncremental


class ProbeEvaluator(BaseEvaluator):
    def __init__(
        self,
        method: BaseMethod,
        train_scenario: Union[ClassIncremental, TaskIncremental],
        eval_scenario: Union[ClassIncremental, TaskIncremental],
        n_epochs: int,
        logger: Optional[BaseLogger] = None,
        device: Optional[torch.device] = None,
    ) -> "ProbeEvaluator":
        """_summary_

        Args:
            method (BaseMethod): _description_
            train_scenario (Union[ClassIncremental, TaskIncremental]): _description_
            eval_scenario (Union[ClassIncremental, TaskIncremental]): _description_
            n_epochs (int): _description_
            logger (Optional[BaseLogger], optional): _description_. Defaults to None.
            device (Optional[torch.device], optional): _description_. Defaults to None.

        Returns:
            ProbeEvaluator: _description_
        """

        super().__init__(method, eval_scenario, logger, device)

        self.train_scenario = train_scenario
        self.n_epochs = n_epochs
        self.linear_heads = {}

    def _train_linear_heads(self, task_id: int) -> None:
        if type(self.train_scenario) is TaskIncremental:
            probe_n_classes = self.train_scenario.loader.sampler.cpt
            sample_all_seen_tasks = False
            task_list = [*range(self.train_scenario.n_tasks)]
        else:
            probe_n_classes = self.train_scenario.n_classes
            sample_all_seen_tasks = True
            task_list = [task_id]

        self.linear_heads = {
            str(t): LinearClassifier(
                input_size=int(self.agent.model.backbone.last_hid),
                output_size=probe_n_classes,
            ).to(self.device)
            for t in task_list
        }
        self.agent.eval()

        train_loader = self.train_scenario.loader

        for task_t in task_list:
            train_loader.sampler.set_task(
                task_id=task_t,
                sample_all_seen_tasks=type(self.train_scenario) is ClassIncremental,
            )
            optim = torch.optim.AdamW(
                self.linear_heads[str(task_t)].parameters(),
                lr=1e-4,
                weight_decay=5e-4,
            )

            for epoch in range(self.n_epochs):
                for _, (x, y, t) in enumerate(train_loader):
                    x, y, t = x.to(self.device), y.to(self.device), t.to(self.device)

                    with torch.no_grad():
                        features = self.agent.model.forward_backbone(x)
                    logits = self.linear_heads[str(task_t)](features.detach())
                    loss = F.cross_entropy(logits, y)

                    optim.zero_grad()
                    loss.backward()
                    optim.step()

                    print(
                        f"Linear head {task_t} | Epoch: {epoch + 1} / {self.n_epochs} - Training loss: {loss}",
                        end="\r",
                    )

    @torch.no_grad()
    def _evaluate(self, task_id: int) -> List[float]:
        """_summary_

        Args:
            task_id (int): _description_
        """
        self.agent.eval()
        tasks_accs = np.zeros(shape=self.eval_scenario.n_tasks)

        for task_t, eval_loader in enumerate(self.eval_scenario):
            n_ok, n_total = 0, 0

            # iterate over samples from task
            for idx, (x, y, t) in enumerate(eval_loader):
                x, y, t = x.to(self.device), y.to(self.device), t.to(self.device)

                probe_id = task_id
                if type(self.eval_scenario) is TaskIncremental:
                    probe_id = task_t

                features = self.agent.model.forward_backbone(x)
                logits = self.linear_heads[str(probe_id)](features)

                n_total += x.size(0)
                if logits is not None:
                    pred = logits.max(1)[1]
                    n_ok += pred.eq(y).sum().item()

            tasks_accs[task_t] = (n_ok / n_total) * 100

        return tasks_accs

    def on_eval_start(self):
        """ """
        pass

    def on_eval_end(self, tasks_accs: List[float], current_task_id: int) -> None:
        """ """
        avg_obs_acc = np.mean(tasks_accs[: current_task_id + 1])
        avg_anytime_acc = np.mean(tasks_accs)
        print(
            "\n",
            "\t".join([str(int(x)) for x in tasks_accs]),
            f"  |  Avg observed Acc: {avg_obs_acc:.2f}  |  Avg anytime Acc: {avg_anytime_acc:.2f}",
        )

        # Reset train_scenario
        self.train_scenario.set_task(task_id=current_task_id + 1)

    def fit(self, current_task_id: int = None) -> None:
        """_summary_

        Args:
            current_task_id (int, optional): _description_. Defaults to None.
        """
        self.on_eval_start()

        self._train_linear_heads(task_id=current_task_id)
        tasks_accs = self._evaluate(task_id=current_task_id)

        self.on_eval_end(tasks_accs, current_task_id)
