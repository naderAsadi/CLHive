from typing import Optional, Union
import copy
import os
import time

import torch
from torch.utils.data import DataLoader

from .evaluators import ContinualEvaluator, ProbeEvaluator
from ..methods import BaseMethod
from ..models import ContinualModel
from ..scenarios import ClassIncremental, TaskIncremental


class Trainer:
    def __init__(
        self,
        method: BaseMethod,
        scenario: Union[ClassIncremental, TaskIncremental],
        n_epochs: int,
        evaluator: Optional[Union[ContinualEvaluator, ProbeEvaluator]] = None,
        logger=None,
        accelerator: Optional[str] = "gpu",
    ) -> "Trainer":

        assert accelerator in ["gpu", "cpu", None], (
            "Currently supported accelerators are [`gpu`, `cpu`],"
            + " but {accelerator} was received."
        )

        self.device = torch.device(
            "cuda" if accelerator == "gpu" and torch.cuda.is_available() else "cpu"
        )

        self.agent = method.to(self.device)
        self.scenario = scenario
        self.logger = logger
        self.evaluator = evaluator

        self.n_epochs = n_epochs

    def _train_task(self, task_id: int, train_loader: DataLoader):

        self.agent.train()
        self.agent.on_task_start()

        start_time = time.time()

        print(f"\n>>> Task #{task_id} --> Model Training")
        for epoch in range(self.n_epochs):
            # adjust learning rate

            for idx, (x, y, t) in enumerate(train_loader):
                x, y, t = x.to(self.device), y.to(self.device), t.to(self.device)
                loss = self.agent.observe(x, y, t)

                print(
                    f"Epoch: {epoch + 1} / {self.n_epochs} | {idx} / {len(train_loader)} - Loss: {loss}",
                    end="\r",
                )

        print(f"Task {task_id}. Time {time.time() - start_time:.2f}")
        self.on_task_end()

    def set_evaluator(self, evaluator: Union[ContinualEvaluator, ProbeEvaluator]):
        self.evaluator = evaluator

    def on_task_end(self):

        finished_task_id = self.agent._current_task_id
        self.agent.on_task_end()

        if self.evaluator is not None:
            self.evaluator.fit(task_id=finished_task_id)

    def fit(self):

        for task_id, train_loader in enumerate(self.scenario):
            self._train_task(task_id=task_id, train_loader=train_loader)
