import copy
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader

from .base import BaseEvaluator
from ...loggers import BaseLogger, Logger
from ...methods import BaseMethod
from ...scenarios import ClassIncremental, TaskIncremental


class ContinualEvaluator(BaseEvaluator):
    def __init__(
        self,
        method: BaseMethod,
        scenario: Union[ClassIncremental, TaskIncremental],
        logger: Optional[BaseLogger] = None,
        accelerator: Optional[str] = "gpu",
    ) -> "ContinualEvaluator":

        super().__init__(method, scenario, logger, accelerator)

    @torch.no_grad()
    def _evaluate(self, task_id: int, eval_loader: DataLoader):
        """_summary_

        Args:
            task_id (int): _description_
            eval_loader (DataLoader): _description_

        Returns:
            _type_: _description_
        """
        self.agent.eval()
        n_ok, n_total = 0, 0
        # iterate over samples from task
        for idx, (x, y, t) in enumerate(eval_loader):
            x, y, t = x.to(self.device), y.to(self.device), t.to(self.device)

            logits = self.agent.predict(x=x, t=t)

            n_total += x.size(0)
            if logits is not None:
                pred = logits.max(1)[1]
                n_ok += pred.eq(y).sum().item()

        task_acc = (n_ok / n_total) * 100

        return task_acc

    def on_eval_start(self):
        """ """
        pass

    def on_eval_end(self, tasks_accs: List[float], current_task_id: int):
        """ """
        avg_acc = np.mean(tasks_accs[: current_task_id + 1])
        print(
            "\n",
            "\t".join([str(int(x)) for x in tasks_accs]),
            f"\tAvg Acc: {avg_acc:.2f}",
        )

    def fit(self, current_task_id: int):
        """_summary_

        Args:
            current_task_id (int): _description_
        """
        self.on_eval_start()

        tasks_accs = np.zeros(shape=self.scenario.n_tasks)
        for task_id, eval_loader in enumerate(self.scenario):
            tasks_accs[task_id] = self._evaluate(task_id, eval_loader)

        self.on_eval_end(tasks_accs=tasks_accs, current_task_id=current_task_id)
