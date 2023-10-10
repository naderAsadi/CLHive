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
        eval_scenario: Union[ClassIncremental, TaskIncremental],
        logger: Optional[BaseLogger] = None,
        device: Optional[torch.device] = None,
    ) -> "ContinualEvaluator":
        """_summary_

        Args:
            method (BaseMethod): _description_
            eval_scenario (Union[ClassIncremental, TaskIncremental]): _description_
            logger (Optional[BaseLogger], optional): _description_. Defaults to None.
            device (Optional[torch.device], optional): _description_. Defaults to None.

        Returns:
            ContinualEvaluator: _description_
        """

        super().__init__(method, eval_scenario, logger, device)

    @torch.no_grad()
    def _evaluate(self, task_id: int) -> List[float]:
        """_summary_

        Args:
            task_id (int): _description_

        Returns:
            _type_: _description_
        """
        self.agent.eval()
        tasks_accs = np.zeros(shape=self.eval_scenario.n_tasks)

        for task_id, eval_loader in enumerate(self.eval_scenario):
            n_ok, n_total = 0, 0

            # iterate over samples from task
            for idx, (x, y, t) in enumerate(eval_loader):
                x, y, t = x.to(self.device), y.to(self.device), t.to(self.device)

                logits = self.agent.predict(x=x, t=t)

                n_total += x.size(0)
                if logits is not None:
                    pred = logits.max(1)[1]
                    n_ok += pred.eq(y).sum().item()

            tasks_accs[task_id] = (n_ok / n_total) * 100

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

    def fit(self, current_task_id: int) -> None:
        """_summary_

        Args:
            current_task_id (int): _description_
        """
        self.on_eval_start()

        tasks_accs = self._evaluate(task_id=current_task_id)

        self.on_eval_end(tasks_accs=tasks_accs, current_task_id=current_task_id)
