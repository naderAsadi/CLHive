import copy
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader

from ...methods import BaseMethod
from ...models import ContinualModel


class ContinualEvaluator:
    def __init__(
        self,
        method: BaseMethod,
        test_loader: DataLoader,
        config: Dict[str, Any],
        logger,
    ) -> None:

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.agent = method.to(self.device)
        self.test_loader = test_loader
        self.logger = logger
        self.config = config

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

                if self.config.eval.scenario is "multi_head":
                    target -= target.min()

                logits = self.agent.predict(data=data, task_id=task_t)

                n_total += data.size(0)
                if logits is not None:
                    pred = logits.max(1)[1]
                    n_ok += pred.eq(target).sum().item()

            accs[task_t] = (n_ok / n_total) * 100

        avg_acc = np.mean(accs[: task_id + 1])
        print("\n", "\t".join([str(int(x)) for x in accs]), f"\tAvg Acc: {avg_acc:.2f}")

    def fit(self, task_id: int):
        self._evaluate(task_id)
