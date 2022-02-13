import copy
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from torchcl.methods import BaseMethod
from torchcl.models import ModelWrapper


class Trainer:

    def __init__(
        self,
        method: BaseMethod,
        train_loader: torch.utils.data.DataLoader,
        logger,
        config: Dict[str, Any],
    ) -> None:
        
        self.agent = method
        self.loader = train_loader
        self.logger = logger
        self.config = config

    def _train_task(self, task_id: int):
        
        self.loader.sampler.set_task(task_id)

        self.agent.train()
        self.agent.on_task_start()

        start_time = time.time()

        print('\n>>> Task #{} --> Model Training'.format(task_id))
        for epoch in range(self.config.train.n_epochs):
            # adjust learning rate

            for idx, (data, targets) in enumerate(self.loader):

                inc_data = {'x': data, 'y': targets, 't': task_id}
                loss = self.agent.observe(inc_data)

                print(f'Epoch: {epoch + 1} / {self.config.train.n_epochs} | {idx} / {len(self.loader)} - Loss: {loss}', end='\r')
        
        print(f'Task {task_id}. Time {time.time() - start_time:.2f}')
        self.agent.on_task_end()
        

    def fit(self):

        for task_id in range(self.config.data.n_tasks):
            self._train_task(task_id=task_id)
    