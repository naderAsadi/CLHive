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
        config: Dict[str, Any],
        logger=None,
        evaluator=None,
    ) -> None:
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.agent = method.to(self.device)
        self.loader = train_loader
        self.config = config
        self.logger = logger
        self.evaluator = evaluator

    def _train_task(self, task_id: int):
        
        self.loader.sampler.set_task(task_id)

        self.agent.train()
        self.agent.on_task_start()


        start_time = time.time()

        print('\n>>> Task #{} --> Model Training'.format(task_id))
        for epoch in range(self.config.train.n_epochs):
            # adjust learning rate

            for idx, (data, targets, task) in enumerate(self.loader):
                if self.config.train.scenario == 'multi_head':
                    targets -= targets.min()

                inc_data = {
                    'x': data.to(self.device), 
                    'y': targets.to(self.device), 
                    't': task
                }
                loss = self.agent.observe(inc_data)

                print(f'Epoch: {epoch + 1} / {self.config.train.n_epochs} | {idx} / {len(self.loader)} - Loss: {loss}', end='\r')
        
        print(f'Task {task_id}. Time {time.time() - start_time:.2f}')
        self.on_task_end()

    def configure_optimizer(self):
        pass

    def on_task_end(self):

        finished_task_id = self.agent._current_task_id
        self.agent.on_task_end()

        if self.evaluator is not None:
            self.evaluator.fit(task_id=finished_task_id)

    def fit(self):

        for task_id in range(self.config.data.n_tasks):
            self._train_task(task_id=task_id)
    