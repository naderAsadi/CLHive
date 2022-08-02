from typing import Any, Dict, List, Optional, Tuple, Union
import torch

from ...loggers import BaseLogger, Logger
from ...methods import BaseMethod
from ...scenarios import ClassIncremental, TaskIncremental


class BaseEvaluator:
    def __init__(
        self,
        method: BaseMethod,
        eval_scenario: Union[ClassIncremental, TaskIncremental],
        logger: Optional[BaseLogger] = None,
        device: Optional[torch.device] = None,
    ) -> "BaseEvaluator":
        """_summary_

        Args:
            method (BaseMethod): _description_
            eval_scenario (Union[ClassIncremental, TaskIncremental]): _description_
            logger (Optional[BaseLogger], optional): _description_. Defaults to None.
            device (Optional[torch.device], optional): _description_. Defaults to None.

        Returns:
            BaseEvaluator: _description_
        """
        if device is None:
            device = torch.device("cpu")
        self.device = device

        self.agent = method.to(self.device)
        self.eval_scenario = eval_scenario

        if logger is None:
            logger = Logger(n_tasks=self.eval_scenario.n_tasks)
        self.logger = logger

    @torch.no_grad()
    def _evaluate(self, task_id: int):
        pass

    def on_eval_start(self):
        """ """
        pass

    def on_eval_end(self):
        """ """
        pass

    def fit(self, task_id: int):
        pass
