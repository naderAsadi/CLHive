from typing import Any, Dict, List, Optional

from .base import BaseLogger


class ContinualLogger(BaseLogger):
    def __init__(
        self,
        n_tasks: int,
        loggers: Optional[List[BaseLogger]] = None,
        save_dir: Optional[str] = "./logs/",
    ):
        super().__init__(log_dir)

        self.loggers = loggers

    def add_logger(self, logger: BaseLogger):
        self.loggers.append(logger)

    def log_metric(self, name: str, value: float, step: Optional[int] = None):
        """_summary_

        Args:
            name (str): _description_
            value (float): _description_
            step (Optional[int], optional): _description_. Defaults to None.
        """
        pass

    def log_dict(self, metrics: Dict[str, float], step: Optional[int] = None):
        """_summary_

        Args:
            metrics (Dict[str, float]): _description_
            step (Optional[int], optional): _description_. Defaults to None.
        """
        pass

    def save(self) -> None:
        """Save log data."""
        pass

    def finalize(self, status: str) -> None:
        """Do any processing that is necessary to finalize an experiment.

        Args:
            status (str): Status that the experiment finished with (e.g. success, failed, aborted)
        """
        self.save()
