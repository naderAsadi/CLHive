from abc import abstractmethod, ABC
from typing import Any, Dict, List, Optional


class BaseLogger(ABC):
    @abstractmethod
    def __init__(
        self, log_dir: str,
    ):
        self.log_dir = log_dir
        self._step: int = 0
        self._metrics: List[Dict[str, float]] = []

    @abstractmethod
    def log_metric(self, name: str, value: float, step: Optional[int] = None):
        """_summary_

        Args:
            name (str): _description_
            value (float): _description_
            step (Optional[int], optional): _description_. Defaults to None.
        """
        pass

    @abstractmethod
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
