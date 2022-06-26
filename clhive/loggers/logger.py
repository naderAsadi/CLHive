from typing import Any, Dict, List, Optional

from .base import BaseLogger
from ..utils.console_display import ConsoleDisplay


class Logger(BaseLogger):
    def __init__(
        self,
        n_tasks: int,
        loggers: Optional[List[BaseLogger]] = None,
        log_dir: Optional[str] = "./logs/",
    ):
        super().__init__()

        self._step: int = 0
        self._metrics: Dict[str, List[float]] = {}

        self.loggers = loggers
        self.display = ConsoleDisplay(n_tasks)

    @property
    def metrics(self) -> Dict[str, List[float]]:
        return self._metrics

    def add_logger(self, logger: BaseLogger):
        self.loggers.append(logger)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):

        for key, value in metrics.items():
            if key not in self._metrics.keys():
                self._metrics[key] = [value]
            else:
                self._metrics[key].append(value)
            self._step += 1

    def save(self) -> None:
        """Save log data."""
        pass

    def finalize(self, status: str) -> None:
        """Do any processing that is necessary to finalize an experiment.

        Args:
            status (str): Status that the experiment finished with (e.g. success, failed, aborted)
        """
        self.save()
