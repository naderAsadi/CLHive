from typing import Any, Dict, List, Optional
from rich import box
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.theme import Theme
from rich.progress import Progress

from clhive.loggers.base import BaseLogger


class Logger(BaseLogger):
    def __init__(
        self,
        loggers: Optional[List[BaseLogger]] = None,
        log_dir: Optional[str] = "./logs/",
    ):
        super().__init__()

        self._step: int = 0
        self._metrics: Dict[str, List[float]] = {}

        self.loggers = loggers

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



def create_display(n_tasks: int):
    # Create display table
    table = Table(box=box.ROUNDED, header_style="bold light_goldenrod3")
    table.add_column("Task ID")
    for t in range(n_tasks):
        table.add_column(f"Task {t}\nAcc", justify="center")
    table.add_column("Avg.\nAcc.", justify="center", style="light_goldenrod3")
    table.add_column("Avg.\nFgt.", justify="center", style="light_goldenrod3")

    progress = Progress(expand=True)
    progress_bar = progress.add_task("", total=100)