from typing import Any, Dict, Iterable, List
import logging

from rich.pretty import pprint
from rich.console import Console
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    SpinnerColumn,
    TimeElapsedColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
)
import aim


class Logger:
    def __init__(self, log_file: str = None, log_format: str = None):
        if log_format is None:
            log_format = "[%(asctime)s][%(levelname)s] | %(message)s"

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # Setup file handler
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_formatter = logging.Formatter(log_format)
            file_handler.setFormatter(file_formatter)
            file_handler.setLevel(logging.DEBUG)
            self.logger.addHandler(file_handler)

        # Setup console handler
        self.console = Console()

    def _beautify(self, logs: Dict[str, Any]) -> List[str]:
        results = []
        for key, value in logs.items():
            results.append(f"{key}: {value}")

        return results

    def print(self, *args, end: str = "\n") -> None:
        self.console.print(*args, end=end)

    def log(self, log: str, end: str = "\n") -> None:
        self.logger.info(log)
        self.console.print(log, end=end)

    def log_items(
        self, logs: Dict[str, Any], sep: str = " | ", end: str = "\n"
    ) -> None:
        logs = self._beautify(logs)
        self.logger.info(logs)
        self.console.print(*logs, sep=sep, end=end)

    def summary(self, logs: Dict[str, Any]) -> None:
        raise NotImplementedError()

    def progress_bar(
        self,
        iterable: Iterable,
        description: str = "Processing...",
        expand: bool = False,
        transient: bool = True,
    ):
        progress = Progress(
            SpinnerColumn(),
            TextColumn(description),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            TimeElapsedColumn(),
            console=self.console,
            expand=expand,
            transient=transient,
        )
        with progress:
            for item in progress.track(iterable):
                yield item


class AimLogger(Logger):
    def __init__(
        self,
        hparams: Dict[str, Any],
        aim_repo: str,
        experiment: str = None,
        run_hash: str = None,
        log_file: str = None,
        log_format: str = None,
    ):
        # Set Aim logging level to Warning
        logging.getLogger(aim.__name__).setLevel(logging.WARNING)

        self.aim_run = aim.Run(run_hash=run_hash, repo=repo, experiment=experiment)
        self.aim_run["hparams"] = vars(hparams)

        super().__init__(log_file=log_file, log_format=log_format)

    def log_items(
        self, logs: Dict[str, Any], subset: str, sep: str = " | ", end: str = "\n"
    ) -> None:
        self.aim_run.track(logs, context={"subset": subset})
        super().log_items(logs=logs, sep=sep, end=end)
