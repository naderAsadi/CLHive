from typing import Any, Dict, List, Optional

from rich import box
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.theme import Theme
from rich.progress import (
    Progress,
    Column,
    BarColumn,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)


class ConsoleDisplay:
    def __init__(
        self,
        n_tasks: int,
        progress_metrics_columns: Optional[Dict[str, Any]] = None,
        primary_style: Optional[str] = "light_goldenrod3",
        accent_style: Optional[str] = "dark_sea_green4",
        secondary_accent_style: Optional[str] = "",
    ):
        self.primary_style = primary_style
        self.accent_style = accent_style
        self.secondary_accent_style = secondary_accent_style
        self.theme = Theme({"repr.number": f"{self.primary_style}"})

        self.console = Console(theme=self.theme)
        self.table = self.create_metrics_table(n_tasks)
        self.create_progress(metrics_columns=progress_metrics_columns)
        self.progress_bar = None

        self.body = Table.grid(expand=True)
        self.body.add_row(self.table)
        self.body.add_row(self.progress)
        self.display = Live(self.body, refresh_per_second=10)

    def start(self):
        self.display.start()

    def stop(self):
        self.progress.remove_task(self.progress_bar)
        self.display.stop()

    def create_metrics_table(self, n_tasks: int) -> Table:
        """Creates the main table reporting performance across tasks."""
        table = Table(box=box.ROUNDED, header_style=f"bold {self.primary_style}")
        table.add_column("Task\nID")
        for t in range(n_tasks):
            table.add_column(f"Task {t}\nAcc", justify="center")
        table.add_column("Avg.\nAcc.", justify="center", style=f"{self.primary_style}")
        table.add_column("Avg.\nFgt.", justify="center", style=f"{self.primary_style}")

        return table

    def create_progress(
        self,
        metrics_columns: Optional[Dict[str, Any]] = None,
    ):

        main_columns = {}

        main_columns["spinner"] = SpinnerColumn()
        main_columns["description"] = TextColumn("{task.description}")
        main_columns["seperator"] = TextColumn("|")
        main_columns["dot"] = TextColumn("•")
        main_columns["epoch"] = TextColumn("Epoch")
        main_columns["bar"] = BarColumn(
            bar_width=40, style="grey70", complete_style=self.accent_style
        )
        main_columns["percentage"] = TextColumn(
            "[progress.percentage]{task.percentage:>3.0f}%"
        )
        main_columns["time_elapsed"] = TimeElapsedColumn()

        columns = [*list(main_columns.values())]
        if metrics_columns is not None:
            for key, value in metrics_columns.items():
                assert (
                    key not in main_columns.keys()
                ), f"Duplicate key error, `{key}` already exists in the default main_columns."
                main_columns[key] = TextColumn(text_format=metrics_columns[key])
                columns.extend([main_columns["seperator"], main_columns[key]])

        self.columns = main_columns
        self.progress = Progress(*columns)

    def add_progress_bar(self, description: str, total_steps: int):
        """_summary_

        Args:
            description (str): _description_
            total_steps (int): _description_
        """
        if self.progress_bar is not None:
            self.progress.remove_task(self.progress_bar)

        self.progress_bar = self.progress.add_task(description, total=total_steps)

    def update_table(self, metrics: List[float]):
        """_summary_

        Args:
            metrics (List[float]): _description_
        """
        assert len(metrics) == len(
            self.table.columns
        ), "Number of metrics does not match the number of columns."
        metrics = [str(x) for x in metrics]
        self.table.add_row(*metrics)
        self.display.update(self.body)

    def update_progress_bar(
        self,
        epoch: int,
        advance: Optional[int] = 1,
        metrics_columns: Optional[Dict[str, Any]] = None,
    ):

        self.columns["epoch"].text_format = f"Epoch: {epoch}"
        for key, value in metrics_columns.items():
            self.columns[key].text_format = f"{key}: {str(value)}"

        self.progress.update(self.progress_bar, advance=advance)
        self.display.update(self.body)

    def reset_progress_bar(self):
        if self.progress_bar is not None:
            self.progress.reset(self.progress_bar)


if __name__ == "__main__":
    import time

    stats = [
        [50, 60, 70, 80, 90],
        [50, 60, 70, 80, 90],
        [50, 60, 70, 80, 90],
        [50, 60, 70, 80, 90],
        [50, 60, 70, 80, 90],
    ]
    progress_metrics = {"train/loss": 50.0}

    display = ConsoleDisplay(n_tasks=5, progress_metrics_columns=progress_metrics)
    # display.create_progress(metrics_columns={"train/loss": 100.0})

    display.start()
    for t in range(5):
        display.add_progress_bar(
            description=f"Task {t} Training",
            total_steps=2 * 100,
        )
        for e in range(2):
            for i in range(100):
                time.sleep(0.01)
                progress_metrics["train/loss"] -= 0.5
                display.update_progress_bar(
                    epoch=e, advance=1, metrics_columns=progress_metrics
                )

        display.update_table(metrics=[t, *stats[t], 70.0, 0.0])

    display.stop()
