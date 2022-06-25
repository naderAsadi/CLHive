import time

from rich import box
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.theme import Theme
from rich.progress import Progress


# Theme
theme = Theme({
    "repr.number": "light_goldenrod3"
})

# Table
n_tasks = 4
table = Table(box=box.ROUNDED, header_style="bold light_goldenrod3")
table.add_column("Task ID")
for t in range(n_tasks):
    table.add_column(f"Task {t}\nAcc", justify="center")
table.add_column("Avg.\nAcc.", justify="center", style="light_goldenrod3")
table.add_column("Avg.\nFgt.", justify="center", style="light_goldenrod3")


# Progress Bar
progress = Progress(expand=True)
progress_bar = progress.add_task("Task 1", total=100)


# Body
body = Table.grid(expand=True)
body.add_row(table)
body.add_row(progress)

console = Console(theme=theme)
console.print("Summary\nNumber of Tasks: 10")

# with Live(body, refresh_per_second=4) as live:  # update 4 times a second to feel fluid
    # for row in range(10):
    #     table.add_row(f"{row}", f"description {row}", "[red]ERROR")
    #     progress.update(progress_bar, advance=row)
    #     time.sleep(0.4)

display = Live(body, refresh_per_second=4)
display.start()

for row in range(10):
    task_metrics = [f"{row}", "90.3", "91.0", "90.3", "91.0", "90.7", "0.7"]
    table.add_row(*task_metrics)
    progress.update(progress_bar, advance=row)
    display.update(body)
    time.sleep(0.4)

display.stop()
