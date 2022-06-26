import time

from rich import box
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.theme import Theme
from rich.progress import Progress, Column, BarColumn, SpinnerColumn, TextColumn, TimeElapsedColumn


# Theme
theme = Theme({
    "repr.number": "light_goldenrod3"
})

# Table
n_tasks = 4
table = Table(box=box.ROUNDED, header_style="bold light_goldenrod3")
table.add_column("Task\nID")
for t in range(n_tasks):
    table.add_column(f"Task {t}\nAcc", justify="center")
table.add_column("Avg.\nAcc.", justify="center", style="light_goldenrod3")
table.add_column("Avg.\nFgt.", justify="center", style="light_goldenrod3")



# Progress Bar
print(Progress.get_default_columns()[2].text_format)

spinner_column = SpinnerColumn()
text_column = TextColumn("{task.description}")
seperator = TextColumn("|")
dot = TextColumn("•")
epoch_column = TextColumn("Epoch")
bar_column = BarColumn(bar_width=40, style="grey70", complete_style="dark_sea_green4")
percent_column = TextColumn("[progress.percentage]{task.percentage:>3.0f}%")
time_column = TimeElapsedColumn()

progress = Progress(
    spinner_column,
    text_column, 
    seperator,
    epoch_column,
    bar_column, 
    percent_column,
    dot,
    time_column,
    seperator,
    expand=False
)
# progress = Progress()
train_bar = progress.add_task("Task 1 Training", total=10)


# Body
body = Table.grid(expand=True)
body.add_row(table)
body.add_row(progress)

console = Console(theme=theme)
console.print("Summary\nNumber of Tasks: 10")

# Display
display = Live(body, refresh_per_second=4)
display.start()

for row in range(10):
    task_metrics = [f"{row}", "90.3", "91.0", "90.3", "91.0", "90.7", "0.7"]
    table.add_row(*task_metrics)
    epoch_column.text_format = f"Epoch {row}"
    progress.update(train_bar, advance=1)
    display.update(body)
    time.sleep(0.7)

progress.remove_task(train_bar)
display.stop()
