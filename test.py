from clhive.data import MNISTDataset
from clhive.scenarios import ClassIncremental, TaskIncremental


dataset = MNISTDataset(root="my/data/path")

scenario = ClassIncremental(dataset=dataset, n_tasks=5, batch_size=32, n_workers=6)

print(f"Number of tasks: {scenario.n_tasks} | Number of classes: {scenario.n_classes}")

for task_id, train_loader in enumerate(scenario):
    for x, y, t in train_loader:
        # Do your cool stuff here
