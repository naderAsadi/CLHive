import torch

from clhive.data import SplitCIFAR10
from clhive.scenarios import ClassIncremental, TaskIncremental
from clhive.models import ContinualModel

dataset = SplitCIFAR10(root="../cl-datasets/data/")
scenario = TaskIncremental(dataset=dataset, n_tasks=5, batch_size=32, n_workers=6)

print(f"Number of tasks: {scenario.n_tasks} | Number of classes: {scenario.n_classes}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ContinualModel.auto_model("resnet18", scenario, image_size=32).to(device)

for task_id, train_loader in enumerate(scenario):
    for x, y, t in train_loader:
        # Do your cool stuff here
        x, y, t = x.to(device), y.to(device), t.to(device)
        pred = model(x, t)
        print(task_id, y.min(), y.max(), pred.shape)
