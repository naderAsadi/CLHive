import torch
from torch.optim import SGD, AdamW

from clhive.data import SplitCIFAR100
from clhive.utils.evaluators import ContinualEvaluator, ProbeEvaluator
from clhive.scenarios import ClassIncremental, TaskIncremental
from clhive.models import ContinualModel
from clhive.methods import auto_method
from clhive import Trainer, ReplayBuffer


# HParams
batch_size = 32
n_tasks = 5
buffer_capacity = 50 * 10

# Train dataset and scenario
dataset = SplitCIFAR100(root="../cl-datasets/")
scenario = ClassIncremental(
    dataset=dataset, n_tasks=n_tasks, batch_size=batch_size, n_workers=6
)

print(f"Number of tasks: {scenario.n_tasks} | Number of classes: {scenario.n_classes}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model
model = ContinualModel.auto_model("resnet18", scenario).to(device)

# Replay buffer and ER agent
buffer = ReplayBuffer(capacity=buffer_capacity, device=device)
agent = auto_method(
    name="er_ace",
    model=model,
    optim=SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4),
    buffer=buffer,
)

# Evaluator
test_dataset = SplitCIFAR100(root="../cl-datasets/", train=False)
test_scenario = ClassIncremental(
    test_dataset, n_tasks=n_tasks, batch_size=batch_size, n_workers=6
)
evaluator = ContinualEvaluator(method=agent, eval_scenario=test_scenario, device=device)

# Trainer
trainer = Trainer(
    method=agent, scenario=scenario, evaluator=evaluator, n_epochs=5, device=device
)
trainer.fit()
