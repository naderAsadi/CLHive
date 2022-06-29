import torch
from torch.optim import SGD, AdamW

from clhive.data import SplitCIFAR100
from clhive.utils.evaluators import ContinualEvaluator, ProbeEvaluator
from clhive.scenarios import ClassIncremental, TaskIncremental
from clhive.models import ContinualModel
from clhive.methods import auto_method
from clhive import Trainer

dataset = SplitCIFAR100(root="../cl-datasets/data/")
scenario = TaskIncremental(dataset=dataset, n_tasks=10, batch_size=128, n_workers=6)

print(f"Number of tasks: {scenario.n_tasks} | Number of classes: {scenario.n_classes}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ContinualModel.auto_model("resnet18", scenario, image_size=32).to(device)

agent = auto_method(
    name="finetuning",
    model=model,
    optim=AdamW(model.parameters(), lr=1e-5, weight_decay=5e-4),
)

test_dataset = SplitCIFAR100(root="../cl-datasets/data/", train=False)
test_scenario = TaskIncremental(test_dataset, n_tasks=10, batch_size=128, n_workers=6)
evaluator = ProbeEvaluator(
    method=agent,
    train_scenario=scenario,
    eval_scenario=test_scenario,
    n_epochs=5,
    accelerator="gpu",
)

trainer = Trainer(
    method=agent, scenario=scenario, n_epochs=5, evaluator=evaluator, accelerator="gpu"
)
trainer.fit()
