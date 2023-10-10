import torch
from clhive.data.datasets import CIFAR100Dataset
from clhive.data.scenarios import ClassIncrementalLoader
from clhive.data import DataConfig, ReplayBuffer
from clhive.models import resnet18
from clhive.methods import ER
from clhive.utils import Logger

logger = Logger()

data_config = DataConfig(
    dataset="cifar100", root="./files/", num_classes=100, image_size=32, num_workers=0
)
train_dataset = CIFAR100Dataset.from_config(data_config, split="train")
train_scenario = ClassIncrementalLoader(
    train_dataset, n_tasks=20, batch_size=64, n_workers=data_config.num_workers
)

buffer = ReplayBuffer(capacity=1000)

model = resnet18(num_classes=data_config.num_classes)
agent = ER(
    model=model,
    optimizer=torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9),
    buffer=buffer,
    n_replay_samples=32,
)

for train_loader in train_scenario:
    for pixel_values, labels, task_ids in logger.progress_bar(train_loader):
        loss = agent.observe(pixel_values, labels, task_ids)
        logger.log_items({"train/loss": loss.item()})
