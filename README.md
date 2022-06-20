## Overview
**CLHive** is a codebase on top of [PyTorch](https://pytorch.org) for Continual Learning research. It provides the components necessary to run CL experiments, for both task-incremental and class-incremental settings. It is designed to be readable and easily extensible, to allow users to quickly run and experiment with their own ideas.


## How To Use

```python
from torch.utils.data import DataLoader

from clhive import ClassIncremental
from clhive.datasets import MNIST

train_set = MNIST("my/data/path", download=True, train=True)
test_set = MNIST("my/data/path", download=True, train=False)

scenario = ClassIncremental(
    train_dataset=train_set,
    test_dataset=test_set,
    n_tasks=5,
    batch_size=32
)

print(f"Number of tasks: {scenario.n_tasks}.")

for task_id, train_loader in enumerate(scenario):
    for x, y, t in train_loader:
        # Do your cool stuff here
```

<details>
  <summary>Training examples</summary>
  
Train CLIP with ViT-base on COCO Captions dataset:

```
python main.py data=coco model/vision_model=vit-b  model/text_model=vit-b
```
  
</details>

## Reading The Commits
Here is a reference to what each emoji in the commits means:

* 📎 : Some basic updates.
* ♻️ : Refactoring.
* 💩 : Bad code, needs to be revised!
* 🐛 : Bug fix.
* 💡 : New feature.
* ⚡ : Performance Improvement.
