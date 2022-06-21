## Overview
**CLHive** is a codebase on top of [PyTorch](https://pytorch.org) for Continual Learning research. It provides the components necessary to run CL experiments, for both task-incremental and class-incremental settings. It is designed to be readable and easily extensible, to allow users to quickly run and experiment with their own ideas.


## How To Use

```python
from clhive.data import SplitCIFAR10
from clhive.scenarios import ClassIncremental
from clhive.models import ContinualModel

dataset = SplitCIFAR10(root="../cl-datasets/data/")
scenario = ClassIncremental(dataset=dataset, n_tasks=5, batch_size=32)

print(
  f"Number of tasks: {scenario.n_tasks} | Number of classes: {scenario.n_classes}"
)

model = ContinualModel.auto_model("resnet18", scenario, image_size=32)

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
