<h2 align="center">🚧 Under Development 🚧</h2>

## Overview
**CLHive** is a codebase on top of [PyTorch](https://pytorch.org) for Continual Learning research. It provides the components necessary to run CL experiments, for both task-incremental and class-incremental settings. It is designed to be readable and easily extensible, to allow users to quickly run and experiment with their own ideas.

### Currently Supported Methods

- [LwF, 2016](https://arxiv.org/abs/1606.09282)
- [~~iCaRL, 2016~~](https://arxiv.org/abs/1611.07725)
- [EWC, 2017](https://arxiv.org/abs/1612.00796)
- [~~AGEM, 2018~~](https://arxiv.org/abs/1812.00420)
- [ER, 2019](https://arxiv.org/abs/1902.10486)
- [~~MIR, 2019~~](https://arxiv.org/abs/1908.04742)
- [~~DER, 2020~~](https://arxiv.org/abs/2004.07211)
- [ER-ACE/ER-AML, 2022](https://arxiv.org/abs/2104.05025)

## Installation

### Dependencies

CLHive requires **Python 3.6+**.

- fvcore>=0.1.5
- hydra-core>=1.0.0
- numpy>=1.22.4
- rich>=12.4.4
- pytorch>=1.12.0
- torchvision>=0.13.0
- wandb>=0.12.19

<!-- ### PyPI Installation
You can install Lightly and its dependencies from PyPI with:
```
pip install clhive
``` -->

### Manual Installation
It is strongly recommend that you install CLHive in a dedicated virtualenv, to avoid conflicting with your system packages.

```
git clone https://github.com/naderAsadi/CLHive.git
cd CLHive
pip install -e .
```


## How To Use

With `clhive` you can use latest continual learning methods in a modular way using the full power of PyTorch. Experiment with different backbones, models and loss functions. The framework has been designed to be easy to use from the ground up.

### Quick Start

```python
from clhive.data import SplitCIFAR10
from clhive.scenarios import ClassIncremental
from clhive.models import ContinualModel
from clhive.methods import auto_method

train_dataset = SplitCIFAR10(root="path/to/data/", train=True)
train_scenario = ClassIncremental(dataset=dataset, n_tasks=5, batch_size=32)

print(
  f"Number of tasks: {train_scenario.n_tasks} | Number of classes: {train_scenario.n_classes}"
)

model = ContinualModel.auto_model("resnet18", train_scenario, image_size=32)
agent = auto_method(
    name="finetuning", model=model, optim=SGD(model.parameters(), lr=0.01)
)

for task_id, train_loader in enumerate(train_scenario):
    for x, y, t in train_loader:
        loss = agent.observe(x, y, t)
        ...
```

To create a replay buffer for rehearsal-based methods, *e.g.* ER, you can use `clhive.ReplayBuffer` class. 

```python
from clhive import ReplayBuffer

device = torch.device("cuda")
buffer = ReplayBuffer(capacity=20 * 10, device=device)

agent = auto_method(
    name="er",
    model=model,
    optim=SGD(model.parameters(), lr=0.01),
    buffer=buffer
)
```

Instead of iterating over all tasks manually, you can easily use `clhive.Trainer` to train the continual agent in any of the supported scenarios. 

```python
from clhive import Trainer

trainer = Trainer(method=agent, scenario=train_scenario, n_epochs=5, device=device)
trainer.fit()
```

Similar to the `Trainer` class, `clhive.utils.evaluators` package offers several evaluators, *e.g.*  ContinualEvaluator and ProbeEvaluator.

```python
from clhive.utils.evaluators import ContinualEvaluator, ProbeEvaluator

test_dataset = SplitCIFAR10(root="path/to/data/", train=False)
test_scenario = ClassIncremental(test_dataset, n_tasks=5, batch_size=32, n_workers=6)

evaluator = ContinualEvaluator(method=agent, scenario=test_scenario, device=device)
evaluator.fit()
```

Evaluators can also be passed to `clhive.Trainer` for automatic evaluation after each task.

```python
trainer = Trainer(
    method=agent, scenario=scenario, n_epochs=5, evaluator=evaluator, device=device
)
trainer.fit()
```


### Command-Line Interface

CLHive is accessible also through a command-line interface (CLI). To train a ER model on Tiny-ImageNet you can simply run the following command:

```
python main.py ...
```

<details>
  <summary>More CLI examples:</summary>
  
Train CLIP with ViT-base on COCO Captions dataset:

```
python main.py data=coco model/vision_model=vit-b  model/text_model=vit-b
```

</details>



## Terminology

Below you can see a schematic overview of the different concepts present in the *clhive* Python package.



## Reading The Commits
Here is a reference to what each emoji in the commits means:

* 📎 : Some basic updates.
* ♻️ : Refactoring.
* 💩 : Bad code, needs to be revised!
* 🐛 : Bug fix.
* 💡 : New feature.
* ⚡ : Performance Improvement.
