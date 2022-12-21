from typing import Any, List, Optional, Tuple, Union
import copy
import torch

from . import register_method, BaseMethod
from ..loggers import BaseLogger
from ..models import ContinualModel


def zero_like_params_dict(model: torch.nn.Module):
    """
    Create a list of (name, parameter), where parameter is initialized to zero.
    The list has as many parameters as the model, with the same size.
    :param model: a pytorch model
    """

    return [(k, torch.zeros_like(p).to(p.device)) for k, p in model.named_parameters()]


def copy_params_dict(model: torch.nn.Module, copy_grad=False):
    """
    Create a list of (name, parameter), where parameter is copied from model.
    The list has as many parameters as model, with the same size.
    :param model: a pytorch model
    :param copy_grad: if True returns gradients instead of parameter values
    """

    if copy_grad:
        return [(k, p.grad.data.clone()) for k, p in model.named_parameters()]
    else:
        return [(k, p.data.clone()) for k, p in model.named_parameters()]


@register_method("ewc")
class EWC(BaseMethod):
    def __init__(
        self,
        model: Union[ContinualModel, torch.nn.Module],
        optim: torch.optim,
        train_loader: torch.utils.data.DataLoader,
        logger: Optional[BaseLogger] = None,
        **kwargs,
    ) -> "EWC":
        """_summary_

        Args:
            model (Union[ContinualModel, torch.nn.Module]): _description_
            optim (torch.optim): _description_
            train_loader (torch.utils.data.DataLoader): _description_
            logger (Optional[BaseLogger], optional): _description_. Defaults to None.

        Returns:
            EWC: _description_
        """
        super().__init__(model, optim, logger)

        self.train_loader = train_loader
        self.loss = torch.nn.CrossEntropyLoss()
        self.ewc_lambda = 100

        self.saved_parameters = dict()
        self.importance_matrices = dict()

    @property
    def name(self):
        return "ewc"

    def _compute_importance(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.CrossEntropyLoss,
        optimizer: torch.optim.Optimizer,
        train_loader: torch.utils.data.DataLoader,
        current_task_id: int,
    ) -> torch.FloatTensor:
        """
        Compute EWC importance matrix for each parameter
        """
        model.eval()
        importance_matrix = zero_like_params_dict(model=model)
        device = next(self.model.parameters()).device

        train_loader.sampler.set_task(current_task_id)
        for idx, (data, targets, tasks) in enumerate(train_loader):

            data, targets = data.to(device), targets.to(device)
            # targets -= current_task_id * self.args.n_classes_per_task

            optimizer.zero_grad()
            predictions = model(data, t=tasks)

            loss = criterion(predictions, targets)
            loss.backward()

            for (net_param_name, net_param_value), (
                imp_param_name,
                imp_param_value,
            ) in zip(model.named_parameters(), importance_matrix):
                assert net_param_name == imp_param_name
                if net_param_value.grad is not None:
                    imp_param_value += net_param_value.grad.data.clone().pow(2)

        # average over mini batch length
        for _, imp_param_value in importance_matrix:
            imp_param_value /= float(len(train_loader))

        return importance_matrix

    def _record_state(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.CrossEntropyLoss,
        optimizer: torch.optim.Optimizer,
        train_loader: torch.utils.data.DataLoader,
        current_task_id: int,
    ) -> None:
        # to be called at the end of training each task
        importance_matrix = self._compute_importance(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            train_loader=train_loader,
            current_task_id=current_task_id,
        )

        self.importance_matrices[current_task_id] = importance_matrix
        self.saved_parameters[current_task_id] = copy_params_dict(model)

    def ewc_loss(
        self, model: torch.nn.Module, current_task_id: int
    ) -> torch.FloatTensor:
        if current_task_id == 0:
            return 0

        loss = 0
        for task_id in range(current_task_id):
            for (
                (_, current_parameters),
                (_, saved_parameters),
                (_, importance_weight),
            ) in zip(
                model.named_parameters(),
                self.saved_parameters[task_id],
                self.importance_matrices[task_id],
            ):
                loss += (
                    importance_weight * (current_parameters - saved_parameters).pow(2)
                ).sum()

        return loss

    def process_inc(
        self, features: torch.FloatTensor, y: torch.FloatTensor, t: torch.FloatTensor
    ) -> torch.FloatTensor:
        pred = self.model(x, t)
        loss = self.loss(pred, y)

        return loss

    def observe(
        self, x: torch.FloatTensor, y: torch.FloatTensor, t: torch.FloatTensor
    ) -> torch.FloatTensor:

        features = self.model.forward_backbone(x)

        pred = self.model.forward_head(features, t)
        inc_loss = self.loss(pred, y)

        ewc_loss = self.ewc_loss(
            model=self.model, current_task_id=self._current_task_id
        )

        loss = inc_loss + self.ewc_lambda * ewc_loss
        self.update(loss)

        return loss

    def on_task_end(self) -> None:
        self._record_state(
            model=self.model,
            criterion=self.loss,
            optimizer=self.optim,
            train_loader=self.train_loader,
            current_task_id=self._current_task_id,
        )

        super().on_task_end()
