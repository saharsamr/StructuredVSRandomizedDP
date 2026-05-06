# Projected SGD optimizer

from typing import Callable, Iterable

import torch
from torch import nn
from torch.optim import Optimizer


class DPSGD(Optimizer):
    """
    Implements SDG with momentum, with projections of gradients.

    Parameters:
        params (`Iterable[nn.parameter.Parameter]`):
            Iterable of parameters to optimize or dictionaries defining parameter groups.
        lr (`float`, *optional*, defaults to 0.001):
            The learning rate to use.
        momentum (`float`, *optional*, defaults to `0`):
            beta for momentum parameters (b1, b2).
        dampening (`float`, *optional*, defaults to `0`)
    """

    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter],
        lr: float = 1e-3,
        momentum: float = 0,
        dampening: float = 0,
        no_deprecation_warning: bool = False,
    ):
        #require_version("torch>=1.5.0")  # add_ with alpha
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        defaults = {"lr": lr, "momentum": momentum, "dampening": dampening}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, skip_project: bool = False, closure: Callable = None):
        """
        Performs a single optimization step.

        Arguments:
            zero_moments: If true, zeros out moments (do this when using new SVD)
            closure (`Callable`, *optional*): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:

                state = self.state[p]
                if p.grad is None:
                    continue


                if "rank" in group and "projector" in state and not skip_project:
                    grad = p.proj_grad
                    # Put larger dim first
                    if grad.shape[0] < grad.shape[1]:
                        grad = grad.t()
                else:
                    grad = p.grad

                if "step" not in state:
                    state["step"] = 0
                
                if 'dim' not in group:
                    group['dim'] = 2

                momentum = group['momentum']
                dampening = group['dampening']

                if momentum != 0:
                    if "exp_avg" not in state:  # First iteration
                        state["exp_avg"] = torch.clone(grad).detach()
                    else:
                        state["exp_avg"].mul_(momentum).add_(grad, alpha=1 - dampening)
                    grad = state["exp_avg"]

                state["step"] += 1
                lr = group["lr"]
                
                # GaLore Projection Back
                if "rank" in group and "projector" in state and not skip_project:
                    # Generate projection matrix, use
                    state["projector"].update(p.shape, p.dtype, p.device)
                    grad = state["projector"].project_back(grad)
                    # Delete projection matrix
                    if grad.shape != p.shape:
                        grad = grad.t()
                p.add_(grad, alpha=-lr)

        return loss

    def zero_grad(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    p.grad = None
                    if p.proj_grad is not None:
                        p.proj_grad = None