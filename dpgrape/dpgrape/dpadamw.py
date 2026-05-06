# Projected AdamW optimizer based on galore_torch.adamw.py

import math
import warnings
from typing import Callable, Iterable, Tuple

import torch
from torch import nn
from torch.optim import Optimizer

from transformers.utils.versions import require_version


class DPAdamW(Optimizer):
    """
    Implements Adam algorithm with weight decay fix as introduced in [Decoupled Weight Decay
    Regularization](https://arxiv.org/abs/1711.05101).

    Parameters:
        params (`Iterable[nn.parameter.Parameter]`):
            Iterable of parameters to optimize or dictionaries defining parameter groups.
        lr (`float`, *optional*, defaults to 0.001):
            The learning rate to use.
        betas (`Tuple[float,float]`, *optional*, defaults to `(0.9, 0.999)`):
            Adam's betas parameters (b1, b2).
        eps (`float`, *optional*, defaults to 1e-06):
            Adam's epsilon for numerical stability.
        weight_decay (`float`, *optional*, defaults to 0.0):
            Decoupled weight decay to apply.
        correct_bias (`bool`, *optional*, defaults to `True`):
            Whether or not to correct bias in Adam (for instance, in Bert TF repository they use `False`).
        no_deprecation_warning (`bool`, *optional*, defaults to `False`):
            A flag used to disable the deprecation warning (set to `True` to disable the warning).
    """

    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
        dp_bias_correction: float = 0,
        no_deprecation_warning: bool = False,
    ):
        if not no_deprecation_warning:
            warnings.warn(
                "This implementation of AdamW is deprecated and will be removed in a future version. Use the PyTorch"
                " implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this"
                " warning",
                FutureWarning,
            )
        require_version("torch>=1.5.0")  # add_ with alpha
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0)")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0)")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay, "correct_bias": correct_bias, "dp_bias_correction": dp_bias_correction}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, zero_moments: bool = False, skip_project: bool = False, closure: Callable = None):
        """
        Performs a single optimization step.

        Arguments:
            zero_moments: If true, zeros out moments (do this when using new SVD) (old)
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

                # State initialization (or zeroing moments when switching subspaces)
                # Also need to reset step so bias correction is done correctly
                if "exp_avg" not in state or zero_moments:
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(grad)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(grad)

                    state["step"] = 0

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                
                # DP-bias correct denom
                if group['dp_bias_correction'] != 0:
                    if "rank" in group and "projector" in state:  # Galore params, correct the bias correction term with singular values
                        galore_dp_bias_correction = torch.pow(torch.sqrt(torch.pow(state["projector"].s, -1) * group['dp_bias_correction']), 2)
                        denom = (torch.max(exp_avg_sq - galore_dp_bias_correction, torch.zeros(exp_avg_sq.shape, device=exp_avg_sq.device) + 5e-8)).sqrt() 
                    else:
                        denom = (torch.max(exp_avg_sq - group['dp_bias_correction'], torch.zeros(exp_avg_sq.shape, device=exp_avg_sq.device) + 5e-8)).sqrt()   
                else:
                    # Original denom
                    denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                if group["correct_bias"]:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                # compute norm gradient
                norm_grad = exp_avg / denom

                # GaLore Projection Back
                if "rank" in group and "projector" in state and not skip_project:
                    # Generate projection matrix, use
                    state["projector"].generate(p.shape, p.dtype, p.device)
                    norm_grad = state["projector"].project_back(norm_grad)
                    # Delete projection matrix
                    state["projector"].clear_projection_matrix()
                    if norm_grad.shape != p.shape:
                        norm_grad = norm_grad.t()
                p.add_(norm_grad, alpha=-step_size)

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                # Add weight decay at the end (fixed version)
                if group["weight_decay"] > 0.0:
                    p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))

                state["done_batch0"] = True

        return loss

    def zero_grad(self, set_to_none: bool = True):  
        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    p.grad = None
                    if p.proj_grad is not None:
                        p.proj_grad = None