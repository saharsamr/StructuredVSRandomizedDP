# This file contains a per-layer clipping optimizer for distributed training with DP-GRAPE, based on opacus.optimizers.ddp_perlayeroptimizer.py

from __future__ import annotations

from typing import List, Optional, Callable
from functools import partial

import torch
from torch import nn
from torch.optim import Optimizer

from opacus.optimizers import DistributedDPOptimizer
from opacus.optimizers.optimizer import _generate_noise

from ..rand_projector_dp import RandProjectorDP


def _clip_and_accumulate_parameter(p: nn.Parameter, max_grad_norm: float):
    per_sample_norms = p.grad_sample.view(len(p.grad_sample), -1).norm(2, dim=-1)
    per_sample_clip_factor = (max_grad_norm / (per_sample_norms + 1e-6)).clamp(max=1.0)

    grad = torch.einsum("i,i...", per_sample_clip_factor, p.grad_sample)
    if p.summed_grad is not None:
        p.summed_grad += grad
    else:
        p.summed_grad = grad



class DistributedDPPerLayerOptimizerRandomProj(DistributedDPOptimizer):
    """
    :class:`~opacus.optimizers.optimizer.DPOptimizer` that implements
    per layer clipping strategy with random projections 
    and is compatible with distributed data parallel
    """
    def __init__(
        self,
        optimizer: Optimizer,
        *,
        noise_multiplier: float,
        max_grad_norm: List[float],
        expected_batch_size: Optional[int],
        loss_reduction: str = "mean",
        generator=None,
        secure_mode: bool = False,
    ):
        #self.rank = torch.distributed.get_rank()
        #self.world_size = torch.distributed.get_world_size()
        self.max_grad_norms = max_grad_norm
        max_grad_norm = torch.norm(torch.Tensor(self.max_grad_norms), p=2).item()
        super().__init__(
            optimizer,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            expected_batch_size=expected_batch_size,
            loss_reduction=loss_reduction,
            generator=generator,
            secure_mode=secure_mode,
        )
        # New for Rand Projection
        for p in self.params:
            # For DP-random projection, need a place to store accumulated projected gradients
            p.proj_grad = None
        self._register_hooks()

    @property
    def accumulated_iterations(self) -> int:
        # Some params may have grad turned off
        return max([p.accumulated_iterations if hasattr(p, "accumulated_iterations") else 0 for p in self.params])

    def _add_noise_parameter(self, p: nn.Parameter):
        state = self.original_optimizer.state[p]
        print("MAX GRAD NORM:", self.max_grad_norm)
        noise = _generate_noise(
            std=self.noise_multiplier * self.max_grad_norm,
            reference=p.summed_grad,  
            generator=self.generator,
            secure_mode=self.secure_mode,
        )

        # New for Rand Projection
        if "projector" in state:  # GaLore params
            p.proj_grad = p.summed_grad + noise # Need additional proj_grad field beacuse it is a different shape than grad
        else:
            p.grad = p.summed_grad + noise

    def _scale_grad_parameter(self, p: nn.Parameter):
        state = self.original_optimizer.state[p]
        if not hasattr(p, "accumulated_iterations"):
            p.accumulated_iterations = 0
        p.accumulated_iterations += 1
        if self.loss_reduction == "mean":
            if "projector" in state:
                p.proj_grad /= (
                    self.expected_batch_size * p.accumulated_iterations * self.world_size
                )
            else:
                p.grad /= (
                    self.expected_batch_size * p.accumulated_iterations * self.world_size
                )

    def clip_and_accumulate(self):
        raise NotImplementedError(
            "Clip and accumulate is added per layer in DPDDP Per Layer."
        )

    def add_noise(self):
        raise NotImplementedError("Noise is added per layer in DPDDP Per Layer.")
    
    def _ddp_per_layer_hook(
        self, p: nn.Parameter, max_grad_norm: float, _: torch.Tensor
    ):
        state = self.original_optimizer.state[p]
        _clip_and_accumulate_parameter(p, max_grad_norm)
        # Equivalent to _check_skip_next_step but without popping because it has to be done for every parameter p
        # Equivalent to _check_skip_next_step but without popping because it has to be done for every parameter p
        if self._check_skip_next_step(pop_next=False):
            return


        if self.rank == 0:
            self._add_noise_parameter(p)
        else:
            if "projector" in state:
                p.proj_grad = p.summed_grad
            else:
                p.grad = p.summed_grad
        self._scale_grad_parameter(p)

        return None
    
    def pre_step(
        self, closure: Optional[Callable[[], float]] = None
    ) -> Optional[float]:
        if self._check_skip_next_step():
            self._is_last_step_skipped = True
            return False

        if self.step_hook:
            self.step_hook(self)

        for p in self.params:
            p.accumulated_iterations = 0

        self._is_last_step_skipped = False
        return True
    
    # Modify to handle projected gradients
    def reduce_gradients(self):
        for p in self.params:
            if not p.requires_grad:
                continue
            state = self.original_optimizer.state[p]
            if "projector" in state:
                torch.distributed.all_reduce(p.proj_grad, op=torch.distributed.ReduceOp.SUM)
                if self.loss_reduction == "mean":
                    p.proj_grad /= self.world_size
            else:
                torch.distributed.all_reduce(p.grad, op=torch.distributed.ReduceOp.SUM)
                if self.loss_reduction == "mean":
                    p.grad /= self.world_size
                
        
    def _register_hooks(self):
        for p, max_grad_norm in zip(self.params, self.max_grad_norms):
            if not p.requires_grad:
                continue

            if not hasattr(p, "ddp_hooks"):
                p.ddp_hooks = []

            p.ddp_hooks.append(
                p.register_hook(partial(self._ddp_per_layer_hook, p, max_grad_norm))
            )

    def _register_hooks(self):
        for p, max_grad_norm in zip(self.params, self.max_grad_norms):
            if not p.requires_grad:
                continue

            if not hasattr(p, "ddp_hooks"):
                p.ddp_hooks = []

            p.ddp_hooks.append(
                p.register_hook(partial(self._ddp_per_layer_hook, p, max_grad_norm))
            )

    # New for Rand Projection
    def update_projectors(self, rand_type):
        """
        Create random projector object for every galore layer
        (but don't generate the projection matrices)
        Args:
            rand_type (str) : Type of random projection to use, current options are 'orthonormal' and 'gaussian'
        Returns:
            None
        """
        for group in self.original_optimizer.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if "rank" in group:
                    if "projector" not in state:
                        state["projector"] = RandProjectorDP(group["rank"], scale=group["scale"], proj_type=group["proj_type"], rand_type=rand_type)    
                    state["projector"].update_seed()
                    