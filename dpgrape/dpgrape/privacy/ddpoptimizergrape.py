# This file contains a flat clipping optimizer for distributed training with DP-GRAPE, based on opacus.optimizers.ddpoptimizer.py

from __future__ import annotations

from typing import Optional

import torch
from torch.optim import Optimizer

from opacus.optimizers import DistributedDPOptimizer
from opacus.optimizers.optimizer import _check_processed_flag, _mark_as_processed, _generate_noise

from ..rand_projector_dp import RandProjectorDP


class DistributedDPOptimizerRandomProj(DistributedDPOptimizer):
    
    def __init__(
        self,
        optimizer: Optimizer,
        *,
        noise_multiplier: float,
        max_grad_norm: float,
        expected_batch_size: Optional[int],
        loss_reduction: str = "mean",
        generator=None,
        secure_mode: bool = False,
    ):
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

    #@property
    #def accumulated_iterations(self) -> int:
        # Some params may have grad turned off
    #    print("Accumulated iters:", max([p.accumulated_iterations if hasattr(p, "accumulated_iterations") else 0 for p in self.params]), "\n\n\n\n")
    #    return max([p.accumulated_iterations if hasattr(p, "accumulated_iterations") else 0 for p in self.params])

    def add_noise(self):
        # Noise only gets added to the first worker
        if self.rank == 0:
            for group in self.original_optimizer.param_groups:
                for p in group["params"]:
                    state = self.original_optimizer.state[p]
                    _check_processed_flag(p.summed_grad) 
                    noise = _generate_noise(
                        std=self.noise_multiplier * self.max_grad_norm,
                        reference=p.summed_grad,  
                        generator=self.generator,
                        secure_mode=self.secure_mode,
                    )
                    # New for Rand Projection
                    if "rank" in group and "projector" in state:  # GaLore params
                        p.proj_grad = p.summed_grad + noise  # Need additional proj_grad field beacuse it is a different shape than grad
                    else:
                        p.grad = p.summed_grad + noise
                        
                    _mark_as_processed(p.summed_grad)
        else:
            for p in self.params:
                state = self.original_optimizer.state[p]
                if "projector" in state:
                    p.proj_grad = p.summed_grad
                else:
                    p.grad = p.summed_grad.view_as(p)

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
