# Not DP, for SVD experiment

from __future__ import annotations

from typing import List, Optional, Callable

import torch
from opacus.optimizers.utils import params
from opt_einsum import contract
from torch.optim import Optimizer

from opacus.optimizers import DPOptimizer
from opacus.optimizers.optimizer import _check_processed_flag, _mark_as_processed, _generate_noise

from ..rand_projector_dp import RandProjectorDP


class DPOptimizerRandomProjNoClip(DPOptimizer):

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


    def add_noise(self):
        """
        Add noise to clipped gradients, divide by batch size
        put result in p.proj_grad for galore params or p.grad for non-galore params
        Args:
            None
        Returns:
            None
        """
        for group in self.original_optimizer.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    _check_processed_flag(p.summed_grad) 
                    state = self.original_optimizer.state[p]
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

    def scale_grad(self):
        """
        Applies given ``loss_reduction`` to ``p.grad`` or ``p.proj_grad``.

        Does nothing if ``loss_reduction="sum"``. Divides gradients by
        ``self.expected_batch_size`` if ``loss_reduction="mean"``
        """
        if self.loss_reduction == "mean":
            for group in self.original_optimizer.param_groups:
                for p in group["params"]:
                    if p.grad is not None:
                        state = self.original_optimizer.state[p]
                        # New for Rand Projection
                        if "rank" in group and "projector" in state:
                            p.proj_grad /= self.expected_batch_size * self.accumulated_iterations
                        else:
                            p.grad /= self.expected_batch_size * self.accumulated_iterations

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

    def zero_grad(self, set_to_none: bool = False):
        """
        Clear gradients.
        Clears ``p.grad``, ``p.grad_sample``, ``p.proj_grad`` and ``p.summed_grad``.
        Notes:
            ``set_to_none`` argument only affects ``p.grad``. ``p.grad_sample`` and
            ``p.summed_grad`` is never zeroed out and always set to None.
            Normal grads can do this, because their shape is always the same.
            Grad samples do not behave like this, as we accumulate gradients from different
            batches in a list
        Args:
            set_to_none: instead of setting to zero, set the grads to None. (only
            affects regular gradients. Per sample gradients are always set to None)
        """
        for p in self.params:
            p.grad_sample = None
            p.proj_grad = None   # New for Rand Projection
            
            if not self._is_last_step_skipped:
                p.summed_grad = None

        self.original_optimizer.zero_grad()

    def clip_and_accumulate(self):
        """
        Doesn't actually clip.
        Stores aggregated gradients into `p.summed_grad```
        """

        #if len(self.grad_samples[0]) == 0:
            # Empty batch
        #    per_sample_clip_factor = torch.zeros(
        #        (0,), device=self.grad_samples[0].device
        #    )
        #else:
        #    per_param_norms = [
        #        g.reshape(len(g), -1).norm(2, dim=-1) for g in self.grad_samples
        #    ]
        #    per_sample_norms = torch.stack(per_param_norms, dim=1).norm(2, dim=1)
        #    per_sample_clip_factor = (
        #        self.max_grad_norm / (per_sample_norms + 1e-6)
        #    ).clamp(max=1.0)

        for p in self.params:
            _check_processed_flag(p.grad_sample)
            grad_sample = self._get_flat_grad_sample(p)
            grad = torch.sum(grad_sample, dim=0)
            #grad = torch.einsum("i,i...", per_sample_clip_factor, grad_sample)

            if p.summed_grad is not None:
                p.summed_grad += grad
            else:
                p.summed_grad = grad

            _mark_as_processed(p.grad_sample)