# This file contains modified functions from opacus.grad_sample.grad_sample_module.py

from __future__ import annotations

from functools import partial

import torch
import torch.nn as nn
from opacus.grad_sample.functorch import ft_compute_per_sample_gradient, prepare_layer
from opacus.utils.module_utils import (
    has_trainable_params,
    trainable_modules
)
from typing import Iterable
from opacus.layers.dp_rnn import DPGRU, DPLSTM, DPRNN, RNNLinear

from opacus.grad_sample import GradSampleModule, create_or_accumulate_grad_sample

from opacus.grad_sample.linear import compute_linear_grad_sample
from .proj_linear_grad_sampler import compute_proj_linear_grad_sample

from ..rand_projector_dp import RandProjectorDP

class ProjectedGradSampleModule(GradSampleModule):

    def __init__(
        self,
        m: nn.Module,
        *,
        batch_first=True,
        loss_reduction="mean",
        strict: bool = True,
        force_functorch=False,
    ):
        """

        Args:
            m: nn.Module to be wrapped
            batch_first: Flag to indicate if the input tensor to the corresponding module
                has the first dimension representing the batch. If set to True, dimensions on
                input tensor are expected be ``[batch_size, ...]``, otherwise
                ``[K, batch_size, ...]``
            loss_reduction: Indicates if the loss reduction (for aggregating the gradients)
                is a sum or a mean operation. Can take values "sum" or "mean"
            strict: If set to ``True``, the input module will be validated to check that
                ``GradSampleModule`` has grad sampler functions for all submodules of
                the input module (i.e. if it knows how to calculate per sample gradients)
                for all model parameters. If set to ``False``, per sample gradients will
                be computed on "best effort" basis - they will be available where
                possible and set to None otherwise. This is not recommended, because
                some unsupported modules (e.g. BatchNorm) affect other parameters and
                invalidate the concept of per sample gradients for the entire model.
            force_functorch: If set to ``True``, will use functorch to compute
                all per sample gradients. Otherwise, functorch will be used only
                for layers without registered grad sampler methods.

        Raises:
            NotImplementedError
                If ``strict`` is set to ``True`` and module ``m`` (or any of its
                submodules) doesn't have a registered grad sampler function.
        """
        # Overwrite opacus linear grad sampler with projected one
        compute_linear_grad_sample = compute_proj_linear_grad_sample  
        self.projectors = []
        self.skip_proj_module_names = ["lm_head", "head"]
        super().__init__(
            m,
            batch_first=batch_first,
            loss_reduction=loss_reduction,
            strict=strict,
            force_functorch=False,
        )

    # Gives a little more control over what modules to include projected linear hooks for
    # ex: for OPT models, want to just use regular linear hook for lm_head
    def iterate_submodules_with_names(self, module: nn.Module, module_name: str = '') -> Iterable[tuple[str, nn.Module]]:
        if has_trainable_params(module):
            yield (module_name, module)

        # Don't recurse if module is handled by functorch
        if (
            has_trainable_params(module)
            and type(module) not in self.GRAD_SAMPLERS
            and type(module) not in [DPRNN, DPLSTM, DPGRU]
        ):
            return

        for name, m in module.named_children():
            full_name = f"{module_name}.{name}" if module_name else name
            yield from self.iterate_submodules_with_names(m, full_name)

    def add_hooks(
        self,
        *,
        loss_reduction: str = "mean",
        batch_first: bool = True,
        force_functorch: bool = False,
        skip_functorch_layer_prep: bool = False,
    ) -> None:
        """
        Adds hooks to model to save activations and backprop values.
        The hooks will
        1. save activations into param.activations during forward pass
        2. compute per-sample gradients in params.grad_sample during backward pass.
        Call ``remove_hooks(model)`` to disable this.

        Args:
            model: the model to which hooks are added
            batch_first: Flag to indicate if the input tensor to the corresponding module
                has the first dimension representing the batch. If set to True, dimensions on
                input tensor are expected be ``[batch_size, ...]``, otherwise
                ``[K, batch_size, ...]``
            loss_reduction: Indicates if the loss reduction (for aggregating the gradients)
                is a sum or a mean operation. Can take values "sum" or "mean"
            force_functorch: If set to ``True``, will use functorch to compute all per sample gradients.
                Otherwise, functorch will be used only for layers without registered grad sampler methods.
        """
        if hasattr(self._module, "autograd_grad_sample_hooks"):
            raise ValueError("Trying to add hooks twice to the same model")
        else:
            self._module.autograd_grad_sample_hooks = []
            self.autograd_grad_sample_hooks = self._module.autograd_grad_sample_hooks

        for module_idx, (module_name, module) in enumerate(self.iterate_submodules_with_names(self._module)):
            self.projectors.append(None)
            # Do not add hooks to DPRNN, DPLSTM or DPGRU as the hooks are handled by the `RNNLinear`
            if type(module) in [DPRNN, DPLSTM, DPGRU]:
                continue

            if force_functorch or not type(module) in self.GRAD_SAMPLERS:
                prepare_layer(module, batch_first=batch_first)

            self.autograd_grad_sample_hooks.append(
                module.register_forward_hook(self.capture_activations_hook)
            )

            if isinstance(module, nn.Linear) and not any(target_key in module_name for target_key in self.skip_proj_module_names):
                self.autograd_grad_sample_hooks.append(
                        module.register_backward_hook(
                            partial(
                                self.capture_backprops_hook,
                                loss_reduction=loss_reduction,
                                batch_first=batch_first,
                                projector=self.projectors[module_idx]
                            )
                        )
                    )

            else:
                self.autograd_grad_sample_hooks.append(
                    module.register_backward_hook(
                        partial(
                            self.capture_backprops_hook,
                            loss_reduction=loss_reduction,
                            batch_first=batch_first,
                        )
                    )
                )

        self.enable_hooks()

    def remove_hooks(self, keep_ddp_hooks: bool = False) -> None:
        """
        Removes hooks added by ``add_hooks()``
        """
        self.disable_hooks()
        if not keep_ddp_hooks:
            for p in self.parameters():
                if hasattr(p, "ddp_hooks"):
                    while p.ddp_hooks:
                        handle = p.ddp_hooks.pop()
                        handle.remove()
                    delattr(p, "ddp_hooks")
        if not hasattr(self, "autograd_grad_sample_hooks"):
            raise ValueError("Asked to remove hooks, but no hooks found")
        else:
            while self.autograd_grad_sample_hooks:
                handle = self.autograd_grad_sample_hooks.pop()
                handle.remove()
            delattr(self, "autograd_grad_sample_hooks")
            delattr(self._module, "autograd_grad_sample_hooks")
        # Remove functorch hooks
        for _module_name, module in trainable_modules(self._module):
            if hasattr(module, "ft_compute_sample_grad"):
                delattr(module, "ft_compute_sample_grad")

    def capture_backprops_hook(
        self,
        module: nn.Module,
        _forward_input: torch.Tensor,
        forward_output: torch.Tensor,
        loss_reduction: str,
        batch_first: bool,
        projector: RandProjectorDP = None
    ):
        """
        Computes per sample gradients given the current backprops and activations
        stored by the associated forward hook. Computed per sample gradients are
        stored in ``grad_sample`` field in each parameter.

        For non-recurrent layers the process is straightforward: for each
        ``loss.backward()`` call this hook will be called exactly one. For recurrent
        layers, however, this is more complicated and the hook will be called multiple
        times, while still processing the same batch of data.

        For this reason we first accumulate the gradients from *the same batch* in
        ``p._current_grad_sample`` and then, when we detect the end of a full backward
        pass - we store accumulated result on ``p.grad_sample``.

        From there, ``p.grad_sample`` could be either a Tensor or a list of Tensors,
        if accumulated over multiple batches

        Args:
            module: nn.Module,
            _forward_input: torch.Tensor,
            forward_output: torch.Tensor,
            loss_reduction: str,
            batch_first: bool,
            projector: RandProjectorDP
        """
        if not self.hooks_enabled:
            return
        backprops = forward_output[0].detach()
        activations, backprops = self.rearrange_grad_samples(
            module=module,
            backprops=backprops,
            loss_reduction=loss_reduction,
            batch_first=batch_first,
        )
        if not self.force_functorch and type(module) in self.GRAD_SAMPLERS:
            grad_sampler_fn = self.GRAD_SAMPLERS[type(module)]
        else:
            grad_sampler_fn = ft_compute_per_sample_gradient

        if projector is not None:
            grad_samples = grad_sampler_fn(module, activations, backprops, projector)
        else:
            grad_samples = grad_sampler_fn(module, activations, backprops)

        # We can skip the create_or_accumulate_grad_sample and promote_current_grad_sample
        # functions because we are not using recurrent layers?
        for param, gs in grad_samples.items():
            if param.requires_grad:
                param.grad_sample = gs

        if len(module.activations) == 0:
            if hasattr(module, "max_batch_len"):
                del module.max_batch_len


    def update_projectors(self, optimizer):
        #target_modules_list = ["attn", "mlp"] # Don't think this is necessary
        # Collect projectors from the optimizer
        #target_modules = ["attn", "mlp", "attention", "dense"]
        optimizer_projectors = []
        for group in optimizer.original_optimizer.param_groups:
            for p in group["params"]:
                state = optimizer.original_optimizer.state[p]
                if "rank" in group:
                    optimizer_projectors.append(state["projector"])
        # Assign projectors to corresponding layer
        projector_count = 0
        module_idx = 0
        for module_name, module in self.iterate_submodules_with_names(self._module):
            if not has_trainable_params(module):
                continue
            if isinstance(module, nn.Linear) and not any(target_key in module_name for target_key in self.skip_proj_module_names):
                print
                self.projectors[module_idx] = optimizer_projectors[projector_count]
                projector_count += 1
            module_idx += 1


    # Call this function to clear all projectors
    def clear_projectors(self):
        module_idx = 0
        for module_name, module in self.iterate_submodules_with_names(self._module):
            if not has_trainable_params(module):
                continue
            if isinstance(module, nn.Linear) and not any(target_key in module_name for target_key in self.skip_proj_module_names):
                self.projectors[module_idx] = None
            module_idx += 1
