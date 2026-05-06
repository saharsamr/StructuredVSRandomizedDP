# This file contains modified functions from opacus.grad_sample.linear.py

from typing import Dict, List

import torch
import torch.nn as nn
from opt_einsum import contract

from opacus.grad_sample.utils import register_grad_sampler

from ..rand_projector_dp import RandProjectorDP

@register_grad_sampler(nn.Linear)
def compute_proj_linear_grad_sample(
    layer: nn.Linear, activations: List[torch.Tensor], backprops: torch.Tensor, projector: RandProjectorDP = None
) -> Dict[nn.Parameter, torch.Tensor]:
    """
    Computes per sample gradients for ``nn.Linear`` layer,
    optionally with projection
    Args:
        layer: Layer
        activations: Activations
        backprops: Backpropagations
    """
    activations = activations[0]
    ret = {}
    if layer.weight.requires_grad:
        gs = contract("n...i,n...j->nij", backprops, activations)
        if projector is not None:
            # Generate projection matrix
            projector.generate(layer.weight.shape, layer.weight.dtype, layer.weight.device)
            ret[layer.weight] = projector.project(gs)
            # Delete projection matrix
            projector.clear_projection_matrix()
        else:
            ret[layer.weight] = gs
    if layer.bias is not None and layer.bias.requires_grad:
        ret[layer.bias] = contract("n...k->nk", backprops)
    return ret