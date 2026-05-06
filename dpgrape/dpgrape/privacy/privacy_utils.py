# This file contains modified functions from opacus.grad_sample.utils.py

from typing import Sequence, Type, Union
import torch.nn as nn
from opacus.grad_sample import GradSampleModule, AbstractGradSampleModule, GradSampleModuleExpandedWeights#, GradSampleModuleNoOp
from .projected_grad_sample_module import ProjectedGradSampleModule


from opacus.optimizers.adaclipoptimizer import AdaClipDPOptimizer
from opacus.optimizers.ddp_perlayeroptimizer import (
    DistributedPerLayerOptimizer,
    SimpleDistributedPerLayerOptimizer,
)
from opacus.optimizers.ddpoptimizer import DistributedDPOptimizer
from opacus.optimizers.optimizer import DPOptimizer
from opacus.optimizers.perlayeroptimizer import DPPerLayerOptimizer

from .optimizergrape import DPOptimizerRandomProj
from .perlayeroptimizergrape import DPPerLayerOptimizerRandomProj
from .ddpoptimizergrape import DistributedDPOptimizerRandomProj
from .ddpperlayeroptimizergrape import DistributedDPPerLayerOptimizerRandomProj

from .optimizerrand_noclip import DPOptimizerRandomProjNoClip


def wrap_model(model: nn.Module, grad_sample_mode: str, *args, **kwargs):
    cls = get_gsm_class(grad_sample_mode)
    if grad_sample_mode == "functorch":
        kwargs["force_functorch"] = True
    return cls(model, *args, **kwargs)

def get_gsm_class(grad_sample_mode: str) -> Type[AbstractGradSampleModule]:
    """
    Returns AbstractGradSampleModule subclass correspinding to the input mode.
    See README for detailed comparison between grad sample modes.

    :param grad_sample_mode:
    :return:
    """
    if grad_sample_mode == "projected":
        return ProjectedGradSampleModule
    elif grad_sample_mode in ["hooks", "functorch"]:
        return GradSampleModule
    elif grad_sample_mode == "ew":
        return GradSampleModuleExpandedWeights
    #elif grad_sample_mode == "no_op":
    #    return GradSampleModuleNoOp
    else:
        raise ValueError(
            f"Unexpected grad_sample_mode: {grad_sample_mode}. "
            f"Allowed values: hooks, ew"
        )
        

def get_optimizer_class(clipping: str, random_proj: bool, distributed: bool, grad_sample_mode: str = None):
    if random_proj and clipping == 'per_layer' and distributed is True:
        return DistributedDPPerLayerOptimizerRandomProj
    elif random_proj and clipping == 'flat' and distributed is True:
        return DistributedDPOptimizerRandomProj
    elif random_proj and clipping == 'per_layer' and distributed is False:
        return DPPerLayerOptimizerRandomProj
    elif random_proj and clipping == 'flat' and distributed is False:
        return DPOptimizerRandomProj
    elif clipping == "flat" and distributed is False:
        return DPOptimizer
    elif clipping == "flat" and distributed is True:
        return DistributedDPOptimizer
    elif clipping == "per_layer" and distributed is False:
        return DPPerLayerOptimizer
    elif clipping == "per_layer" and distributed is True:
        if grad_sample_mode == "hooks":
            return DistributedPerLayerOptimizer
        elif grad_sample_mode == "ew":
            return SimpleDistributedPerLayerOptimizer
        else:
            raise ValueError(f"Unexpected grad_sample_mode: {grad_sample_mode}")
    elif clipping == "adaptive" and distributed is False:
        return AdaClipDPOptimizer
    elif clipping == "none":
        return DPOptimizerRandomProjNoClip
    raise ValueError(
        f"Unexpected optimizer parameters. Clipping: {clipping}, distributed: {distributed}"
    )