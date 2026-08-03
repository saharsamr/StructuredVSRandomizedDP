"""The Trainer class, to easily train a 🤗 Transformers from scratch or finetune it on a new task."""

import collections
import inspect
import math
import os
import re
import shutil
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from packaging import version
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import (
    BatchSampler,
    RandomSampler,
    SequentialSampler,
    SubsetRandomSampler,
)
from torch.optim.lr_scheduler import LambdaLR
import math
import time

import transformers
from transformers.file_utils import is_datasets_available, is_in_notebook #, is_torch_tpu_available
#from transformers.utils import is_torch_tpu_available
#from transformers.utils import is_torch_xla_available
from transformers.integrations import (
    is_comet_available,
    is_optuna_available,
    is_ray_available,
    is_tensorboard_available,
    is_wandb_available,
)
from transformers.optimization import AdamW, get_linear_schedule_with_warmup, get_scheduler

from transformers.trainer_callback import (
    DefaultFlowCallback,
    ProgressCallback,
)
from transformers.trainer_utils import (
    default_compute_objective,
)
from transformers.training_args import TrainingArguments
from transformers.utils import logging
from logging import FileHandler, Formatter
from transformers.trainer_utils import TrainOutput

from tqdm import tqdm, trange
from torch.optim import SGD
import torch.nn.functional as F

from .linearhead_trainer import LinearHeadTrainer
from . import wandb_utils
from .canary import CanaryProbe, RecordingBatchSampler
from transformers.trainer_callback import TrainerState

import copy

from opacus.accountants.utils import get_noise_multiplier
from opacus.distributed import DifferentiallyPrivateDistributedDataParallel as DPDDP
from opacus import PrivacyEngine

from dpgrape.dpadamw import DPAdamW as DPGrapeAdamW
from dpgrape.privacy.privacy_engine_modified import PrivacyEngineModified
from dpgrape.low_rank_projector_dp import (
    attach_low_rank_projectors,
    iter_low_rank_projectors,
)
from dpgrape.private_subspace import PerLayerGaussianMechanism

from opacus.utils.module_utils import has_trainable_params

from accelerate import Accelerator


_use_native_amp = False
_use_apex = False

DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

if is_in_notebook():
    from transformers.utils.notebook import NotebookProgressCallback

    DEFAULT_PROGRESS_CALLBACK = NotebookProgressCallback

# Check if Pytorch version >= 1.6 to switch between Native AMP and Apex
if version.parse(torch.__version__) < version.parse("1.6"):
    from transformers.file_utils import is_apex_available

    if is_apex_available():
        from apex import amp
    _use_apex = True
else:
    _use_native_amp = True
    from torch.cuda.amp import autocast

if version.parse(torch.__version__) < version.parse("1.2"):
    _use_ddp_no_sync = False
else:
    _use_ddp_no_sync = True

if is_datasets_available():
    import datasets

#if is_torch_xla_available():
#    import torch_xla.core.xla_model as xm
#    import torch_xla.debug.metrics as met
#    import torch_xla.distributed.parallel_loader as pl

if is_tensorboard_available():
    from transformers.integrations import TensorBoardCallback

    DEFAULT_CALLBACKS.append(TensorBoardCallback)


if is_wandb_available():
    from transformers.integrations import WandbCallback

    DEFAULT_CALLBACKS.append(WandbCallback)

if is_comet_available():
    from transformers.integrations import CometCallback

    DEFAULT_CALLBACKS.append(CometCallback)

if is_optuna_available():
    import optuna

if is_ray_available():
    from ray import tune

logger = logging.get_logger(__name__)
logger.setLevel(logging.INFO)

########## The above part is copied from Transformers' trainer (3.4.0) ##########

def default_dev_objective(metrics):
    """
    Objective used for picking the best model on development sets
    """
    if "eval_mnli/acc" in metrics:
        return metrics["eval_mnli/acc"]
    elif "eval_mnli-mm/acc" in metrics:
        return metrics["eval_mnli-mm/acc"]
    elif "eval_f1" in metrics:
        return metrics["eval_f1"]
    elif "eval_mcc" in metrics:
        return metrics["eval_mcc"]
    elif "eval_pearson" in metrics:
        return metrics["eval_pearson"]
    elif "eval_acc" in metrics:
        return metrics["eval_acc"]

    raise Exception("No metric founded for {}".format(metrics))


def dpzero_clip(loss_diff, C=1.):
    tmp = torch.min(torch.ones_like(loss_diff), torch.div(C * torch.ones_like(loss_diff), torch.abs(loss_diff)))
    return torch.mul(tmp, loss_diff)

class Trainer(LinearHeadTrainer):
    """
    Adding some functions based on Transformers' Trainer class.
    """

    # Set by roberta_finetuning_run.py when --canary_probe is on. Scored after every
    # subspace update; see roberta_utils/canary.py for what it measures.
    canary_probe = None
    # Set alongside subspace_dataloader for the dev-split modes; None under 'shared',
    # where the subspace comes from the training batch and there is no dev draw to record.
    subspace_batch_sampler = None
    # Size of the pool the subspace samples from -- len(dev) normally, less when the canary
    # probe fences off a holdout. Drives the private-skip sampling rate.
    subspace_pool_size = None

    @property
    def low_rank_method(self):
        """'galore', 'subtrack', or None -- which data-driven subspace to track.

        Under --oracle_batch_mode shared/skip both are eps = infinity as a whole: the
        subspace comes from bare (unclipped, unnoised) batch gradients and only the weight
        updates are DP. Under private-skip the subspace is itself a DP release. See
        DPTRACK_DESIGN.md and ``subspace_is_private``.
        """
        if getattr(self.args, "dpgalore", False):
            return "galore"
        if getattr(self.args, "dptrack", False):
            return "subtrack"
        return None

    @property
    def subspace_is_private(self):
        """True when the subspace is estimated from a privatized gradient, not a bare one.

        private-skip clips per-sample gradients per layer, noises the sum, and only then
        runs the SVD / geodesic step. The subspace is then post-processing of a DP output,
        so the run has a finite eps -- see ``setup_private_subspace_mechanism`` for how it
        is accounted.
        """
        return (
            self.low_rank_method is not None
            and getattr(self.args, "oracle_batch_mode", None) == "private-skip"
        )

    @property
    def subspace_uses_dev_split(self):
        """True for the two modes that draw subspace batches from the held-out dev split."""
        return getattr(self.args, "oracle_batch_mode", None) in ("skip", "private-skip")

    @property
    def uses_projected_dp(self):
        """True for every method that runs the projected opacus pipeline."""
        return self.args.dpgrape or self.low_rank_method is not None

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Based on Transformers' default one, we add fixing layer option where the bottom n layers' parameters
        are fixed and only the top layers are further fine-tuned.
        """
        if self.args.hf_inference_model:
            return
        
        # Following params have requires_grad = True but gradients are not computed for them
        # If requires_grad is not turned off for them there is a memory leakage issue with opacus at
        # every step that may eventually cause CUDA OOM
        target_params = ["roberta.pooler.dense.weight", "roberta.pooler.dense.bias", 
                         "classifier.weight", "classifier.bias"]
        for name, param in self.model.named_parameters():
            if name in target_params:
                param.requires_grad = False

        if self.optimizer is None:
            params = {}
            for n, p in self.model.named_parameters():
                if p.requires_grad:
                    if self.args.fix_layers > 0:
                        if 'encoder.layer' in n:
                            try:
                                layer_num = int(n[n.find('encoder.layer') + 14:].split('.')[0])
                            except:
                                print(n)
                                raise Exception("")
                            if layer_num >= self.args.fix_layers:
                                print('yes', n)
                                params[n] = p
                            else:
                                print('no ', n)
                        elif 'embeddings' in n:
                            print('no ', n)
                        else:
                            print('yes', n)
                            params[n] = p
                    else:
                        params[n] = p
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in params.items() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in params.items() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
            if self.uses_projected_dp:
                galore_params = []
                target_modules = ["attn", "attention", "dense", "mlp"]
                skip_modules = ["lm_head"]
                for module_name, module in self.model.named_modules():
                    if not isinstance(module, torch.nn.Linear) \
                        or not any(target_key in module_name for target_key in target_modules) \
                        or any(key in module_name for key in skip_modules):
                            continue 
                    if module.weight.requires_grad:
                        galore_params.append(module.weight)
                id_galore_params = [id(p) for p in galore_params]
                logger.info(f"Total params with GaLore enabled: {sum(p.numel() for p in galore_params) / 1_000_000:.2f}M")
                
                # Make parameters without "rank" to another group
                regular_params = [p for p in self.model.parameters() if id(p) not in id_galore_params and p.requires_grad]
                param_groups = [{'params': regular_params}, 
                                {'params': galore_params, 'rank': self.args.subspace_r, 'update_proj_gap': self.args.subspace_T, 'scale': 1.0, 'proj_type': 'std'}]
                self.optimizer = DPGrapeAdamW(param_groups, 
                            lr=self.args.learning_rate,
                            betas=(self.args.adam_beta1, self.args.adam_beta2),
                            eps=self.args.adam_epsilon)
            elif self.args.optimizer == 'adam':
                self.optimizer = AdamW(
                    optimizer_grouped_parameters,
                    lr=self.args.learning_rate,
                    betas=(self.args.adam_beta1, self.args.adam_beta2),
                    eps=self.args.adam_epsilon,
                )
            elif self.args.optimizer == 'sgd':
                self.optimizer = SGD(
                    optimizer_grouped_parameters,
                    lr=self.args.learning_rate
                )
            else:
                raise NotImplementedError
        if self.lr_scheduler is None:
            self.lr_scheduler = get_scheduler(
                self.args.lr_scheduler_type,
                optimizer=self.optimizer,
                num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                num_training_steps=num_training_steps,
            )

    def should_optim(self, name, param):
        return (not self.args.layer_wise_optim or f".{self.state.global_step % self.model.config.num_hidden_layers}." in name) and param.requires_grad


    def zo_forward(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        model.eval()
        inputs = self._prepare_inputs(inputs)
        if self.args.optimize_acc:
            loss, logits = model(**inputs)
            preds = F.softmax(logits, dim=-1)
            acc = torch.sum(torch.argmax(preds, 1) == inputs['labels']) / len(preds)
            loss = -acc
        else:
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            if self.args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
        self.state.zo_forward_step += 1
        return loss.detach()


    def efficient_perturb_parameters(self, model: nn.Module, random_seed: int, scaling_factor=1):
        torch.manual_seed(random_seed)
        for name, param in self.named_parameters_to_optim:
            z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            param.data = param.data + scaling_factor * z * self.args.zero_order_eps
        return model

    
    def perturb_parameters(self, model: nn.Module, random_vector=None, scaling_factor=1):
        if random_vector is None:
            random_vector = {}

        for name, param in self.named_parameters_to_optim:
            if name in random_vector:
                z = random_vector[name]
            else:
                z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
                random_vector[name] = z
            param.data = param.data + scaling_factor * z * self.args.zero_order_eps

        return model, random_vector


    def get_num_samples(self):
        if self.args.zero_order_sample_scheduler is None:
            noise_sample_time = 1 
        elif self.args.zero_order_sample_scheduler == "linear":
            noise_sample_time = max(1, int(self.state.global_step / self.args.max_steps * self.args.zero_order_sample))
        elif self.args.zero_order_sample_scheduler == "constant":
            noise_sample_time = int(self.args.zero_order_sample)
        else:
            raise NotImplementedError

        return noise_sample_time

    def train(self, model_path=None, dev_objective=None):
        """
        Main training entry point.

        The training logic is directly borrowed from transformers.Trainer (version 3.0.2).
        Add early stopping.
        """

        # Set up a file handler to write logs to a file
        file_handler = FileHandler(self.args.log_file)
        file_handler.setLevel(logging.INFO)  # Set the desired log level, e.g., INFO, DEBUG, etc.

        # Create a formatter and set it for the file handler
        formatter = Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        # Add the file handler to the logger
        logger.addHandler(file_handler)


        if self.args.from_linearhead and model_path is None:
            super().train(model_path, dev_objective) # Train output layer using LinearHeadTrainer

        self.best_dir = None
        self.objective = -float("inf")
        self.dev_objective = dev_objective if dev_objective is not None else default_dev_objective

        # Data loading.
        train_dataloader = self.get_train_dataloader()
        num_update_steps_per_epoch = len(train_dataloader) // self.args.gradient_accumulation_steps
        if num_update_steps_per_epoch == 0:
            num_update_steps_per_epoch = 1
        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            num_train_epochs = self.args.max_steps // num_update_steps_per_epoch + int(
                self.args.max_steps % num_update_steps_per_epoch > 0
            )
        else:
            t_total = int(len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs)
            num_train_epochs = self.args.num_train_epochs

        self.create_optimizer_and_scheduler(num_training_steps=t_total)
        optimizer = self.optimizer
        scheduler = self.lr_scheduler

        # Check if saved optimizer or scheduler states exist
        if (
            model_path is not None
            and os.path.isfile(os.path.join(model_path, "optimizer.pt"))
            and os.path.isfile(os.path.join(model_path, "scheduler.pt"))
        ):
            # Load in optimizer and scheduler states
            # weights_only=False: these are locally-written trainer states that contain
            # plain Python objects; torch>=2.6 defaults to True and would reject them.
            optimizer.load_state_dict(
                torch.load(
                    os.path.join(model_path, "optimizer.pt"),
                    map_location=self.args.device,
                    weights_only=False,
                )
            )
            scheduler.load_state_dict(
                torch.load(os.path.join(model_path, "scheduler.pt"), weights_only=False)
            )

        model = self.model

        if self.args.fp16 and _use_apex:
            if not transformers.is_apex_available():
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, optimizer = amp.initialize(model, optimizer, opt_level=self.args.fp16_opt_level)


        # Multi-gpu training (should be after apex fp16 initialization)
        if "LOCAL_RANK" in os.environ:
            if self.uses_projected_dp or self.args.dpadam:
                model = DPDDP(model)
            else:
                model = torch.nn.parallel.DistributedDataParallel(
                    model,
                    device_ids=[self.args.local_rank],
                    output_device=self.args.local_rank,
                    find_unused_parameters=True,
                )

        """
        # Distributed training (should be after apex fp16 initialization)
        if self.args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.args.local_rank],
                output_device=self.args.local_rank,
                find_unused_parameters=True,
            )"""

        # Train
        if transformers.is_torch_tpu_available():
            total_train_batch_size = self.args.train_batch_size * xm.xrt_world_size()
        else:
            total_train_batch_size = (
                self.args.train_batch_size
                * self.args.gradient_accumulation_steps
                * (torch.distributed.get_world_size() if "LOCAL_RANK" in os.environ else 1)
            )
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", self.num_examples(train_dataloader))
        logger.info("  Num Epochs = %d", num_train_epochs)
        logger.info("  Instantaneous batch size per device = %d", self.args.per_device_train_batch_size)
        logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", total_train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

        wandb_utils.update_config({
            "num_train_examples": self.num_examples(train_dataloader),
            "num_train_epochs": num_train_epochs,
            "total_train_batch_size": total_train_batch_size,
            "total_optimization_steps": t_total,
            "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        })

        self.state = TrainerState()
        self.state.global_step = 0
        start_time = time.time()
        self.state.zo_forward_step = 0
        self.epoch = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0

        if self.args.gradient_checkpointing:
            model.gradient_checkpointing_enable()

        # Check if continuing training from a checkpoint
        if model_path is not None:
            # set global_step to global_step of last saved checkpoint from model path
            try:
                self.state.global_step = int(model_path.split("-")[-1].split("/")[0])
                epochs_trained = self.state.global_step // (len(train_dataloader) // self.args.gradient_accumulation_steps)
                steps_trained_in_current_epoch = self.state.global_step % (
                    len(train_dataloader) // self.args.gradient_accumulation_steps
                )

                logger.info("  Continuing training from checkpoint, will skip to saved global_step")
                logger.info("  Continuing training from epoch %d", epochs_trained)
                logger.info("  Continuing training from global step %d", self.state.global_step)
                logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
            except ValueError:
                self.state.global_step = 0
                logger.info("  Starting fine-tuning.")

        tr_loss = torch.tensor(0.0).to(self.args.device)
        logging_loss_scalar = 0.0
        model.zero_grad()
        metrics = None

        # set up dp parameters:
        if self.args.dpzero or self.uses_projected_dp or self.args.dpadam:
            if "LOCAL_RANK" in os.environ:
                k = model.module.num_k
                num_labels = model.module.num_labels
            else:
                k = model.num_k
                num_labels = model.num_labels
            sample_rate= total_train_batch_size / (k *  num_labels)
            multiplier = get_noise_multiplier(target_epsilon=self.args.dp_epsilon,
                                              target_delta=self.args.dp_delta,
                                              steps=t_total,
                                              sample_rate=sample_rate,
                                              )
            print("NOISE MULTIPLIER: ", multiplier)
            self.dpzero_gaussian_std = 2 * multiplier * self.args.dp_clip_threshold / total_train_batch_size

            # The accountant's calibration is the thing you most often need to check after
            # the fact, so it goes in the run config rather than only stdout.
            wandb_utils.update_config({
                "noise_multiplier": multiplier,
                "sample_rate": sample_rate,
                "dpzero_gaussian_std": self.dpzero_gaussian_std,
            })

        # DP-Grape / DP-GaLore / DPTrack-Oracle - use opacus
        if self.uses_projected_dp:
            privacy_engine = PrivacyEngineModified()
            model.train()  # Required for opacus

            if self.args.dp_clip_strategy == 'per_layer':
                # Layerwise clipping: Each layer has the same clipping threshold. The total grad norm is still bounded by `args.clip_C`.
                n_layers = len([(n, p) for n, p in model.named_parameters() if p.requires_grad])
                max_grad_norm_list = [self.args.dp_clip_threshold / np.sqrt(n_layers)] * n_layers
                model, optimizer, train_dataloader = privacy_engine.make_private(
                    module=model,
                    optimizer=optimizer,
                    data_loader=train_dataloader,
                    noise_multiplier=multiplier,
                    max_grad_norm=max_grad_norm_list,
                    clipping="per_layer",
                    poisson_sampling=False,
                    random_proj=True,
                    grad_sample_mode="projected",
                )
            elif self.args.dp_clip_strategy == 'flat':
                model, optimizer, train_dataloader = privacy_engine.make_private(
                        module=model,
                        optimizer=optimizer,
                        data_loader=train_dataloader,
                        noise_multiplier=multiplier,
                        max_grad_norm=self.args.dp_clip_threshold,
                        clipping="flat",
                        poisson_sampling=False,
                        random_proj=True,
                        grad_sample_mode="projected",
                    )
            if self.low_rank_method is None:
                optimizer.update_projectors('gaussian')
                model.update_projectors(optimizer)
                model.remove_hooks(keep_ddp_hooks=True)
                model.add_hooks()
            else:
                method_label = 'DP-GaLore' if self.low_rank_method == 'galore' else 'DPTrack'
                if self.subspace_is_private:
                    logger.warning(
                        "*** %s (private-skip): the subspace is estimated from per-layer "
                        "clipped, noised dev-split gradients, so it is post-processing of a "
                        "DP release and the run has a FINITE eps. See the privacy/* config "
                        "keys for the accounting. ***",
                        method_label,
                    )
                else:
                    logger.warning(
                        "*** %s-Oracle: the subspace is estimated from BARE batch gradients. "
                        "The weight updates are DP but the subspace is not, so the method as "
                        "a whole is eps = infinity. Report it separately from DP-GRAPE. "
                        "Use --oracle_batch_mode private-skip for the DP-legal version. ***",
                        method_label,
                    )
                attach_low_rank_projectors(
                    optimizer,
                    model,
                    method=self.low_rank_method,
                    subspace_update_interval=self.args.subspace_T,
                    st_step_size=self.args.st_step_size,
                    st_step_size_scheduler=(
                        self.args.st_step_size_scheduler
                        if self.args.st_step_size_scheduler != 'constant' else None
                    ),
                )
                # Batch used for the subspace backward at subspace update steps.
                if self.subspace_uses_dev_split:
                    # Held-out dev split: disjoint from train.tsv and from test.tsv by
                    # construction (generate_k_shot_data.py writes [:k] to train and
                    # [k:2k] to dev). private-skip's privacy accounting depends on that
                    # disjointness -- see setup_private_subspace_mechanism.
                    assert self.eval_dataset is not None, \
                        f'--oracle_batch_mode {self.args.oracle_batch_mode} requires --do_eval'
                    # An explicit BatchSampler, not batch_size + sampler: DataLoader builds
                    # exactly this internally, so the draw is unchanged and the accountant's
                    # fixed batch size still holds, but wrapping it lets the canary probe
                    # read which dev examples fed each subspace update.
                    #
                    # With the probe on, the base sampler is restricted to the probe's pool,
                    # fencing the holdout canaries out of every subspace batch. That is what
                    # makes them a control the sampler cannot exhaust. self.subspace_pool_size
                    # carries the reduced size to the accountant.
                    if self.canary_probe is not None:
                        base_sampler = SubsetRandomSampler(
                            self.canary_probe.subspace_pool_indices
                        )
                        self.subspace_pool_size = len(self.canary_probe.subspace_pool_indices)
                    else:
                        base_sampler = RandomSampler(self.eval_dataset)
                        self.subspace_pool_size = len(self.eval_dataset)
                    self.subspace_batch_sampler = RecordingBatchSampler(
                        BatchSampler(
                            base_sampler,
                            batch_size=self.args.train_batch_size,
                            # drop_last keeps the batch size fixed, which the accountant assumes.
                            drop_last=True,
                        )
                    )
                    self.subspace_dataloader = DataLoader(
                        self.eval_dataset,
                        batch_sampler=self.subspace_batch_sampler,
                        collate_fn=self.data_collator,
                    )
                else:
                    self.subspace_batch_sampler = None
                    self.subspace_dataloader = train_dataloader
                self.subspace_iter = None
                self.subspace_batches_consumed = 0
                self.subspace_mechanism = None
                if self.subspace_is_private:
                    self.setup_private_subspace_mechanism(optimizer, t_total, multiplier)

        elif self.args.dpadam:
            privacy_engine = PrivacyEngine()
            model.train()  # Required for opacus

            if self.args.dp_clip_strategy == 'per_layer':
                # Layerwise clipping: Each layer has the same clipping threshold. The total grad norm is still bounded by `args.clip_C`.
                n_layers = len([(n, p) for n, p in model.named_parameters() if p.requires_grad])
                max_grad_norm_list = [self.args.dp_clip_threshold / np.sqrt(n_layers)] * n_layers
                model, optimizer, train_dataloader = privacy_engine.make_private(
                        module=model,
                        optimizer=optimizer,
                        data_loader=train_dataloader,
                        noise_multiplier=multiplier,
                        max_grad_norm=max_grad_norm_list,
                        clipping="per_layer",
                        poisson_sampling=False,
                        grad_sample_mode="hooks",
                    )
            elif self.args.dp_clip_strategy == 'flat':
                model, optimizer, train_dataloader = privacy_engine.make_private(
                        module=model,
                        optimizer=optimizer,
                        data_loader=train_dataloader,
                        noise_multiplier=multiplier,
                        max_grad_norm=self.args.dp_clip_threshold,
                        clipping="flat",
                        poisson_sampling=False,
                        grad_sample_mode="hooks",
                    )
        print("Optimizer:", type(optimizer), "\n")
        last_global_step = -1
        for epoch in range(epochs_trained, int(num_train_epochs)):
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)

            if transformers.is_torch_tpu_available():
                parallel_loader = pl.ParallelLoader(train_dataloader, [self.args.device]).per_device_loader(
                    self.args.device
                )
                epoch_iterator = tqdm(parallel_loader, desc="Iteration", disable=not self.is_local_process_zero())
            else:
                epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=True)

            # Reset the past mems state at the beginning of each epoch if necessary.
            if self.args.past_index >= 0:
                self._past = None

            for step, inputs in enumerate(epoch_iterator):

                torch.cuda.empty_cache()

                if self.args.dpgrape and self.state.global_step % self.args.subspace_T == 0 and self.state.global_step != last_global_step:
                    optimizer.update_projectors('gaussian')
                    optimizer.zero_grad()

                if self.low_rank_method is not None and self.state.global_step % self.args.subspace_T == 0 and self.state.global_step != last_global_step:
                    self.update_low_rank_subspaces(model, optimizer, inputs)

                if self.args.sync_embedding_layers:
                    assert model.module.model_type == 'opt', 'did not implement embedding layer synchronization for non-OPT models'
                    model.module.model.decoder.embed_tokens.weight = model.module.lm_head.weight
                
                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue
                    
                if self.args.zero_order_optim:
                    # Get parameters that should be optimized (for layer-wise optimization and prefix-tuning)
                    self.named_parameters_to_optim = []
                    for name, param in model.named_parameters():
                        if self.should_optim(name, param):
                            self.named_parameters_to_optim.append((name, param))

                    # get number of zs to sample
                    num_zs = self.get_num_samples()
                    if num_zs > 1:
                        assert self.args.zero_order_use_trainer_optim, 'cannot sample multiple zs without storing intermediate gradient. use trainer.'

                    for _ in range(num_zs):
                        # prepare for sampling new zs
                        random_vector = None
                        if self.args.efficient_zero_order:
                            random_seed = np.random.randint(1000000000)

                        with torch.no_grad():
                            # first function evaluation
                            if self.args.efficient_zero_order:
                                model = self.efficient_perturb_parameters(model, random_seed)
                            else:
                                model, random_vector = self.perturb_parameters(model)
                            loss1 = self.zo_forward(model, inputs)

                            # second function evaluation
                            if self.args.efficient_zero_order:
                                model = self.efficient_perturb_parameters(model, random_seed, scaling_factor=-2)
                            else:
                                model, random_vector = self.perturb_parameters(model, random_vector, scaling_factor=-2)
                            loss2 = self.zo_forward(model, inputs)

                        projected_grad = (loss1 - loss2) / (2 * self.args.zero_order_eps)

                        # DPZero gradient clipping and noise injection
                        if self.args.dpzero:
                            projected_grad = dpzero_clip(projected_grad, self.args.dpzero_clip_threshold).mean()
                            projected_grad += torch.randn(1).item() * self.dpzero_gaussian_std

                        # scale grad according to accumulation
                        if self.args.gradient_accumulation_steps > 1:
                            assert self.args.zero_order_use_trainer_optim, 'grad accumulation not implemented for non-trainer ZO yet'
                            projected_grad = projected_grad / self.args.gradient_accumulation_steps

                        # scale grad according to number of zs sampled
                        if not self.args.scale_lr_with_samples:
                            projected_grad = projected_grad / float(num_zs)

                        # store gradient in parameter buffer if using trainer
                        # o/w, the loop will exit after one round and the update will be applied directly (see below)
                        if self.args.zero_order_use_trainer_optim:
                            if self.args.efficient_zero_order:
                                torch.manual_seed(random_seed)

                            for name, param in self.named_parameters_to_optim:
                                # recover noise used in perturbations
                                if self.args.efficient_zero_order:
                                    z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
                                else:
                                    z = random_vector[name]

                                if param.grad is None:
                                    param.grad = projected_grad * z
                                else:
                                    param.grad += projected_grad * z

                        # reset model back to its parameters at start of step
                        if self.args.efficient_zero_order:
                            model = self.efficient_perturb_parameters(model, random_seed)
                        else:
                            model, random_vector = self.perturb_parameters(model, random_vector)

                    # apply gradient updates
                    # if using trainer, follow trainer logic to clip grad and check if parameters should be updated
                    if self.args.zero_order_use_trainer_optim:
                        if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
                            # last step in epoch but step is always smaller than gradient_accumulation_steps
                            len(epoch_iterator) <= self.args.gradient_accumulation_steps
                            and (step + 1) == len(epoch_iterator)
                        ):
                            # Gradient norm clipping
                            if self.args.zero_order_clip_grad:
                                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)

                            # Update the parameters and step scheduler
                            optimizer.step()
                            scheduler.step()
                        
                            # logging
                            if (self.args.logging_steps > 0 and self.state.global_step % self.args.logging_steps == 0) or (
                                self.state.global_step == 1 and self.args.logging_first_step
                            ):
                                logs = {}
                                logs["loss"] = loss1.item()
                                if not self.args.zero_order_clip_grad:
                                    norm = 0.0
                                    for _, p in model.named_parameters():
                                        if p.grad is not None:
                                            norm += torch.sum(p.grad ** 2)
                                    norm = torch.sqrt(norm)
                                logs["grad_norm"] = norm.item()
                                logs["learning_rate"] = (
                                    scheduler.get_last_lr()[0]
                                    if version.parse(torch.__version__) >= version.parse("1.4")
                                    else scheduler.get_lr()[0]
                                )
                                logs["num_zs"] = num_zs
                                logs["global_step"] = self.state.global_step
                                logs["zo_forward_step"] = self.state.zo_forward_step
                                logs["max_steps"] = self.args.max_steps
                                logs["max_zo_forward_steps"] = self.args.max_zo_forward_steps
                                logs["time"] = time.time() - start_time
                                self.log(logs)
                                logger.info(str(logs))
                                wandb_utils.log(logs, step=self.state.global_step, prefix="train")

                            model.zero_grad()
                            self.state.global_step += 1
                            self.epoch = epoch + (step + 1) / len(epoch_iterator)
                    # if not using the trainer, the updates are resampled and directly applied to the parameters
                    else:
                        # Efficient mode 
                        # WARNING: no gradient accumulation when not storing the grad
                        assert self.args.gradient_accumulation_steps == 1, 'gradient accumulation is not supported for zero-order optimization'
                        assert self.args.zero_order_sample_scheduler is None
                        assert not self.args.zero_order_clip_grad, 'gradient clipping not implemented yet for non-trainer ZO'

                        if self.args.efficient_zero_order:
                            torch.manual_seed(random_seed)     
                        for name, param in self.named_parameters_to_optim:
                            if self.args.efficient_zero_order:
                                z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
                            else:
                                z = random_vector[name]
                            param.data = param.data - self.args.learning_rate * projected_grad * z 

                        if (self.args.logging_steps > 0 and self.state.global_step % self.args.logging_steps == 0) or (
                                self.state.global_step == 1 and self.args.logging_first_step
                            ):
                                logs = {}
                                if self.args.dpzero:
                                    logs["loss"] = loss1.mean().item()
                                else:
                                    logs["loss"] = loss1.item()
                                logs["learning_rate"] = self.args.learning_rate
                                logs["global_step"] = self.state.global_step
                                logs["zo_forward_step"] = self.state.zo_forward_step
                                logs["max_steps"] = self.args.max_steps
                                logs["max_zo_forward_steps"] = self.args.max_zo_forward_steps
                                logs["time"] = time.time() - start_time
                                logs["max_memory_allocated"] = torch.cuda.max_memory_allocated()
                                logs["max_memory_reserved"] = torch.cuda.max_memory_reserved()
                                self.log(logs)
                                logger.info(str(logs))
                                wandb_utils.log(logs, step=self.state.global_step, prefix="train")


                        self.state.global_step += 1
                        self.epoch = epoch + (step + 1) / len(epoch_iterator)
                    

                # standard, non-ZO optimization
                else:
                    tr_loss += self.training_step(model, inputs)
                    last_global_step = self.state.global_step

                    if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
                        # last step in epoch but step is always smaller than gradient_accumulation_steps
                        len(epoch_iterator) <= self.args.gradient_accumulation_steps
                        and (step + 1) == len(epoch_iterator)
                    ):
                        if not self.uses_projected_dp and not self.args.dpadam: # dpgrape/dpadam do their own clipping
                            if self.args.fp16 and _use_native_amp:
                                self.scaler.unscale_(optimizer)
                                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)
                            elif self.args.fp16:
                                norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), self.args.max_grad_norm)
                            else:
                                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)

                        if self.args.optimizer_variant == 'signgd':
                            for n,p in model.named_parameters():
                                if p.grad is not None:
                                    p.grad = torch.sign(p.grad)

                        if transformers.is_torch_tpu_available():
                            xm.optimizer_step(optimizer)
                        elif self.args.fp16 and _use_native_amp:
                            self.scaler.step(optimizer)
                            self.scaler.update()
                        else:
                            optimizer.step()


                        scheduler.step()

                        if self.uses_projected_dp or self.args.dpadam:
                            optimizer.zero_grad()
                        else:
                            model.zero_grad()
                       
                        self.state.global_step += 1
                        self.epoch = epoch + (step + 1) / len(epoch_iterator)
       

                        if (self.args.logging_steps > 0 and self.state.global_step % self.args.logging_steps == 0) or (
                            self.state.global_step == 1 and self.args.logging_first_step
                        ):
                            logs = {}
                            tr_loss_scalar = tr_loss.item()
                            logs["global_step"] = self.state.global_step
                            logs["max_memory_allocated"] = torch.cuda.max_memory_allocated()
                            logs["max_memory_reserved"] = torch.cuda.max_memory_reserved()
                            logs["time"] = time.time() - start_time
                            logs["loss"] = (tr_loss_scalar - logging_loss_scalar) / self.args.logging_steps
                            #logs["norm"] = norm.item()
                            # backward compatibility for pytorch schedulers
                            logs["learning_rate"] = (
                                scheduler.get_last_lr()[0]
                                if version.parse(torch.__version__) >= version.parse("1.4")
                                else scheduler.get_lr()[0]
                            )
                            logging_loss_scalar = tr_loss_scalar

                            logs["rank"] = self.args.local_rank

                            self.log(logs)
                            logger.info(str(logs))
                            wandb_utils.log(logs, step=self.state.global_step, prefix="train")
                    
                    elif self.uses_projected_dp or self.args.dpadam:
                        optimizer.signal_skip_step()
                        optimizer.step()
                        optimizer.zero_grad()
                
                if self.args.max_steps > 0 and self.state.global_step > self.args.max_steps or (self.args.max_zo_forward_steps > 0 and self.state.zo_forward_step > self.args.max_zo_forward_steps):
                    epoch_iterator.close()
                    break

                # Log dev loss
                if self.state.global_step % self.args.eval_steps == 0 and self.state.global_step != 0:
                    eval_metrics = self.evaluate()
                    logs = {}
                    logs["global_step"] = self.state.global_step
                    logs["eval"] = eval_metrics
                    logger.info(str(logs))
                    wandb_utils.log(eval_metrics.metrics, step=self.state.global_step, prefix="eval")

                if self.args.evaluate_during_training and self.state.global_step % self.args.eval_steps == 0:
                    output = self.evaluate()
                    metrics = output.metrics
                    objective = self.dev_objective(metrics)
                    wandb_utils.log({"dev_objective": objective, "best_dev_objective": max(objective, self.objective)},
                                    step=self.state.global_step, prefix="eval")
                    if objective > self.objective:
                        logger.info("Best dev result: {}".format(objective))
                        self.objective = objective
                        # self.save_model(self.args.output_dir)

                        # Now we save this to (CPU) memory instead of disk <-- much faster
                        self.best_model_ckpt = {k: v.detach().cpu() for k, v in model.state_dict().items()}


            if self.args.max_steps > 0 and self.state.global_step > self.args.max_steps or (self.args.max_zo_forward_steps > 0 and self.state.zo_forward_step > self.args.max_zo_forward_steps):
                # train_iterator.close()
                break
            if self.args.tpu_metrics_debug or self.args.debug:
                # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                xm.master_print(met.metrics_report())

        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")

        train_summary = {
            "train/total_time": time.time() - start_time,
            "train/global_step": self.state.global_step,
            "train/zo_forward_step": self.state.zo_forward_step,
            "train/max_memory_allocated": torch.cuda.max_memory_allocated(),
            "train/max_memory_reserved": torch.cuda.max_memory_reserved(),
        }
        if math.isfinite(self.objective):  # -inf whenever --evaluate_during_training is off
            train_summary["eval/best_dev_objective"] = self.objective
        wandb_utils.set_summary(train_summary)

        return TrainOutput(self.state.global_step, tr_loss / self.state.global_step, metrics), self.objective


    def next_subspace_batch(self):
        """Draw a batch from a dedicated iterator, for the dev-split subspace modes."""
        if self.subspace_iter is None:
            self.subspace_iter = iter(self.subspace_dataloader)
        try:
            return next(self.subspace_iter)
        except StopIteration:
            self.subspace_iter = iter(self.subspace_dataloader)
            return next(self.subspace_iter)

    def setup_private_subspace_mechanism(self, optimizer, t_total, train_noise_multiplier):
        """Calibrate the Gaussian mechanism that private-skip releases its gradients through.

        The subspace mechanism is a *separate* mechanism from the training loop's: it runs
        once per subspace update, at sampling rate ``B / |dev|``, and it only ever touches
        the dev split. So it gets its own noise multiplier from the accountant, calibrated
        to --private_skip_epsilon.

        **Why it is legitimate to give the subspace its own full epsilon.** train.tsv and
        dev.tsv are disjoint (generate_k_shot_data.py), so every record lives in exactly one
        of them. A neighbouring dataset therefore perturbs either the training releases or
        the subspace releases, never both, and the two chains compose in *parallel*: the
        guarantee for a record is max(eps_train, eps_subspace), not their sum. The dev-side
        releases are also post-processing from the training side's point of view (they are
        functions of the public weights and of D_dev), and vice versa, so the interleaving is
        just adaptive composition within each chain.

        Both numbers are logged. eps_sequential is also logged as the conservative fallback
        for anyone who would rather not lean on the disjointness.
        """
        if "LOCAL_RANK" in os.environ:
            raise NotImplementedError(
                "--oracle_batch_mode private-skip is single-process only: per-sample "
                "clipping is done with a microbatch-of-one loop that is not synchronized "
                "across ranks."
            )

        projected_params = [p for p, _ in iter_low_rank_projectors(optimizer)]
        if not projected_params:
            raise RuntimeError("private-skip found no projected parameters to privatize")

        # The pool the subspace actually samples from, which --canary_probe shrinks by
        # fencing off its holdout. The sampling rate -- and so the noise -- must follow the
        # pool, not len(dev): a smaller pool means each record is drawn more often.
        num_dev = getattr(self, "subspace_pool_size", None) or len(self.eval_dataset)
        batch_size = self.args.train_batch_size
        if batch_size > num_dev:
            raise ValueError(
                f"subspace batch size {batch_size} exceeds the pool the subspace samples "
                f"from ({num_dev} examples). With --canary_probe the pool is the dev split "
                f"minus the holdout; lower --canary_holdout_frac or the batch size"
            )

        # Trigger is `global_step % subspace_T == 0`, so updates land on 0, T, 2T, ...
        # Count the endpoint too: the loop's break is `global_step > max_steps`, so a run can
        # reach global_step == t_total and take one more. Over-counting only ever adds noise.
        num_updates = (t_total // self.args.subspace_T) + 1
        sample_rate = batch_size / num_dev

        target_eps = self.args.private_skip_epsilon
        if target_eps <= 0:
            target_eps = self.args.dp_epsilon
        clip_threshold = self.args.private_skip_clip_threshold
        if clip_threshold <= 0:
            # C bounds a *projected* per-sample gradient, C_sub a full-dimensional one over
            # the projected matrices only. The projector shrinks norms by sqrt(rank) and some
            # gradient energy sits outside those matrices; at rank 16 and RoBERTa-Large's
            # ~85/15 split that nets out to ~2x. See DPTRACK_DESIGN.md section 4.5.
            clip_threshold = 2.0 * self.args.dp_clip_threshold
            logger.info(
                "private-skip: C_sub defaulted to 2 * --dp_clip_threshold = %.4g. Check "
                "subspace/clipped_fraction (want ~1.0) and mean_sample_norm_pre_clip; halve "
                "it if the clip is barely binding.",
                clip_threshold,
            )

        sigma = get_noise_multiplier(
            target_epsilon=target_eps,
            target_delta=self.args.dp_delta,
            steps=num_updates,
            sample_rate=sample_rate,
        )
        self.subspace_mechanism = PerLayerGaussianMechanism(
            params=projected_params,
            clip_threshold=clip_threshold,
            noise_multiplier=sigma,
            expected_batch_size=batch_size,
        )

        privacy = {
            "eps_train": self.args.dp_epsilon,
            "eps_subspace": target_eps,
            # Disjoint splits => parallel composition; this is the number to report.
            "eps_parallel": max(self.args.dp_epsilon, target_eps),
            # If you refuse the disjointness argument, this is the fallback.
            "eps_sequential": self.args.dp_epsilon + target_eps,
            "delta": self.args.dp_delta,
            "train_noise_multiplier": train_noise_multiplier,
            "subspace_noise_multiplier": sigma,
            "subspace_steps": num_updates,
            "subspace_sample_rate": sample_rate,
            "subspace_dev_examples": num_dev,
            "subspace_clip_threshold": clip_threshold,
            "subspace_per_layer_clip": self.subspace_mechanism.per_layer_clip,
            "subspace_num_layers": self.subspace_mechanism.num_layers,
            "subspace_released_noise_std": self.subspace_mechanism.released_noise_std,
        }
        logger.warning("*** private-skip privacy accounting: %s ***", privacy)
        wandb_utils.update_config({f"privacy/{k}": v for k, v in privacy.items()})

    def privatized_subspace_gradients(self, model, optimizer, batch):
        """Per-layer clipped, noised batch gradient for the projected weight matrices.

        One backward per sample, with opacus' hooks off, so ``p.grad`` holds exactly that
        sample's gradient and the clip bounds a real per-sample contribution. A larger
        microbatch would clip a microbatch *mean*, which bounds nothing.

        The cost is B small backwards instead of one big one -- at the defaults (B=64,
        subspace_T=100, 1000 steps) that is roughly a few percent of total training time,
        and it uses *less* memory than a DP step, which materializes per-sample gradients.
        """
        mech = self.subspace_mechanism
        mech.reset()

        batch = self._prepare_inputs(batch)
        # Every field the collator emits is batch-first (input_ids, attention_mask, labels,
        # mask_pos, ...), so slicing dim 0 gives a well-formed single-example batch.
        sizes = {v.shape[0] for v in batch.values() if torch.is_tensor(v) and v.dim() > 0}
        if len(sizes) != 1:
            raise RuntimeError(f"could not infer the subspace batch size, saw {sizes}")
        batch_size = sizes.pop()

        for i in range(batch_size):
            micro = {
                k: (v[i:i + 1] if torch.is_tensor(v) and v.dim() > 0 else v)
                for k, v in batch.items()
            }
            optimizer.zero_grad()
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, micro)
            if self.args.n_gpu > 1:
                loss = loss.mean()
            # compute_loss averages over the batch; with one sample that *is* the sample's
            # loss, so p.grad after backward is the per-sample gradient.
            loss.backward()
            mech.accumulate()

        optimizer.zero_grad()
        return mech.release()

    def update_low_rank_subspaces(self, model, optimizer, inputs):
        """Advance every layer's subspace from one batch gradient.

        oracle_batch_mode:
          shared        reuse this step's training batch, bare gradient      (eps = infinity)
          skip          held-out dev batch, bare gradient, no weight update  (eps = infinity)
          private-skip  held-out dev batch, per-layer clipped and noised     (finite eps)

        For shared/skip the hooks are disabled so opacus does not build per-sample
        gradients and ``p.grad`` is the plain batch mean Gbar_raw; nothing touches the
        accountant. For private-skip the gradient handed to the projector is the output of
        a Gaussian mechanism, so the SVD / geodesic step downstream is post-processing.
        """
        batch = self.next_subspace_batch() if self.subspace_uses_dev_split else inputs

        model.train()   # self.evaluate() may have left the model in eval mode
        optimizer.zero_grad()
        model.disable_hooks()
        grads = None
        canary_summary = None
        try:
            if self.subspace_is_private:
                grads = self.privatized_subspace_gradients(model, optimizer, batch)
            else:
                batch = self._prepare_inputs(batch)
                with self.compute_loss_context_manager():
                    loss = self.compute_loss(model, batch)
                if self.args.n_gpu > 1:
                    loss = loss.mean()
                loss.backward()
                # left as None on purpose: the bare modes read p.grad one layer at a time,
                # so they never hold a second copy of every projected gradient at once.

            with torch.no_grad():
                for p, projector in iter_low_rank_projectors(optimizer):
                    if grads is not None:
                        grad = grads.pop(p, None)
                    else:
                        grad = None if p.grad is None else p.grad.detach().float()
                    if grad is None:
                        continue
                    projector.update_subspace(grad, self.state.global_step)

            # Score the dev canaries against the subspace this batch just produced. Inside
            # the try so it still runs with hooks disabled (it needs a plain p.grad, not a
            # per-sample one) and outside the no_grad block because it backwards.
            #
            # last_batch is the draw that produced the gradient a few lines above -- with
            # num_workers=0 the loader fetches synchronously, so it has not run ahead.
            if self.canary_probe is not None:
                canary_summary = self.canary_probe.measure(
                    self, model, optimizer, self.state.global_step,
                    batch_indices=(
                        self.subspace_batch_sampler.last_batch
                        if self.subspace_batch_sampler is not None else None
                    ),
                )
        finally:
            grads = None
            optimizer.zero_grad()
            model.enable_hooks()

        self.subspace_batches_consumed += 1
        rotations = [
            proj.last_rotation_deg for _, proj in iter_low_rank_projectors(optimizer)
            if proj.last_rotation_deg is not None
        ]
        ortho_errs = [
            proj.last_ortho_err for _, proj in iter_low_rank_projectors(optimizer)
            if proj.last_ortho_err is not None
        ]
        diagnostics = {
            "subspace_update_at_step": self.state.global_step,
            "method": self.low_rank_method,
            "oracle_batch_mode": self.args.oracle_batch_mode,
            "subspace_batches_consumed": self.subspace_batches_consumed,
            "mean_rotation_deg": float(np.mean(rotations)) if rotations else None,
            "max_ortho_err": float(np.max(ortho_errs)) if ortho_errs else None,
        }
        if self.subspace_is_private:
            # mean_sample_norm_pre_clip vs per_layer_clip * sqrt(num_layers) is how you tell
            # whether C_sub is set anywhere near the right scale: clipped_fraction near 1.0
            # with a mean norm far above C means almost all signal is being thrown away.
            diagnostics.update(self.subspace_mechanism.diagnostics())
        if canary_summary is not None:
            diagnostics.update(canary_summary)
        logger.info(str(diagnostics))
        # mean_rotation_deg is the number to watch for dptrack (see roberta_finetuning_dptrack.sh):
        # near 0 means the tracker is a no-op, huge means it is thrashing.
        wandb_utils.log(diagnostics, step=self.state.global_step, prefix="subspace")

    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], num_items_in_batch=None
    ) -> torch.Tensor:
        
        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()

        inputs = self._prepare_inputs(inputs)
    
        with self.compute_loss_context_manager():
            #loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        del inputs
        #if (
        #    self.args.torch_empty_cache_steps is not None
        #    and self.state.global_step % self.args.torch_empty_cache_steps == 0
        #):
        #    torch.cuda.empty_cache()

        #kwargs = {}

        #self.accelerator.backward(loss, **kwargs)
        loss.backward()
        # Finally we need to normalize the loss for reporting
        if num_items_in_batch is None:
            return loss.detach() / self.args.gradient_accumulation_steps
        return loss.detach()

    """
    Difference compared to original implementation: return output instead of output.metrics (so there is also the logits)
    """
    def evaluate(self, eval_dataset: Optional[Dataset] = None) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are
        task-dependent (pass it to the init :obj:`compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (:obj:`Dataset`, `optional`):
                Pass a dataset if you wish to override :obj:`self.eval_dataset`. If it is an :obj:`datasets.Dataset`,
                columns not accepted by the ``model.forward()`` method are automatically removed. It must implement
                the :obj:`__len__` method.

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions.
        """
        if eval_dataset is not None and not isinstance(eval_dataset, collections.abc.Sized):
            raise ValueError("eval_dataset must implement __len__")

        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        #output = self.prediction_loop(eval_dataloader, description="Evaluation")
        output = self.evaluation_loop(eval_dataloader, description="Evaluation")

        self.log(output.metrics)
        logger.info(output.metrics)

        if self.args.tpu_metrics_debug or self.args.debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        return output