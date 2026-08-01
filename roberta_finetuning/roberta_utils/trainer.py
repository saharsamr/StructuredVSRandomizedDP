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
from torch.utils.data.sampler import RandomSampler, SequentialSampler
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

    @property
    def low_rank_method(self):
        """'galore', 'subtrack', or None -- which data-driven subspace to track.

        Both are eps = infinity as a whole: the subspace comes from bare (unclipped,
        unnoised) batch gradients. Only the weight updates are DP. See DPTRACK_DESIGN.md.
        """
        if getattr(self.args, "dpgalore", False):
            return "galore"
        if getattr(self.args, "dptrack", False):
            return "subtrack"
        return None

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
            optimizer.load_state_dict(
                torch.load(os.path.join(model_path, "optimizer.pt"), map_location=self.args.device)
            )
            scheduler.load_state_dict(torch.load(os.path.join(model_path, "scheduler.pt")))

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
                logger.warning(
                    "*** %s: the subspace is estimated from BARE batch gradients. "
                    "The weight updates are DP but the subspace is not, so the method as "
                    "a whole is eps = infinity. Report it separately from DP-GRAPE. ***",
                    'DP-GaLore' if self.low_rank_method == 'galore' else 'DPTrack-Oracle',
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
                # Batch used for the bare backward at subspace update steps.
                if self.args.oracle_batch_mode == 'skip':
                    # Held-out dev split: disjoint from train.tsv and from test.tsv by construction.
                    assert self.eval_dataset is not None, '--oracle_batch_mode skip requires --do_eval'
                    self.subspace_dataloader = DataLoader(
                        self.eval_dataset,
                        batch_size=self.args.train_batch_size,
                        sampler=RandomSampler(self.eval_dataset),
                        collate_fn=self.data_collator,
                        drop_last=True,
                    )
                else:
                    self.subspace_dataloader = train_dataloader
                self.subspace_iter = None
                self.subspace_batches_consumed = 0

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

                if self.args.evaluate_during_training and self.state.global_step % self.args.eval_steps == 0:
                    output = self.evaluate()
                    metrics = output.metrics
                    objective = self.dev_objective(metrics)
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
        return TrainOutput(self.state.global_step, tr_loss / self.state.global_step, metrics), self.objective


    def next_subspace_batch(self):
        """Draw a batch from a dedicated iterator, for --oracle_batch_mode skip."""
        if self.subspace_iter is None:
            self.subspace_iter = iter(self.subspace_dataloader)
        try:
            return next(self.subspace_iter)
        except StopIteration:
            self.subspace_iter = iter(self.subspace_dataloader)
            return next(self.subspace_iter)

    def update_low_rank_subspaces(self, model, optimizer, inputs):
        """One ordinary backward pass, then advance every layer's subspace with it.

        This is the non-private half of DP-GaLore / DPTrack-Oracle. Hooks are disabled so
        opacus does not build per-sample gradients: ``p.grad`` is then the plain batch mean
        Gbar_raw. Nothing here touches the accountant -- no clipping, no noise, no step.

        oracle_batch_mode:
          shared  reuse this step's training batch (one extra backward, no data wasted)
          skip    draw a batch from the held-out dev split, no weight update
        """
        skip_mode = self.args.oracle_batch_mode == 'skip'
        batch = self.next_subspace_batch() if skip_mode else inputs

        model.train()   # self.evaluate() may have left the model in eval mode
        optimizer.zero_grad()
        model.disable_hooks()
        try:
            batch = self._prepare_inputs(batch)
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, batch)
            if self.args.n_gpu > 1:
                loss = loss.mean()
            loss.backward()

            with torch.no_grad():
                for p, projector in iter_low_rank_projectors(optimizer):
                    if p.grad is None:
                        continue
                    projector.update_subspace(p.grad.detach().float(), self.state.global_step)
        finally:
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
        logger.info(str({
            "subspace_update_at_step": self.state.global_step,
            "method": self.low_rank_method,
            "oracle_batch_mode": self.args.oracle_batch_mode,
            "subspace_batches_consumed": self.subspace_batches_consumed,
            "mean_rotation_deg": float(np.mean(rotations)) if rotations else None,
            "max_ortho_err": float(np.max(ortho_errs)) if ortho_errs else None,
        }))

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