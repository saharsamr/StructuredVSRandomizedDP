import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp

import torchvision
from torchvision.datasets import CIFAR10, CIFAR100, CelebA, MNIST

from opacus.validators import ModuleValidator
#from opacus import PrivacyEngine
from dpgrape.privacy.privacy_engine_modified import PrivacyEngineModified
from opacus.accountants.utils import get_noise_multiplier
from opacus.distributed import DifferentiallyPrivateDistributedDataParallel as DPDDP

import timm

import argparse

from train_utils import train, test, train_dpgrape

# AdamW optimizer from Galore 
from galore_torch import GaLoreAdamW
from dpgrape.dpadamw import DPAdamW as DPGrapeAdamW
from dpgrape.dpsgd import DPSGD as DPGrapeSGD

import time
import logging

def main():

    parser = argparse.ArgumentParser(description="DP Experiments")

    # Command line args
    parser.add_argument('--log_file', default='output.txt', type=str, help='Where to save results')
    parser.add_argument('--dp', action='store_true', help='Whether to add use DP')
    parser.add_argument('--clipping_strategy', default='standard', type=str, help='How to clip gradients')
    parser.add_argument('--clip_C', default=1.0, type=float, help='Per-sample clipping parameter')
    parser.add_argument('--noise_multiplier', default=2.0, type=float, help='Amount of noise added for DP')
    parser.add_argument('--epsilon', default=-1, type=float, help='Epsilon to achieve if not using fixed noise multiplier')
    parser.add_argument('--lr', default=5e-4, type=float, help='learning rate')
    parser.add_argument('--weight_decay', default=0.0, type=float, help='Weight decay for Adam')
    parser.add_argument('--epochs', default=60, type=int, help='number of epochs')
    parser.add_argument('--physical_bs', default=100, type=int, help='batch size that is loaded at one time')
    parser.add_argument('--logical_bs', default=1000, type=int, help='batch size achieved with gradient accumulation')
    parser.add_argument('--model_name', default='vit_b_16', type=str)
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='Options are CIFAR10, CIFAR100, MNIST, CelebA')
    parser.add_argument('--use_val', type=bool, default=False, help='If true, use existing validation set if available or split train set into train and val if not.')
    parser.add_argument('--seed', default=42, type=int, help='Value used to set seed')
    parser.add_argument('--subspace_r', default=64, type=int, help='Rank projections to use for galore')
    parser.add_argument('--subspace_T', default=250, type=int, help='Subspace switching freq for DP-GRAPE or GaLore')
    parser.add_argument('--project_sample_grads', default=True, action='store_true', help='Whether to project sample grads as they are computed')
    parser.add_argument('--correct_dp_bias', action='store_true', help='True to do DP-Adam bias correction')
    parser.add_argument('--rand_proj', default=None, type=str, help='Type of projector to use for DP-GRAPE (only option currently is gaussian)')
    parser.add_argument('--optimizer', default='Adam', type=str, help='Options are Adam or SGD (default Adam)')
    parser.add_argument('--naive_dp_galore', action='store_true', help='True to use naive DP-Galore')
    parser.add_argument('--finetuning', action='store_true', help='True to start from pretrained model')

    args = parser.parse_args()

    # Set up distributed training
    if "LOCAL_RANK" in os.environ:
        rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

        torch.cuda.device(rank)
    else:
        rank = 0
        world_size = 1

    # Setup logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    file_handler = logging.FileHandler(args.log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info(f"World size: {world_size}")

    np.random.seed(args.seed)
    
    # Load data, set up datasets and dataloaders
    if args.dataset.lower() == 'mnist':
        transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(224),
        torchvision.transforms.Grayscale(num_output_channels=3),   # Make 3 channels 
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5)),
    ])
    else:
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5)),
        ])

    logger.info("Loading data...")
    generator = torch.Generator().manual_seed(42)
    if args.dataset.lower() == 'cifar10':
        DATA_ROOT = 'data/cifar10'
        dataset_num_classes = 10
        if args.use_val:
            full_train_dataset = CIFAR10(root=DATA_ROOT, train=True, download=True, transform=transform)
            train_dataset, test_dataset = torch.utils.data.random_split(full_train_dataset, [0.8, 0.2], generator=generator)
        else:
            train_dataset = CIFAR10(root=DATA_ROOT, train=True, download=True, transform=transform)
            test_dataset = CIFAR10(root=DATA_ROOT, train=False, download=True, transform=transform)
    elif args.dataset.lower() == 'cifar100':
        DATA_ROOT = 'data/cifar100'
        dataset_num_classes = 100
        if args.use_val:
            full_train_dataset = CIFAR100(root=DATA_ROOT, train=True, download=True, transform=transform)
            train_dataset, test_dataset = torch.utils.data.random_split(full_train_dataset, [0.8, 0.2], generator=generator)
        else:
            train_dataset = CIFAR100(root=DATA_ROOT, train=True, download=True, transform=transform)
            test_dataset = CIFAR100(root=DATA_ROOT, train=False, download=True, transform=transform)
    elif args.dataset.lower() == 'mnist':
        DATA_ROOT = 'data/mnist'
        dataset_num_classes = 10
        if args.use_val:
            full_train_dataset = MNIST(root=DATA_ROOT, train=True, download=True, transform=transform)
            train_dataset, test_dataset = torch.utils.data.random_split(full_train_dataset, [0.8, 0.2], generator=generator)
        else:
            train_dataset = MNIST(root=DATA_ROOT, train=True, download=True, transform=transform)
            test_dataset = MNIST(root=DATA_ROOT, train=False, download=True, transform=transform)
    elif args.dataset.lower() == 'celeba':
        DATA_ROOT = 'data/celebA'
        dataset_num_classes = 10
        train_dataset = CelebA(root=DATA_ROOT, split='train', download=True, transform=transform)
        if args.use_val:
            test_dataset = CelebA(root=DATA_ROOT, split='valid', download=True, transform=transform)
        else:
            test_dataset = CelebA(root=DATA_ROOT, split='test', download=True, transform=transform)
    else:
        logger.info("Unsupported Dataset! Options are CIFAR10, CIFAR100, MNIST, and CelebA.")

    logger.info(f"Size of train dataset: {len(train_dataset)}")
    logger.info(f"Size of test dataset: {len(test_dataset)}")
    
    if args.dp:  # Different batch sizes for DP vs. non-DP because DP uses BatchMemoryManager from opacus
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.logical_bs // 1, shuffle=True, num_workers=4)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.logical_bs // 1, shuffle=False, num_workers=4)
    else:
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.physical_bs, shuffle=True, num_workers=4)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.physical_bs, shuffle=False, num_workers=4)
    
    # Load model
    logger.info("Loading model...")
    if args.model_name == 'vit_b_16':
        model = timm.create_model('vit_base_patch16_224', pretrained=args.finetuning, num_classes=dataset_num_classes)
        model.cls_token.requires_grad = False
        model.pos_embed.requires_grad = False

    # Replace problematic layers like batchnorm
    model = ModuleValidator.fix(model)
    model.cuda(rank)

    # Wrap model with DPDDP for distributed training
    if world_size > 1: 
        model = DPDDP(model)

    logger.info(f"Number of total parameters: {sum([p.numel() for p in model.parameters()])}")
    logger.info(f"Number of trainable parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}")
    
    # Set up loss function, optimizer
    criterion = nn.CrossEntropyLoss()
    # Setting up galore optimizer, as in
    # https://github.com/jiaweizzhao/GaLore/blob/master/torchrun_main.py
    if args.rand_proj is not None or args.naive_dp_galore:
        galore_params = []
        target_modules = ["attn", "attention", "dense", "mlp"]
        skip_modules = ["lm_head", "head"]
        for module_name, module in model.named_modules():
            if not isinstance(module, torch.nn.Linear) \
                or not any(target_key in module_name for target_key in target_modules) \
                or any(key in module_name for key in skip_modules):
                    continue 
            if module.weight.requires_grad:
                galore_params.append(module.weight)
        id_galore_params = [id(p) for p in galore_params]
        regular_params = [p for p in model.parameters() if id(p) not in id_galore_params and p.requires_grad]

        param_groups = [{'params': regular_params}, 
                        {'params': galore_params, 'rank': args.subspace_r, 'update_proj_gap': args.subspace_T, 'scale': 1.0, 'proj_type': 'std'}]
        
        if args.dp and args.optimizer == 'Adam' and args.rand_proj is not None:
            optimizer = DPGrapeAdamW(param_groups, lr=args.lr, dp_bias_correction=0)
        elif args.naive_dp_galore:
            optimizer = GaLoreAdamW(param_groups, lr=args.lr)
        elif args.dp:
            optimizer = DPGrapeSGD(param_groups, lr=args.lr, momentum=0, dampening=0)
        scheduler = None
    
    else:
        if args.optimizer == 'Adam':
            optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        else:
            optimizer = optim.SGD(model.parameters(), lr=args.lr)
        scheduler = None   # No lr scheduler for now


    # DP setup
    if args.dp:

        if args.project_sample_grads and args.rand_proj is not None:
            grad_sample_mode = "projected"   
        else:
            grad_sample_mode = "hooks"

        DELTA = 1 / len(train_dataset)
        # Use fixed sigma, calculate epsilon
        if args.noise_multiplier == 0:
            noise_multiplier = 0
        elif args.epsilon != -1:
            noise_multiplier =  get_noise_multiplier(target_epsilon = args.epsilon, target_delta = DELTA, 
                                                     sample_rate = args.logical_bs / len(train_dataset), 
                                                     epochs = args.epochs)
        else:
            noise_multiplier = args.noise_multiplier   # Constant noise, compute epsilon
        logger.info(f"Noise multiplier: {noise_multiplier}")
  
        privacy_engine = PrivacyEngineModified()
        if args.clipping_strategy == 'none':   # For experiments with noise but no clipping
            model, optimizer, train_dataloader = privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=train_dataloader,
                noise_multiplier=noise_multiplier,
                max_grad_norm=args.clip_C,
                clipping="none",
                poisson_sampling=False,
            )
        elif args.clipping_strategy == 'standard':  # Normal clipping+noising - changed to flat clipping
            model, optimizer, train_dataloader = privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=train_dataloader,
                noise_multiplier=noise_multiplier,
                max_grad_norm=args.clip_C,
                clipping="flat",
                poisson_sampling=False,
                random_proj=args.rand_proj is not None,
                grad_sample_mode=grad_sample_mode,
            )
        # Set up projector hooks for DP-Rand Projection
        if args.rand_proj is not None:
            optimizer.update_projectors(args.rand_proj)
            model.update_projectors(optimizer)
            model.remove_hooks(keep_ddp_hooks=True)
            model.add_hooks()
    logger.info(f"Optimizer: {type(optimizer)}")
    if args.dp:
        logger.info(f"(rank {rank}) Average batch size per GPU: {int(optimizer.expected_batch_size)}")

    # Keep track of train and test results
    batch_step = 0  # Number of batches that have been finished

    # Training loop
    for epoch in range(args.epochs):
        if args.dp and args.rand_proj is not None:  # DP-GRAPE
            if rank == 0:
                epoch_start_time = time.time()
            batch_step, epoch_train_loss, epoch_train_acc = train_dpgrape(model, train_dataloader, optimizer, criterion, epoch, rank,
                                            args.physical_bs, batch_step, subspace_T=args.subspace_T, rand_type=args.rand_proj, logger=logger)
        else:
            if rank == 0:
                epoch_start_time = time.time()
            batch_step, epoch_train_loss, epoch_train_acc = train(model, train_dataloader, optimizer, criterion, epoch, rank,
                                            args.physical_bs, batch_step, dp=args.dp, logical_bs=args.logical_bs, scheduler=scheduler, logger=logger)

        if rank == 0:
            epoch_end_time = time.time()
            epoch_train_time = epoch_end_time - epoch_start_time
            epoch_test_loss, epoch_test_acc = test(model, test_dataloader, criterion, rank)
            current_epsilon = None
            if args.dp and not args.noise_multiplier == 0.0 and args.clipping_strategy != 'none':
                current_epsilon = privacy_engine.get_epsilon(DELTA)
            logger.info(f"epoch: {epoch}")
            logger.info(f"epoch time (s): {epoch_train_time}")
            logger.info(f"train loss: {epoch_train_loss}")
            logger.info(f"train acc: {epoch_train_acc}")
            logger.info(f"test loss: {epoch_test_loss}")
            logger.info(f"test acc: {epoch_test_acc}")
            logger.info(f"epsilon: {current_epsilon}")

    if "LOCAL_RANK" in os.environ:
        dist.destroy_process_group()
    
if __name__ == "__main__":
    main()
