
import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
from torchvision.datasets import CIFAR10

from opacus.validators import ModuleValidator

from dpgrape.privacy.privacy_engine_modified import PrivacyEngineModified

import timm

import argparse

from train_utils import train, train_dpgrape

from galore_torch import GaLoreAdamW
from dpgrape.dpadamw import DPAdamW as DPGrapeAdamW

import logging
import time


def main():

    parser = argparse.ArgumentParser(description="DP Experiments")

    # Command line args
    parser.add_argument('--log_file', default='output_logs_vit_memory_exp/output.txt', type=str, help='Where to save results')
    parser.add_argument('--method', required=True, type=str, help='Options are DPAdam, DPGrape, NaiveDPGaLore, or Adam')
    parser.add_argument('--subspace_r', required=False, type=int, help='Rank of subspace for DPGrape or naive DP-GaLore')
    parser.add_argument('--batch_size', required=False, default=None, type=int, help='Manually set batch size if desired')
    
    args = parser.parse_args()

    # Setup logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    file_handler = logging.FileHandler(args.log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Setup model and data
    device= torch.device("cuda:0")

    # Load CIFAR10 data
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5)),
    ])
    DATA_ROOT = 'data/cifar10'
    dataset_num_classes = 10
    train_dataset = CIFAR10(root=DATA_ROOT, train=True, download=True, transform=transform)
    logical_bs = 1000

    if args.batch_size is None:
        if args.method.lower() == 'adam' or args.method.lower() == 'dpgrape':
            batch_size = 500
        elif args.method.lower() == 'dpadam' or args.method.lower() == 'naivedpgalore':
            batch_size = 200
    else:
        batch_size = args.batch_size
    
    if args.method.lower() == 'adam':
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    else:
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=logical_bs, shuffle=True, num_workers=1)
    criterion = nn.CrossEntropyLoss()

    lr = 1e-4
    noise_multiplier = 2
    clip_C = 1

    # Load model
    model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=dataset_num_classes)
    model.cls_token.requires_grad = False
    model.pos_embed.requires_grad = False
    model = ModuleValidator.fix(model)
    model.cuda(0)
    
    if args.method.lower() == 'dpadam':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0)
        privacy_engine = PrivacyEngineModified()
        model, optimizer, train_dataloader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_dataloader,
            noise_multiplier=noise_multiplier,
            max_grad_norm=clip_C,
            clipping="flat",
            poisson_sampling=False,
            random_proj=False,
            grad_sample_mode="hooks",
        )
        try:
            # Train for 1 epoch
            start_time = time.time()
            train(model, train_dataloader, optimizer, criterion, 0, 0,
                    batch_size, 0, dp=True, logical_bs=logical_bs, 
                    scheduler=None)
            torch.cuda.current_stream().synchronize()
            end_time = time.time()
            elapsed_time = end_time - start_time
            logger.info("Batch size: %d, maximum memory reserved: %d", batch_size, torch.cuda.max_memory_reserved(device))
            logger.info("Batch size: %d, epoch time (s): %d", batch_size, elapsed_time)
        except torch.OutOfMemoryError:  # Catch out of memory errors
            logger.info("Batch size: %d, out of memory", batch_size)
        except:
            print("Unexpected error!")  

    elif args.method.lower() == 'naivedpgalore':
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
                        {'params': galore_params, 'rank': args.subspace_r, 'update_proj_gap': 1000, 'scale': 1.0, 'proj_type': 'std'}]
        optimizer = GaLoreAdamW(param_groups, lr=lr)

        privacy_engine = PrivacyEngineModified()
        model, optimizer, train_dataloader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_dataloader,
            noise_multiplier=noise_multiplier,
            max_grad_norm=clip_C,
            clipping="flat",
            poisson_sampling=False,
            random_proj=False,
            grad_sample_mode="hooks",
        )
        try:
            # Train for 1 epoch
            start_time = time.time()
            train(model, train_dataloader, optimizer, criterion, 0, 0,
                    batch_size, 0, dp=True, logical_bs=logical_bs, 
                    scheduler=None)
            torch.cuda.current_stream().synchronize()
            end_time = time.time()
            elapsed_time = end_time - start_time
            logger.info("Batch size: %d, maximum memory reserved: %d", batch_size, torch.cuda.max_memory_reserved(device))
            logger.info("Batch size: %d, epoch time (s): %d", batch_size, elapsed_time)
        except torch.OutOfMemoryError:  # Catch out of memory errors
            logger.info("Batch size: %d, out of memory", batch_size)
        except:
            print("Unexpected error!")  

    elif args.method.lower()[:6] == 'adam':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0)
        try:
            # Train for 1 epoch
            start_time = time.time()
            train(model, train_dataloader, optimizer, criterion, 0, 0,
                    batch_size, 0, dp=False, logical_bs=logical_bs,
                    scheduler=None)
            torch.cuda.current_stream().synchronize()
            end_time = time.time()
            elapsed_time = end_time - start_time
            logger.info("Batch size: %d, maximum memory reserved: %d", batch_size, torch.cuda.max_memory_reserved(device))
            logger.info("Batch size: %d, epoch time (s): %d", batch_size, elapsed_time)
        except torch.OutOfMemoryError:  # Catch out of memory errors
            logger.info("Batch size: %d, out of memory", batch_size)
        except:
            print("Unexpected error!")  

    else:
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
                        {'params': galore_params, 'rank': args.subspace_r, 'update_proj_gap': 1000, 'scale': 1.0, 'proj_type': 'std'}]
        optimizer = DPGrapeAdamW(param_groups, lr=lr, dp_bias_correction=0)

        privacy_engine = PrivacyEngineModified()
        model, optimizer, train_dataloader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_dataloader,
            noise_multiplier=noise_multiplier,
            max_grad_norm=clip_C,
            clipping="flat",
            poisson_sampling=False,
            random_proj=True,
            grad_sample_mode="projected",
        )
        optimizer.update_projectors('gaussian')
        model.update_projectors(optimizer)
        model.remove_hooks(keep_ddp_hooks=True)
        model.add_hooks()

        #try:
        # Train for 1 epoch
        start_time = time.time()
        train_dpgrape(model, train_dataloader, optimizer, criterion, 0, 0,
                        batch_size, 0, subspace_T=1000, rand_type='gaussian')
        torch.cuda.current_stream().synchronize()
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info("Batch size: %d, maximum memory reserved: %d", batch_size, torch.cuda.max_memory_reserved(device))
        logger.info("Batch size: %d, epoch time (s): %d", batch_size, elapsed_time)
        #except torch.OutOfMemoryError:  # Catch out of memory errors
        #    logger.info("Batch size: %d, out of memory", batch_size)
        #except:
        #    print("Unexpected error!")  

if __name__ == "__main__":
    main()