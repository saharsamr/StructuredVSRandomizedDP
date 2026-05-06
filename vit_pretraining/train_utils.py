import torch
from opacus.utils.batch_memory_manager import BatchMemoryManager
import numpy as np
from tqdm import tqdm

# Define train and test functions for training without projection
def train(model, train_dataloader, optimizer, criterion, epoch, rank, physical_batch_size,
            batch_step, dp=False, logical_bs=1000, scheduler=None, max_memory_exp=False, logger=None):

    model.train()

    train_loss = 0
    correct = 0
    total = 0

    current_batch_step = batch_step
    
    if dp:
        n_acc_steps = int(np.ceil(train_dataloader.batch_size / physical_batch_size))
        with BatchMemoryManager(
            data_loader=train_dataloader, 
            max_physical_batch_size=physical_batch_size, 
            optimizer=optimizer
        ) as memory_safe_data_loader:

            for step_idx, (inputs, targets) in tqdm(enumerate(memory_safe_data_loader), total=len(memory_safe_data_loader)):

                optimizer.zero_grad()

                inputs, targets = inputs.cuda(rank), targets.cuda(rank)

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                print(loss.item())

                optimizer.step()

                # End of logical batch
                if (step_idx + 1) % n_acc_steps == 0:
                    if logger is not None and max_memory_exp:
                        logger.info("Batch %d completed, max_memory_reserved: %d", current_batch_step, torch.cuda.max_memory_reserved(torch.cuda.device(rank)))
                    if max_memory_exp and current_batch_step == 5:
                        return
                    current_batch_step += 1
                    
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

    else:
        n_acc_steps = int(np.ceil(logical_bs / physical_batch_size))
        current_batch_step = 0
        for step_idx, (inputs, targets) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):

            inputs, targets = inputs.cuda(rank), targets.cuda(rank)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            # Update weights at end of logical batch, clear gradients
            if ((step_idx + 1) % n_acc_steps == 0) or (step_idx + 1 == len(train_dataloader)):
                if logger is not None and max_memory_exp:
                    logger.info("Batch %d completed, max_memory_reserved: %d", current_batch_step, torch.cuda.max_memory_reserved(torch.cuda.device(rank)))
                optimizer.step()
                optimizer.zero_grad()
                if scheduler is not None:
                    scheduler.step()
                if max_memory_exp and current_batch_step == 5:
                        return
                current_batch_step += 1

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()


    avg_loss = train_loss/(step_idx+1)
    acc = 100.*correct/total
    print('Epoch: ', epoch, len(train_dataloader), 'Train Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (avg_loss, acc, correct, total))
    
    # Return epoch loss and acc and current step for grad accumulation
    return current_batch_step, avg_loss, acc


def train_dpgrape(model, train_dataloader, optimizer, criterion, epoch, rank, physical_batch_size,
                    batch_step, subspace_T, rand_type='gaussian', max_memory_exp=False, logger=None):
    """
    Train for one epoch with DP-GRAPE.
    Args:
        model
        train_dataloader
        optimizer
        criterion
        epoch: Current epoch of training
        rank: local rank
        physical_batch_size (int): largest batch size that is loaded at once onto GPU
        batch_step (int): Total number of batches that have been completed during training
        subspace_T (int): Number of batches between updating projectors
        rand_type (str): Type of random projection to use, currently only option is 'gaussian'
        max_memory_exp (bool) : If True, quit after 5 batches for max memory experiment
    Returns:
        current_batch_step (int): Total number of completed batches during training
        avg_loss (float): Average loss over all samples for the epoch
        acc (float): Training accuracy for the epoch
    """
    
    model.train()

    train_loss = 0
    correct = 0
    total = 0

    current_batch_step = batch_step
    n_acc_steps = int(np.ceil(train_dataloader.batch_size / physical_batch_size)) 

    with BatchMemoryManager(
        data_loader=train_dataloader, 
        max_physical_batch_size=physical_batch_size, 
        optimizer=optimizer
    ) as memory_safe_data_loader:

        for step_idx, (inputs, targets) in tqdm(enumerate(memory_safe_data_loader), total=len(memory_safe_data_loader)):

            inputs, targets = inputs.cuda(rank), targets.cuda(rank)

            # Update projectors every T batches
            if current_batch_step % subspace_T == 0 and step_idx % n_acc_steps == 0:
                optimizer.update_projectors(rand_type)
                optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            #print(loss.item())
            loss.backward()       

            optimizer.step()

            # Keep track of loss and accuracy
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            optimizer.zero_grad()

            # End of logical batch
            if (step_idx + 1) % n_acc_steps == 0:
                #print("Batch", current_batch_step, "completed")
                if logger is not None and max_memory_exp:
                    logger.info("Batch %d completed, max_memory_reserved: %d", current_batch_step, torch.cuda.max_memory_reserved(torch.cuda.device(rank)))
                if max_memory_exp and current_batch_step == 5:
                    return
                current_batch_step += 1
            
    avg_loss = train_loss/(step_idx+1)
    acc = 100.*correct/total
    return current_batch_step, avg_loss, acc


def test(model, test_dataloader, criterion, rank):
    """
    Evaluate model on the test set.
    Args:
        model
        test_dataloader 
        criterion
        rank
    Returns:
        avg_loss (float): Average loss over all samples
        acc (float): Accuracy on test samples
    """
    
    model.eval()

    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_dataloader):
            
            inputs, targets = inputs.cuda(rank), targets.cuda(rank)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        acc = 100.*correct/total
        avg_loss = test_loss/(batch_idx+1)
        
    # Return epoch loss and acc
    return avg_loss, acc
  