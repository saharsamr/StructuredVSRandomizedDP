# Memory-Efficient Differentially Private Training with Gradient Random Projection
This repo contains code to train models with DP-GRAPE as well as other methods for comparison including DP-Adam and DPZero. DP-GRAPE uses random projection of layer gradients to significantly reduce memory usage, while maintaining similar utility to DP-Adam. In particular, DP-GRAPE reduces memory usage by over 63% when pre-training Vision Transformers and over 70% when fine-tuning RoBERTa-Large as compared to DP-Adam, while achieving similar performance. The codebase contains scripts to reproduce our experiments using DP-GRAPE to train Vision Transformers on CIFAR10/100 and MNIST, and fine-tune RoBERTa-Large and OPT 1.3B, 2.7B, and 6.7B models on various NLP datasets.

# Experiments
Note that we used python 3.10 to build the environments for our experiments.

## Installing dpgrape package
Before running any of the experiments, use pip install from the dpgrape folder to install the functions for DP-GRAPE as a package.
```bash
cd dpgrape
pip install .
```

## Vision Transformer Pre-training
First cd to the vit_pretraining directory, and then install the necessary pacakges:
```bash
cd vit_pretraining
pip install -r requirements.txt
```
For all experiments except the memory experiment, the total batch size is set to 1000 and is achieved through gradient accumulation. The maximum number of samples loaded at one time is controlled by PHYSICAL_BS within the scripts. If you are running out of memory with the current settings, decrease PHYSICAL_BS.
### C/LR Search Scripts
The dpadam, dpgrape, and naive_dpgalore folders in the experiments folder contain scripts to run the grid search over DP Clip C and learning rate for MNIST, CIFAR-10, and CIFAR-100 (the hyperparam_search scripts). For example, to run the grid search for CIFAR-10 for DP-GRAPE:
```bash
bash experiments/dpgrape/cifar10_hyperparam_search.sh
```
The results for each hyperparameter configuration will be stored within the output_logs_vit_c_lr_search folder.
### Final Results Scripts
Also within the dpadam, dpgrape, and naive_dpgalore folders are scripts to obtain the final results for each method and datasets at different privacy levels. For example, to get the final results for CIFAR-10 for DP-GRAPE:
```bash
bash experiments/dpgrape/cifar10_final.sh
```
The results for each training run will be stored in the output_logs_vit_final folder.
### Other Experiments
There are scripts to run the memory experiment and timing experiment within the experiments folder, called memory_exp.sh and timing_exp.sh, respectively. The results for these experiments will be saved in output_logs_vit_memory_exp and output_logs_vit_timing_exp.

## RoBERTa-Large Fine-tuning
(The code for RoBERTa-Large fine-tuning is based on the DPZero repo: https://github.com/Liang137/DPZero)
First cd to the roberta_finetuning directory, and then install the necessary pacakges:
```bash
cd roberta_finetuning
pip install -r requirements.txt
```
Next, navigate to the data directory, download the data, and prepare the datasets for finetuning by running the prepare_datasets script from the roberta_finetuning/data directory:
```bash
cd data
bash prepare_datasets.sh
```
Before running any of the scripts, set the visible GPUs with:
```bash
export CUDA_VISIBLE_DEVICES=...
```
For all of the hyperparameter search, C search, and final few-shot results scripts, the physical batch size is set as PER_DEVICE_TRAIN_BS, and the total batch size is PER_DEVICE_TRAIN_BS * GRAD_ACCUM_STEPS. If you are running out of memory, decrease PER_DEVICE_TRAIN_BS.
### Hyperparameter Search Script
The experiments/dpgrape_hyperparam_search.sh script can be used to run a full grid search over learning rates, DP Clip Cs, projection dimensions r, projection switching frequency F, and total number of fine-tuning steps T. Pass the dataset to run the search over as a command line arg. For example, to run the search for SST-2:
```bash
bash experiments/dpgrape_hyperparam_search.sh SST-2
```
The results for each individual fine-tuning run will be saved in a .txt file in a folder named output_logs_roberta_convergence_exp.
### DP Clip C Tuning Scripts
The C search scripts in experiments/dpgrape for all of the different datasets expect CUDA_VISIBLE_DEVICES to include either 1, 2, 4, 8, or 16 GPUs. The per-device size and number of gradient accumulation steps are set based on the number of GPUs so that the total batch size for each step is 64. The scripts for each dataset can be run indivdually, or all at once with the C_search_all_datasets.sh script (run from the roberta_finetuning directory):
```bash
bash experiments/dpgrape/C_search_all_datasets.sh
```
The results for each individual fine-tuning run will be saved in a .txt file in a folder named output_logs_roberta_C_search.
### Final Fine-Tuning Results
The final fine-tuning results over 3 random seeds and 2 privacy levels for each dataset can be obtained by running the few_shot_(dataset).sh scripts in experiments/dpgrape. For example, to get the final results for SST-2:
```bash
bash experiments/dpgrape/few_shot_sst2.sh
```
The results for each individual fine-tuning run will be saved in a .txt file in a folder named output_logs_roberta_few_shot.
### Other Experiments
There are scripts to run the memory experiment, timing experiment, and convergence comparison between DP-GRAPE and DPZero, called memory_exp.sh, timing_exp.sh, and convergence_exp.sh, respectively, which are all in the experiments folder. The results for each of these experiments will be stored in output_logs_roberta_memory_exp, output_logs_roberta_timing_exp, and output_logs_roberta_convergence_exp.
### Logging to Weights & Biases
All of the RoBERTa fine-tuning scripts also stream their logs and results to [Weights & Biases](https://wandb.ai). This is on by default; log in once with `wandb login` and the runs show up on their own:
```bash
TASK=SST-2 SEED=42 bash roberta_finetuning_dpgrape.sh
```
To opt out, either for one run or for a whole shell:
```bash
USE_WANDB=false TASK=SST-2 SEED=42 bash roberta_finetuning_dpgrape.sh
export USE_WANDB=false
```
Every RoBERTa script declares `USE_WANDB=${USE_WANDB:-true}` alongside its other settings and passes `--use_wandb $USE_WANDB` on to roberta_finetuning_run.py, so this works the same way for every method (dpgrape, dpgalore, dptrack, dpadam, dpzero, adam) and for each fine-tuning run launched by the scripts in the experiments folder. The .txt log files are still written exactly as before; W&B logging is purely additive.

Each run logs the full argument set as its config (plus the noise multiplier the accountant calibrated, the total batch size, and the trainable parameter count), the training curves at every `logging_steps` (loss, learning rate, peak memory, wall clock), the dev metrics at every `eval_steps`, the subspace diagnostics for DP-GaLore/DPTrack (`subspace/mean_rotation_deg`, `subspace/max_ortho_err`), and the final dev and test metrics as both a logged point and a run summary value. The run is tagged with its method, task, and model, so DP-GRAPE runs can be filtered apart from the eps = infinity methods.

The other W&B settings are environment variables:

| Variable | Default | Meaning |
| --- | --- | --- |
| `USE_WANDB` | `true` | set to `false` to turn logging off |
| `WANDB_PROJECT` | `structured-vs-randomized-dp` | project to log to |
| `WANDB_ENTITY` | your default entity | team or user |
| `WANDB_RUN_NAME` | the grid-search tag | run name, which by default matches the .txt log file name |
| `WANDB_GROUP` | none | groups runs together, e.g. the seeds of one experiment |
| `WANDB_TAGS` | none | extra tags, comma separated |
| `WANDB_MODE` | `online` | use `offline` on a cluster without internet access, then upload later with `wandb sync` |

For example, to group a seed sweep and log offline from a compute node:
```bash
WANDB_GROUP=sst2-eps6 WANDB_MODE=offline TASK=SST-2 SEED=42 bash roberta_finetuning_dpgrape.sh
```
Under multi-GPU torchrun only rank 0 logs. Because logging is on by default, a machine that is not set up for W&B must never be held up by it: if wandb is not installed, or no API key is configured, or `wandb.init` fails for any other reason (no network, bad entity), the fine-tuning run prints a warning saying what to do and then continues, unaffected and never prompting for input.

## OPT Fine-Tuning
(The code for OPT fine-tuning is based on the DPZero repo: https://github.com/Liang137/DPZero)
To set-up the environment, navigate to the opt_finetuning directory and then install the necessary pacakges:
```bash
cd opt_finetuning
pip install -r requirements.txt
```
(The requirements for the OPT fine-tuning are the same as the RoBERTa fine-tuning).
For all experiments, the scripts expect CUDA_VISIBLE_DEVICES to include either 1, 2, 4, or 8 GPUs. The per-device size and number of gradient accumulation steps are set based on the number of GPUs so that the total batch size for each step is 8. For the OPT experiments, there is no need to download the data before running the scripts.
### DP Clip C Search Scripts
The C search scripts for the different datasets and model sizes for DP-GRAPE and DP-Adam are in experiments/dpadam and experiments/dpgrape. For example, to run the C search for the 1.3B model with DP-GRAPE on SQuAD:
```bash
bash experiments/dpgrape/C_search_squad_1-3b.sh
```
The results for each fine-tuning run will be saved in output_logs_opt_C_search.
### Final Results
The final fine-tuning results over 3 random seeds and 2 privacy levels for each dataset and model size can be obtained by running the final_results_(dataset)_(model_size).sh scripts in experiments/dpadam and experiments/dpgrape. For example, to run the final results for DP-GRAPE on SQuAD with the 1.3B model:
```bash
bash experiments/dpgrape/final_results_squad_1-3b.sh
```
The results will be saved in output_logs_opt_final.
### Other Experiments
The scripts for the memory and timing experiments for the different model sizes are located in the experiments directory. The results will be saved in output_logs_opt_memory_exp and output_logs_opt_timing_exp, respectively.

# Citation
```bibtex
@article{mulrooney2025memory,
  title={Memory-Efficient Differentially Private Training with Gradient Random Projection},
  author={Mulrooney, Alex and Gupta, Devansh and Flemings, James and Zhang, Huanyu and Annavaram, Murali and Razaviyayn, Meisam and Zhang, Xinwei},
  journal={arXiv preprint arXiv:2506.15588},
  year={2025}
}
```
