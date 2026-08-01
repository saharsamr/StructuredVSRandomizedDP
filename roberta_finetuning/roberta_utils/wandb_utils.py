"""Weights & Biases logging for the RoBERTa fine-tuning runs.

Logging is on by default. Every function here is still a no-op whenever it cannot log --
``--use_wandb False``, wandb not installed, no credentials, a failed init -- so the trainer
and the run script can call them unconditionally and a machine without wandb set up runs
exactly as it did before this module existed. Only rank 0 logs.

This is deliberately separate from the HF Trainer's built-in wandb integration
(``--report_to wandb``): the training loop in roberta_utils/trainer.py is a fork of an old
transformers loop and never fires the callback events the integration relies on. The
scripts keep ``--report_to none`` and log through here instead.

Run ``wandb login`` once, then any of the roberta_finetuning_*.sh scripts logs by itself:

    WANDB_PROJECT=my-project TASK=SST-2 bash roberta_finetuning_dpgrape.sh
    USE_WANDB=false TASK=SST-2 bash roberta_finetuning_dpgrape.sh      # opt out
"""

import logging
import os

logger = logging.getLogger(__name__)

_run = None  # the active wandb Run, or None whenever logging is off
_wandb = None  # the imported module, so callers never have to import wandb themselves

# Lifted out of the namespaced config below into flat keys, because these are the columns
# you actually want to sort and group the runs table by.
_HIGHLIGHT_KEYS = (
    ("data", "task_name", "task"),
    ("data", "num_k", "num_k"),
    ("model", "model_name_or_path", "model"),
    ("model", "few_shot_type", "few_shot_type"),
    ("train", "seed", "seed"),
    ("train", "learning_rate", "learning_rate"),
    ("train", "max_steps", "max_steps"),
    ("train", "per_device_train_batch_size", "per_device_train_batch_size"),
    ("train", "gradient_accumulation_steps", "gradient_accumulation_steps"),
    ("train", "dp_epsilon", "dp_epsilon"),
    ("train", "dp_delta", "dp_delta"),
    ("train", "dp_clip_threshold", "dp_clip_threshold"),
    ("train", "dp_clip_strategy", "dp_clip_strategy"),
    ("train", "subspace_r", "subspace_r"),
    ("train", "subspace_T", "subspace_T"),
    ("train", "oracle_batch_mode", "oracle_batch_mode"),
    ("train", "st_step_size", "st_step_size"),
)


def is_active():
    """True when there is a live wandb run to log to."""
    return _run is not None


def method_name(training_args):
    """Short name of the optimization method, used for the run tags and config.

    Note that dpgalore and dptrack are eps = infinity as a whole (their subspace comes from
    bare batch gradients); the tag is there so those runs are easy to filter out of a
    comparison against the genuinely private methods.
    """
    for flag in ("dpgrape", "dpgalore", "dptrack", "dpzero", "dpadam"):
        if getattr(training_args, flag, False):
            return flag
    return getattr(training_args, "optimizer", "unknown")


def init(model_args, data_args, training_args):
    """Start a run. Returns True if logging is on, False if it stayed off for any reason."""
    global _run, _wandb

    if not getattr(training_args, "use_wandb", False):
        return False

    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank not in (-1, 0):
        return False  # every rank computes the same metrics; only rank 0 reports them

    try:
        import wandb
    except ImportError:
        logger.warning("wandb is not installed, so this run will not be logged to W&B; "
                       "`pip install wandb` to enable it, or pass USE_WANDB=false to silence this")
        return False

    if training_args.wandb_mode == "online" and not _has_credentials(wandb):
        # Logging is on by default, so an unconfigured machine has to degrade to a warning
        # here rather than stall the fine-tuning run on wandb's interactive login prompt.
        logger.warning("no W&B credentials found, so this run will not be logged to W&B; "
                       "run `wandb login` (or set WANDB_API_KEY) to enable it, WANDB_MODE=offline "
                       "to log locally, or USE_WANDB=false to silence this")
        return False

    method = method_name(training_args)
    tags = [method, str(data_args.task_name), str(model_args.model_name_or_path)]
    if training_args.wandb_tags:
        tags += [t for t in training_args.wandb_tags.split(",") if t]

    try:
        _run = wandb.init(
            project=training_args.wandb_project,
            entity=training_args.wandb_entity or None,
            name=_run_name(data_args, training_args, method),
            group=training_args.wandb_group or None,
            tags=tags,
            mode=training_args.wandb_mode,
            config=_config(model_args, data_args, training_args, method),
            dir=training_args.wandb_dir or None,
        )
    except Exception as e:  # a dead network or a bad API key must not kill the fine-tuning
        logger.warning("wandb.init failed (%s); continuing without wandb", e)
        _run = None
        return False

    _wandb = wandb
    logger.info("Logging to wandb run %s (%s)", _run.name, _run.url)
    return True


def _has_credentials(wandb):
    """True when wandb can authenticate without prompting for a key on stdin."""
    if os.environ.get("WANDB_API_KEY"):
        return True
    try:
        return bool(wandb.api.api_key)  # reads ~/.netrc, where `wandb login` writes the key
    except Exception:
        return False


def _run_name(data_args, training_args, method):
    if training_args.wandb_run_name:
        return training_args.wandb_run_name
    # The scripts point --log_file at a per-configuration .txt whose stem already encodes
    # the whole hyperparameter setting; reuse it so the run name matches the log on disk.
    stem = os.path.splitext(os.path.basename(training_args.log_file or ""))[0]
    if stem and stem != "log":
        return stem
    return f"{method}-{data_args.task_name}-seed{training_args.seed}"


def _config(model_args, data_args, training_args, method):
    """Namespaced dump of all three argument dataclasses, plus flat highlight columns."""
    config = {"method": method}
    namespaces = {"model": model_args, "data": data_args, "train": training_args}

    for namespace, args in namespaces.items():
        for key, value in vars(args).items():
            if key.startswith("_"):
                continue  # transformers keeps private bookkeeping (_n_gpu, ...) in here
            if isinstance(value, (int, float, bool, str)) or value is None:
                config[f"{namespace}/{key}"] = value
            else:
                config[f"{namespace}/{key}"] = str(value)

    for namespace, key, flat_key in _HIGHLIGHT_KEYS:
        value = getattr(namespaces[namespace], key, None)
        if value is not None:
            config[flat_key] = value

    return config


def update_config(values):
    """Add derived values (noise multiplier, batch size, ...) to the run config."""
    if _run is None:
        return
    _run.config.update(_clean(values), allow_val_change=True)


def log(metrics, step=None, prefix=None):
    """Log a metrics dict. Non-numeric entries are dropped; nested dicts are flattened.

    ``step`` should be the global step. wandb refuses to move its step counter backwards,
    so callers must pass a value that never decreases (self.state.global_step does not).
    """
    if _run is None:
        return
    metrics = _clean(metrics, prefix=prefix)
    if not metrics:
        return
    try:
        _run.log(metrics, step=step)
    except Exception as e:
        logger.warning("wandb log failed (%s); continuing", e)


def set_summary(values, prefix=None):
    """Pin final numbers to the run summary, where the runs table can show them."""
    if _run is None:
        return
    for key, value in _clean(values, prefix=prefix).items():
        _run.summary[key] = value


def finish():
    global _run
    if _run is None:
        return
    try:
        _run.finish()
    except Exception as e:
        logger.warning("wandb finish failed (%s); continuing", e)
    _run = None


def _clean(metrics, prefix=None, _out=None):
    """Flatten nested dicts to ``a/b`` keys and keep only what wandb can chart."""
    out = {} if _out is None else _out
    for key, value in metrics.items():
        key = str(key)
        if prefix:
            # HF's evaluation loop stamps every metric with "eval_", which would double up
            # under a namespace of ours: eval_acc reads better as final_dev/acc.
            for redundant in (prefix + "_", "eval_"):
                if key.startswith(redundant):
                    key = key[len(redundant):]
                    break
            key = f"{prefix}/{key}"
        if isinstance(value, dict):
            _clean(value, prefix=key, _out=out)
        elif isinstance(value, bool) or isinstance(value, (int, float)):
            out[key] = value
        elif hasattr(value, "item"):  # 0-d torch tensors / numpy scalars
            try:
                out[key] = value.item()
            except (ValueError, RuntimeError):
                pass
    return out
