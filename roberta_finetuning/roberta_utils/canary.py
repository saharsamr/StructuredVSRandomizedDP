# Dev-split canary probe: does the data-driven subspace respond to individual records?
#
# WHAT THIS MEASURES, AND WHY IT IS THE RIGHT PLACE TO LOOK
#
# Under --oracle_batch_mode skip / private-skip the dev split reaches the model through
# exactly one channel: it is backwarded to produce the gradient that the SVD / geodesic
# step turns into the subspace U. Dev examples never enter a weight update -- the trainer
# takes no optimizer step on the subspace batch. So any dev-side signal we can detect is
# subspace leakage and nothing else. That isolation is the entire point of probing dev
# rather than train.
#
# It also rules out the obvious readout. Since no weight update ever pushes the model
# toward a dev example's label, asking "does the final model predict the flipped label?"
# measures a channel that does not exist and will report ~zero for every method. U carries
# *directions*, not labels. So the probe measures directions:
#
#     capture(c) = || U^T g_c ||^2 / || g_c ||^2      summed over the projected matrices
#
# the fraction of canary c's gradient energy that survives projection onto the subspace.
#
# THE COMPARISON
#
# Half the dev examples get a flipped label, half keep the true one. A flipped label makes
# that example's gradient large and unusual. If the subspace is fit to bare dev gradients
# (skip), it chases those outliers and the flipped group's capture rises above the
# unflipped group's. The size of that gap is the leak.
#
#   skip          gap > 0 expected -- subspace is a noiseless function of dev data
#   private-skip  gap ~ 0 expected -- per-sample clipping bounds each record's pull on the
#                 SVD and the noise masks what is left
#   shared        the subspace is fit to the *train* batch, so the dev canaries provably
#                 never touch it. This is the placebo arm, and it does NOT come out at
#                 zero: a mislabeled example's gradient is unusual, and a subspace fit to
#                 any natural batch captures unusual directions *less* well, which shows up
#                 as a negative gap. That offset is geometry, not leakage, and it is
#                 present in the skip and private-skip numbers too.
#
# So the raw gap is not the answer. The leakage estimate is the baseline-adjusted gap,
# gap(arm) - gap(shared), which is what analyze_canaries.py reports and why the shared arm
# is not optional. It is a bias correction rather than a perfect counterfactual -- the
# shared run follows a different subspace trajectory -- but an unadjusted gap is the more
# wrong number, and in testing the offset was large enough to matter.
#
# ``chance`` in each record is the analytic capture of a *random* rank-r subspace,
# sum_l (r / d_l) ||g_l||^2 / sum_l ||g_l||^2. For a random projector (DP-GRAPE) this is
# exactly what capture concentrates on, for any gradient, which is what data-independence
# buys. It is the scale reference: capture well above chance means the subspace has
# aligned itself to the data.

import dataclasses
import json
import os

import torch
from transformers.utils import logging

from dpgrape.low_rank_projector_dp import iter_low_rank_projectors

logger = logging.get_logger(__name__)


class Canary:
    """One dev example under test.

    Attributes:
        query_idx: index into ``dataset.query_examples`` -- the canary's identity.
        feature_idx: position in ``dataset.features`` used for the gradient measurement.
            With num_sample > 1 one query example expands into several features (different
            demonstration draws / templates); all of them get the flipped label, and this
            is the first, which is the one we measure.
        flipped: True if this canary's label was changed.
        true_label / used_label: equal unless flipped.
    """

    __slots__ = ("query_idx", "feature_idx", "flipped", "true_label", "used_label")

    def __init__(self, query_idx, feature_idx, flipped, true_label, used_label):
        self.query_idx = query_idx
        self.feature_idx = feature_idx
        self.flipped = flipped
        self.true_label = true_label
        self.used_label = used_label

    def as_dict(self):
        return {
            "query_idx": self.query_idx,
            "feature_idx": self.feature_idx,
            "flipped": self.flipped,
            "true_label": self.true_label,
            "used_label": self.used_label,
        }


class CanaryProbe:
    """Flips half the dev labels, then measures per-canary subspace capture.

    Construct it *before* training starts (labels must already be flipped when the first
    subspace batch is drawn), then call ``measure`` right after every subspace update.

    Args:
        dataset: the dev-mode FewShotDataset, i.e. the trainer's ``eval_dataset``. Its
            ``features`` list is mutated in place.
        num_canaries: how many dev query examples to use. -1 (default) means all of them,
            which is usually what you want: the dev split in a k-shot run is small, and
            every example in it feeds the subspace anyway.
        seed: controls both which examples are chosen and which half gets flipped.
        log_path: JSONL output. Line 1 is the manifest, one line per (step, canary) after.
    """

    def __init__(self, dataset, num_canaries=-1, seed=0, log_path=None):
        self.dataset = dataset
        self.seed = seed
        self.log_path = log_path
        self._log_started = False

        if dataset.features is None:
            raise ValueError(
                "the canary probe needs a pre-featurized dataset (mode='dev'/'test'); "
                "this one builds features online, so labels cannot be flipped up front"
            )
        self.num_labels = len(dataset.label_list)
        if self.num_labels < 2:
            raise ValueError(
                "the canary probe flips labels, which needs a classification task with >= 2 "
                f"classes; this task has {self.num_labels} (regression)"
            )

        # feature positions grouped by query example, so every copy of a canary gets the
        # same flipped label.
        by_query = {}
        for feat_idx, (query_idx, _, _) in enumerate(dataset.example_idx):
            by_query.setdefault(query_idx, []).append(feat_idx)

        gen = torch.Generator().manual_seed(seed)
        query_ids = sorted(by_query)
        order = torch.randperm(len(query_ids), generator=gen).tolist()
        if num_canaries is not None and num_canaries >= 0:
            if num_canaries > len(query_ids):
                raise ValueError(
                    f"--num_canaries {num_canaries} exceeds the dev split "
                    f"({len(query_ids)} examples)"
                )
            order = order[:num_canaries]

        # Exact half-split rather than independent coin flips: the groups come out balanced,
        # which is worth more here than independence (we are testing a difference in means,
        # not building a Steinke-style eps lower bound, which is what would need the coins).
        num_flip = len(order) // 2
        self.canaries = []
        for rank, pos in enumerate(order):
            query_idx = query_ids[pos]
            feat_positions = by_query[query_idx]
            true_label = dataset.features[feat_positions[0]].label
            flipped = rank < num_flip
            if flipped:
                shift = 1 + int(torch.randint(self.num_labels - 1, (1,), generator=gen))
                used_label = (int(true_label) + shift) % self.num_labels
                for feat_idx in feat_positions:
                    # OurInputFeatures is @dataclass(frozen=True), so the label cannot be
                    # assigned; swap in a copy. dataset.features is the only thing holding
                    # these objects and __getitem__ reads it by index, so replacing the
                    # list entry is what every later read sees.
                    dataset.features[feat_idx] = dataclasses.replace(
                        dataset.features[feat_idx], label=used_label
                    )
            else:
                used_label = true_label
            self.canaries.append(
                Canary(query_idx, feat_positions[0], flipped, int(true_label), int(used_label))
            )

        self.num_flipped = sum(c.flipped for c in self.canaries)
        self.num_clean = len(self.canaries) - self.num_flipped
        logger.warning(
            "*** canary probe: %d dev examples under test, %d with flipped labels. "
            "The dev split now contains deliberately wrong labels, so eval metrics from "
            "this run are NOT meaningful -- do not use them for model selection or to "
            "report accuracy. The output of this run is %s. ***",
            len(self.canaries), self.num_flipped, log_path or "the in-log canary summary",
        )

    # ------------------------------------------------------------------ measurement

    @torch.no_grad()
    def _capture(self, optimizer):
        """Energy of the current p.grad captured by each layer's subspace.

        Returns (captured, total, chance) summed over the projected matrices.
        ``project`` is the projector's own method, so the left/right orientation is
        whatever the training path uses -- there is no second copy of that logic to get
        wrong here.
        """
        captured = total = chance = 0.0
        for p, projector in iter_low_rank_projectors(optimizer):
            if p.grad is None:
                continue
            g = p.grad.detach().float()
            energy = g.pow(2).sum().item()
            if energy == 0.0:
                continue
            low = projector.project(g, batch=False)
            captured += low.pow(2).sum().item()
            total += energy
            # d = the axis the subspace reduces; a random rank-r subspace of R^d keeps
            # r/d of any fixed vector's energy in expectation.
            d = g.shape[1] if projector.side == "right" else g.shape[0]
            chance += (projector.rank / d) * energy
        return captured, total, chance

    def measure(self, trainer, model, optimizer, global_step):
        """One backward per canary; log its capture against the subspace just built.

        Call from inside ``update_low_rank_subspaces``' hooks-disabled region, after the
        projectors have been updated -- the point is to score the canaries against the
        subspace that this step's batch produced.

        The model is put in eval mode for the duration so dropout does not randomize the
        gradient; the flipped-vs-clean difference we are chasing is small enough that
        dropout noise would need many more canaries to see through.
        """
        was_training = model.training
        model.eval()
        records = []
        try:
            for canary in self.canaries:
                feature = self.dataset[canary.feature_idx]
                batch = trainer.data_collator([feature])
                batch = trainer._prepare_inputs(batch)

                optimizer.zero_grad()
                with trainer.compute_loss_context_manager():
                    loss = trainer.compute_loss(model, batch)
                if trainer.args.n_gpu > 1:
                    loss = loss.mean()
                loss.backward()

                captured, total, chance = self._capture(optimizer)
                optimizer.zero_grad()

                if total == 0.0:
                    continue
                records.append({
                    "step": global_step,
                    "query_idx": canary.query_idx,
                    "flipped": canary.flipped,
                    "capture": captured / total,
                    "chance": chance / total,
                    "grad_sq_norm": total,
                    "loss": float(loss.detach().item()),
                })
        finally:
            optimizer.zero_grad()
            if was_training:
                model.train()

        self._write(records)
        return self._summarize(records, global_step)

    @staticmethod
    def _summarize(records, global_step):
        """Means per group and the gap, for the training log / wandb."""
        flip = [r["capture"] for r in records if r["flipped"]]
        clean = [r["capture"] for r in records if not r["flipped"]]
        chance = [r["chance"] for r in records]
        summary = {"canary_step": global_step, "canary_n": len(records)}
        if flip:
            summary["canary_capture_flipped"] = sum(flip) / len(flip)
        if clean:
            summary["canary_capture_clean"] = sum(clean) / len(clean)
        if flip and clean:
            summary["canary_capture_gap"] = (
                summary["canary_capture_flipped"] - summary["canary_capture_clean"]
            )
        if chance:
            summary["canary_capture_chance"] = sum(chance) / len(chance)
        return summary

    # ------------------------------------------------------------------------- log

    def manifest(self, extra=None):
        # 'canary_seed', not 'seed': the caller merges the training args into ``extra``,
        # and a plain 'seed' there would silently overwrite this one.
        m = {
            "record": "manifest",
            "canary_seed": self.seed,
            "num_canaries": len(self.canaries),
            "num_flipped": self.num_flipped,
            "num_clean": self.num_clean,
            "num_labels": self.num_labels,
            "canaries": [c.as_dict() for c in self.canaries],
        }
        if extra:
            m.update(extra)
        return m

    def start_log(self, extra=None):
        """Write the manifest line. Truncates any previous log at this path."""
        if self.log_path is None:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.log_path)), exist_ok=True)
        with open(self.log_path, "w") as f:
            f.write(json.dumps(self.manifest(extra)) + "\n")
        self._log_started = True
        logger.warning("*** canary probe logging to %s ***", self.log_path)

    def _write(self, records):
        if self.log_path is None or not records:
            return
        if not self._log_started:
            self.start_log()
        with open(self.log_path, "a") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
