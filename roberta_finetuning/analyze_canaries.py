#!/usr/bin/env python
"""Read canary_probe.jsonl logs and print every number the experiment produces.

    python analyze_canaries.py canary_logs/<run>/skip.jsonl canary_logs/<run>/private-skip.jsonl

What is being measured
----------------------
Every dev example gets a capture score at each subspace update:

    capture = || U^T g ||^2 / || g ||^2      summed over the projected matrices

the fraction of that example's gradient energy the subspace keeps. Chance, for a random
rank-r subspace, is r/d.

THE MAIN TEST: membership
-------------------------
Each subspace update draws one batch (64 of the dev pool). Only those 64 fed the subspace.
A fraction of the dev split is also fenced off from every batch, as a permanent control.
So at every update the canaries fall into three groups:

    fed      drawn at this update -- influenced the subspace that is being scored
    prior    drawn at an earlier update -- influenced it, but several rotations ago
    holdout  fenced off, never drawn at any update -- provably never touched the subspace

fed vs holdout is the test. Both groups are the same records with the same flipped labels,
split only by the seed, so they are identical in distribution and nothing has to be
subtracted off. The question it asks is the privacy question directly: was this record in
the data that produced this subspace?

    fed > holdout, significant  the subspace carries information about which individual
                                records built it. Under --oracle_batch_mode skip that is
                                the leak, measured.
    fed ~ holdout               no detectable per-record response. Expected for
                                private-skip: clipping bounds each record's pull on the
                                subspace and the noise covers what is left.

fed > prior > holdout, decaying in that order, is the corroborating pattern for a tracker
that accumulates (subtrack): influence fading as the subspace rotates away is much harder
to explain as noise than a single two-group difference. GaLore refits from scratch every
update, so it should show fed >> prior ~ holdout with no gradient in between -- a flat
prior there is expected, not a problem.

THE SECONDARY TEST: flipped vs clean
------------------------------------
Also reported, restricted to the fed group. This one is NOT self-controlled: a mislabeled
example's gradient points somewhere unusual, and a subspace fitted to any natural batch
captures unusual directions less well, which shows up as a negative offset even when the
subspace never saw the record. A 'shared' run measures that offset (its subspace is fitted
to the train batch, so it cannot have seen any dev canary) and it is subtracted where
available. Prefer the membership test, which needs no such correction.

capture/chance says whether the subspace is aligned to gradients at all. 1.0 means it is no
better than a random rank-r subspace, and a difference measured under a subspace that is
not tracking is not evidence about a subspace that is.
"""

import argparse
import json
import math
import os
from collections import defaultdict

try:
    from scipy import stats as _scipy_stats
except ImportError:
    _scipy_stats = None


# --------------------------------------------------------------------------- statistics

def _sf(t, df):
    """Two-sided p-value for a t statistic."""
    if _scipy_stats is not None:
        return float(2.0 * _scipy_stats.t.sf(abs(t), df))
    # Normal approximation; fine at the df we get here (K >= 10 subspace updates), and
    # slightly anti-conservative below that, which the printed df lets you judge.
    return float(math.erfc(abs(t) / math.sqrt(2.0)))


def _mean_var(xs):
    n = len(xs)
    if n == 0:
        return 0.0, 0.0, 0
    mean = sum(xs) / n
    if n == 1:
        return mean, 0.0, 1
    return mean, sum((x - mean) ** 2 for x in xs) / (n - 1), n


def mean(xs):
    return sum(xs) / len(xs) if xs else float('nan')


def welch(a, b):
    """Two-sample Welch t-test. Returns (diff, t, df, p) or None if underpowered."""
    ma, va, na = _mean_var(a)
    mb, vb, nb = _mean_var(b)
    if na < 2 or nb < 2:
        return None
    se2 = va / na + vb / nb
    if se2 <= 0.0:
        return (ma - mb, float('inf') if ma != mb else 0.0, na + nb - 2,
                0.0 if ma != mb else 1.0)
    t = (ma - mb) / math.sqrt(se2)
    df = se2 ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    return ma - mb, t, df, _sf(t, df)


def one_sample_t(xs):
    """One-sample t-test against 0. Returns (mean, se, t, df, p) or None."""
    m, var, n = _mean_var(xs)
    if n < 2:
        return None
    se = math.sqrt(var / n)
    if se == 0.0:
        return (m, 0.0, float('inf') if m != 0 else 0.0, n - 1, 0.0 if m != 0 else 1.0)
    return m, se, m / se, n - 1, _sf(m / se, n - 1)


# -------------------------------------------------------------------------------- input

def load(path):
    """Return (manifest, {step: [record, ...]}), each record tagged fed/prior/holdout.

    'prior' needs the history, so the tagging is done here, in one pass over the sorted
    steps, rather than in the probe: the probe only records whether each canary was in the
    batch it just saw.
    """
    manifest = None
    by_step = defaultdict(list)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("record") == "manifest":
                manifest = rec
            else:
                by_step[rec["step"]].append(rec)
    if manifest is None:
        raise ValueError(f"{path}: no manifest line -- is this a canary_probe.jsonl?")
    if not by_step:
        raise ValueError(
            f"{path}: manifest but no measurements. The run never reached a subspace "
            f"update, or --canary_probe was on for a method without a data-driven subspace"
        )

    seen_before = set()
    for step in sorted(by_step):
        drawn_here = set()
        for rec in by_step[step]:
            if "in_batch" not in rec:
                rec["group"] = None          # shared mode: no dev draw to attribute
                continue
            if rec.get("held_out"):
                # Fenced off from every batch, so it can never be 'fed' or 'prior'.
                rec["group"] = "holdout"
            elif rec["in_batch"]:
                rec["group"] = "fed"
                drawn_here.add(rec["query_idx"])
            elif rec["query_idx"] in seen_before:
                rec["group"] = "prior"
            else:
                rec["group"] = "notyet"
        seen_before |= drawn_here

    if not any(r.get("group") == "holdout" for r in by_step[sorted(by_step)[0]]):
        # Older logs, or --canary_holdout_frac 0. 'not drawn yet' is the only control
        # available; it empties out once the sampler has been through an epoch.
        for step in by_step:
            for rec in by_step[step]:
                if rec.get("group") == "notyet":
                    rec["group"] = "holdout"
    return manifest, by_step


def label_for(manifest, path):
    mode = manifest.get("oracle_batch_mode", "?")
    method = manifest.get("method", "?")
    if mode == "?" and method == "?":
        return os.path.basename(os.path.dirname(os.path.abspath(path))) or path
    return f"{method}/{mode}"


def caps(recs, group=None, flipped=None):
    out = []
    for r in recs:
        if group is not None and r.get("group") != group:
            continue
        if flipped is not None and r["flipped"] != flipped:
            continue
        out.append(r["capture"])
    return out


# ------------------------------------------------------------------------------- report

def report_run(path, manifest, by_step, verbose=True):
    name = label_for(manifest, path)
    print("=" * 100)
    print(f"RUN  {name}")
    print(f"file {path}")
    cfg = "  ".join(
        f"{k}={manifest[k]}" for k in
        ("task_name", "subspace_r", "subspace_T", "max_steps", "dp_epsilon",
         "dp_clip_threshold", "seed", "canary_seed")
        if k in manifest
    )
    if cfg:
        print(f"cfg  {cfg}")
    print(f"     {manifest['num_canaries']} canaries: "
          f"{manifest['num_flipped']} flipped / {manifest['num_clean']} clean, "
          f"{manifest['num_labels']} classes")

    steps = sorted(by_step)
    tagged = any(r.get("group") for r in by_step[steps[0]])
    print()

    row = {"name": name, "path": path, "mode": manifest.get("oracle_batch_mode", "?"),
           "method": manifest.get("method", "?"), "tagged": tagged}

    # ---------------------------------------------------------------- membership test
    if tagged:
        if verbose:
            print("  MEMBERSHIP  (flipped canaries only; fed = drawn into this update's "
                  "subspace batch)")
            print()
            print(f"  {'step':>7} {'n_fed':>6} {'n_hold':>8} {'fed':>10} {'prior':>10} "
                  f"{'holdout':>10} {'fed-hold':>12} {'t':>8} {'p':>9}")
            print("  " + "-" * 96)

        diffs, decay_ok, decay_n = [], 0, 0
        for step in steps:
            recs = by_step[step]
            fed = caps(recs, "fed", flipped=True)
            prior = caps(recs, "prior", flipped=True)
            held = caps(recs, "holdout", flipped=True)
            w = welch(fed, held)
            if w is None:
                continue
            diff, t, df, p = w
            diffs.append(diff)
            if prior:
                decay_n += 1
                if mean(fed) >= mean(prior) >= mean(held):
                    decay_ok += 1
            if verbose:
                print(f"  {step:>7} {len(fed):>6} {len(held):>8} {mean(fed):>10.6f} "
                      f"{mean(prior):>10.6f} {mean(held):>10.6f} {diff:>+12.6f} "
                      f"{t:>8.3f} {p:>9.4f}")
        print()
        res = one_sample_t(diffs)
        if res is None:
            print(f"  not enough updates with both groups to test ({len(diffs)})")
            row["mem_gap"] = diffs[0] if diffs else float('nan')
            row["mem_p"] = float('nan')
        else:
            m, se, t, df, p = res
            row["mem_gap"], row["mem_se"], row["mem_p"] = m, se, p
            print(f"  HEADLINE   fed - holdout over {len(diffs)} updates = "
                  f"{m:+.6f} +/- {se:.6f} (se)")
            print(f"             one-sample t = {t:.3f}, df = {df}, p = {p:.4g}")
            if decay_n:
                print(f"             fed >= prior >= holdout at {decay_ok}/{decay_n} "
                      f"updates that had a prior group")
    else:
        print("  MEMBERSHIP  not available: this run has no dev-split subspace batch")
        print("              (oracle_batch_mode=shared fits the subspace to the train "
              "batch). Used below")
        print("              as the geometry baseline for the flipped-vs-clean test only.")
        print()

    # --------------------------------------------------------- flipped vs clean (2nd)
    fc_diffs, cap_all, chance_all = [], [], []
    for step in steps:
        recs = by_step[step]
        pool = [r for r in recs if r.get("group") == "fed"] if tagged else recs
        cap_all += [r["capture"] for r in pool]
        chance_all += [r["chance"] for r in pool]
        w = welch(caps(pool, flipped=True), caps(pool, flipped=False))
        if w is not None:
            fc_diffs.append(w[0])
    res = one_sample_t(fc_diffs)
    scope = "fed only" if tagged else "all canaries"
    print()
    if res is None:
        print(f"  flipped - clean ({scope}): only {len(fc_diffs)} usable update(s)")
        row["fc_gap"], row["fc_p"] = (fc_diffs[0] if fc_diffs else float('nan')), float('nan')
    else:
        m, se, t, df, p = res
        row["fc_gap"], row["fc_se"], row["fc_p"] = m, se, p
        print(f"  flipped - clean ({scope}) = {m:+.6f} +/- {se:.6f}, p = {p:.4g}"
              f"   [needs the shared baseline]")
    row["fc_gaps"] = fc_diffs
    row["capture"] = mean(cap_all)
    row["chance"] = mean(chance_all)
    print(f"  mean capture = {row['capture']:.6f}, chance = {row['chance']:.6f}, "
          f"ratio = {row['capture'] / row['chance']:.1f}x")
    print()
    return row


def report_comparison(rows, alpha):
    if not rows:
        return
    print("=" * 100)
    print("COMPARISON")
    print()

    mem_rows = [r for r in rows if r["tagged"] and "mem_p" in r]
    if mem_rows:
        print("Membership test -- fed vs holdout, flipped canaries. Self-controlled; quote this one.")
        print()
        print(f"{'run':>22} {'fed - holdout':>14} {'se':>10} {'p':>10}  verdict")
        print("-" * 78)
        verdicts = {}
        for r in mem_rows:
            p = r["mem_p"]
            sig = p == p and p < alpha
            verdict = "LEAK DETECTED" if (sig and r["mem_gap"] > 0) else "no detectable leak"
            if sig and r["mem_gap"] < 0:
                verdict = "negative (investigate)"
            # Keyed by (method, mode): auditing galore and subtrack in one go gives two
            # rows called 'skip', and a mode-only key would silently drop one of them.
            verdicts[(r["method"], r["mode"])] = (r["mem_gap"], p, sig and r["mem_gap"] > 0)
            print(f"{r['name']:>22} {r['mem_gap']:>+14.6f} {r.get('mem_se', float('nan')):>10.6f} "
                  f"{p:>10.4g}  {verdict}")
        print()
    else:
        verdicts = {}
        print("No run had a dev-split subspace batch, so the membership test could not run.")
        print("Use --oracle_batch_mode skip and/or private-skip.")
        print()

    methods = []
    for r in rows:
        if r["method"] not in methods:
            methods.append(r["method"])

    for method in methods:
        mine = [r for r in rows if r["method"] == method]
        placebo = next((r for r in mine if r["mode"] == "shared"), None)
        others = [r for r in mine if r["mode"] != "shared" and "fc_gap" in r]
        if not others:
            continue
        print(f"Flipped vs clean, {method} (secondary; baseline-adjusted if a shared run exists)")
        if placebo is None:
            print("  no shared run for this method -- raw, and includes the geometry offset,")
            print("  which is not leakage. Treat as uninterpreted.")
            for r in others:
                print(f"  {r['name']:>22} raw {r['fc_gap']:>+12.6f}  p = {r['fc_p']:.4g}")
        else:
            print(f"  shared geometry offset = {placebo['fc_gap']:+.6f}")
            for r in others:
                adj = r["fc_gap"] - placebo["fc_gap"]
                w = welch(r["fc_gaps"], placebo["fc_gaps"])
                p = w[3] if w else float('nan')
                print(f"  {r['name']:>22} adjusted {adj:>+12.6f}  p = {p:.4g}")
        print()

    print("Reading it")
    print("-" * 100)
    for method in methods:
        leaky = verdicts.get((method, "skip"))
        fixed = verdicts.get((method, "private-skip"))
        if leaky is None and fixed is None:
            continue
        print(f"  [{method}]")
        if leaky is not None and fixed is not None:
            if leaky[2] and not fixed[2]:
                print("    skip leaks, private-skip does not. Records that fed the bare subspace")
                print("    are identifiable from it; once the subspace is a DP release they are")
                print("    not. That is the empirical counterpart of eps = infinity vs finite eps.")
            elif leaky[2] and fixed[2]:
                print("    Both leak. Check the private-skip mechanism is actually binding:")
                print("    subspace/clipped_fraction in the training log should be ~1.0. Well")
                print("    below means C_sub is too large, almost nothing is clipped, and the")
                print("    'privatized' gradient is close to the bare one.")
            else:
                print("    No leak detected even in skip. Before concluding anything, check:")
                print("      - cap/chance above. Near 1.0 means the subspace is not tracking.")
                print("      - mean_rotation_deg in the training log. The dptrack script targets")
                print("        ~10 deg per update; 40+ means ST_STEP_SIZE is far too large and")
                print("        the subspace is thrashing rather than accumulating.")
                print("      - the fed/prior/holdout decay. If it is flat, one record out of a")
                print("        64-batch simply does not move a rank-r subspace measurably at this")
                print("        scale -- itself a reportable result, but say it that way.")
        else:
            missing = "private-skip" if fixed is None else "skip"
            print(f"    Missing the {missing} arm; skip vs private-skip is the comparison.")

    leak_sizes = {m: verdicts[(m, "skip")][0] for m in methods if (m, "skip") in verdicts}
    if len(leak_sizes) > 1:
        print()
        print("  [across methods]")
        ranked = sorted(leak_sizes.items(), key=lambda kv: -kv[1])
        print("    skip-arm leak size: " +
              ",  ".join(f"{m} {v:+.6f}" for m, v in ranked))
        top, bottom = ranked[0], ranked[-1]
        if bottom[1] > 0:
            print(f"    {top[0]} leaks {top[1] / bottom[1]:.1f}x more than {bottom[0]}.")
        else:
            print(f"    {top[0]} leaks; {bottom[0]} does not measurably.")
        print("    So how hard the tracker refits -- not just whether it is data-driven --")
        print("    sets how much a data-dependent subspace gives away.")
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+", help="canary_probe.jsonl file(s)")
    ap.add_argument("--alpha", type=float, default=0.05, help="significance level")
    ap.add_argument("--quiet", action="store_true", help="skip the per-step tables")
    args = ap.parse_args()

    rows = []
    for path in args.logs:
        manifest, by_step = load(path)
        rows.append(report_run(path, manifest, by_step, verbose=not args.quiet))
    report_comparison(rows, args.alpha)

    if _scipy_stats is None:
        print("note: scipy not installed -- p-values use a normal approximation.")


if __name__ == "__main__":
    main()
