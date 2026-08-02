#!/usr/bin/env python
"""Read canary_probe.jsonl logs and print every number the experiment produces.

    python analyze_canaries.py out/skip/canary_probe.jsonl out/private-skip/canary_probe.jsonl ...

For each run it prints the per-subspace-update capture of the flipped and the clean group,
the gap between them, and a significance test; then one comparison table across runs.

Reading the output
------------------
gap = mean capture(flipped) - mean capture(clean).

The raw gap is NOT the leakage estimate, because it mixes two effects:

  (a) the leak: the subspace was fitted to a batch containing these canaries and chased
      them, so their gradients are captured more.
  (b) geometry: a mislabeled example has a large gradient pointing somewhere unusual, and
      any subspace fitted to any natural batch captures unusual directions less well. This
      shows up as a *negative* gap and is present even when the subspace provably never
      saw the canaries.

The shared arm isolates (b): its subspace is fitted to the train batch, so it cannot
contain any information about the dev canaries, and whatever gap it shows is pure geometry.
The leakage estimate is therefore the baseline-adjusted gap,

    adjusted = gap(arm) - gap(shared)

which is what the ADJUSTED table reports and what you should quote. It is a bias
correction, not a perfect counterfactual: the shared run trains along a different subspace
trajectory, so its geometry offset is close to but not identical to the one inside a skip
run. Run it anyway -- an unadjusted gap is the more wrong number.

  adjusted > 0, significant   the subspace responds to individual dev records. Under
                              --oracle_batch_mode skip that is the leak, measured.
  adjusted ~ 0                no detectable per-record response. Expected for private-skip:
                              clipping bounds each record's pull on the SVD and the noise
                              covers what is left.

capture/chance says whether the subspace is aligned to gradients at all: 1.0 means it is
no better than a random rank-r subspace, and a gap measured under a subspace that is not
tracking anything is not evidence about a subspace that is.

Significance
------------
Within one step, flipped and clean are disjoint sets of examples, so the two-sample Welch
test printed per step is valid. Across steps the same canaries recur, so those records are
correlated and pooling them all into one big test would fake significance. The headline
per-run test therefore treats each subspace update as a single observation of the gap and
runs a one-sample t-test on those K numbers against 0. It is the conservative choice.
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
    var = sum((x - mean) ** 2 for x in xs) / (n - 1)
    return mean, var, n


def welch(a, b):
    """Two-sample Welch t-test. Returns (diff, t, df, p) or None if underpowered."""
    ma, va, na = _mean_var(a)
    mb, vb, nb = _mean_var(b)
    if na < 2 or nb < 2:
        return None
    se2 = va / na + vb / nb
    if se2 <= 0.0:
        return (ma - mb, float('inf') if ma != mb else 0.0, na + nb - 2, 0.0 if ma != mb else 1.0)
    t = (ma - mb) / math.sqrt(se2)
    df = se2 ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    return ma - mb, t, df, _sf(t, df)


def one_sample_t(xs):
    """One-sample t-test against 0. Returns (mean, se, t, df, p) or None."""
    mean, var, n = _mean_var(xs)
    if n < 2:
        return None
    se = math.sqrt(var / n)
    if se == 0.0:
        return (mean, 0.0, float('inf') if mean != 0 else 0.0, n - 1, 0.0 if mean != 0 else 1.0)
    t = mean / se
    return mean, se, t, n - 1, _sf(t, n - 1)


# -------------------------------------------------------------------------------- input

def load(path):
    """Return (manifest, {step: [record, ...]})."""
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
    return manifest, by_step


def label_for(manifest, path):
    mode = manifest.get("oracle_batch_mode", "?")
    method = manifest.get("method", "?")
    if mode == "?" and method == "?":
        return os.path.basename(os.path.dirname(os.path.abspath(path))) or path
    return f"{method}/{mode}"


# ------------------------------------------------------------------------------- report

def report_run(path, manifest, by_step, verbose=True):
    """Print one run's tables. Returns the row for the comparison table."""
    name = label_for(manifest, path)
    print("=" * 92)
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
          f"{manifest['num_labels']} classes, "
          f"canary seed {manifest.get('canary_seed', '?')}")
    print()

    steps = sorted(by_step)
    if verbose:
        print(f"{'step':>7} {'n_flip':>7} {'n_clean':>8} {'capture_flip':>13} "
              f"{'capture_clean':>14} {'gap':>11} {'chance':>10} {'cap/chance':>11} "
              f"{'t':>8} {'p':>9}")
        print("-" * 92)

    gaps, rows = [], []
    for step in steps:
        recs = by_step[step]
        flip = [r["capture"] for r in recs if r["flipped"]]
        clean = [r["capture"] for r in recs if not r["flipped"]]
        chance = [r["chance"] for r in recs]
        w = welch(flip, clean)
        if w is None:
            continue
        gap, t, df, p = w
        mflip = sum(flip) / len(flip)
        mclean = sum(clean) / len(clean)
        mchance = sum(chance) / len(chance) if chance else float('nan')
        gaps.append(gap)
        rows.append((step, mflip, mclean, gap, mchance))
        if verbose:
            ratio = (mflip + mclean) / 2 / mchance if mchance else float('nan')
            print(f"{step:>7} {len(flip):>7} {len(clean):>8} {mflip:>13.6f} "
                  f"{mclean:>14.6f} {gap:>+11.6f} {mchance:>10.6f} {ratio:>11.3f} "
                  f"{t:>8.3f} {p:>9.4f}")

    if not rows:
        print("  (no step had enough canaries in both groups to test)")
        return None

    print()
    res = one_sample_t(gaps)
    mean_chance = sum(r[4] for r in rows) / len(rows)
    mean_capture = sum((r[1] + r[2]) / 2 for r in rows) / len(rows)
    if res is None:
        print(f"  only {len(gaps)} subspace update(s) -- need >= 2 for the run-level test")
        mean_gap = gaps[0]
        se = t = p = float('nan')
        df = 0
    else:
        mean_gap, se, t, df, p = res
        print(f"  RAW GAP   over {len(gaps)} subspace updates = "
              f"{mean_gap:+.6f} +/- {se:.6f} (se)")
        print(f"            one-sample t = {t:.3f}, df = {df}, p = {p:.4g}")
        print(f"            (raw -- subtract the shared arm's gap before quoting this; "
              f"see ADJUSTED below)")
    print(f"            mean capture = {mean_capture:.6f}, "
          f"chance = {mean_chance:.6f}, ratio = {mean_capture / mean_chance:.3f}")

    # relative gap: the gap as a fraction of the capture level, so runs with different
    # r/d are comparable.
    rel = mean_gap / mean_capture if mean_capture else float('nan')
    print(f"            relative gap = {rel:+.2%} of mean capture")
    print()
    return {
        "name": name, "path": path, "mean_gap": mean_gap, "se": se, "t": t, "df": df,
        "p": p, "n_updates": len(gaps), "capture": mean_capture, "chance": mean_chance,
        "rel": rel, "gaps": gaps, "mode": manifest.get("oracle_batch_mode", "?"),
    }


def report_comparison(rows, alpha):
    if len(rows) < 2:
        return
    print("=" * 92)
    print("COMPARISON")
    print()
    print("Raw gaps (not yet baseline-adjusted):")
    print()
    print(f"{'run':>22} {'updates':>8} {'raw gap':>12} {'se':>10} {'p':>10} "
          f"{'rel gap':>10} {'cap/chance':>11}")
    print("-" * 88)
    for r in rows:
        ratio = r["capture"] / r["chance"] if r["chance"] else float('nan')
        print(f"{r['name']:>22} {r['n_updates']:>8} {r['mean_gap']:>+12.6f} "
              f"{r['se']:>10.6f} {r['p']:>10.4g} {r['rel']:>+10.2%} {ratio:>11.3f}")
    print()

    by_mode = {r["mode"]: r for r in rows}
    placebo = by_mode.get("shared")

    if placebo is None:
        print("No 'shared' arm supplied, so the raw gaps above cannot be baseline-adjusted.")
        print("A mislabeled example's gradient is unusual, and any subspace fitted to any")
        print("natural batch captures unusual directions less well -- that offset is in every")
        print("number above and it is not leakage. Re-run with --oracle_batch_mode shared to")
        print("measure it. Until then, treat the table as uninterpreted.")
        print()
        return

    print("ADJUSTED LEAKAGE   (gap minus the placebo's gap -- this is the estimate to quote)")
    print()
    print(f"  placebo (shared) geometry offset = {placebo['mean_gap']:+.6f} "
          f"+/- {placebo['se']:.6f}")
    print("  the subspace there is fitted to the TRAIN batch, so it cannot have seen these")
    print("  dev canaries; whatever gap it shows is geometry, not leakage.")
    print()
    print(f"{'run':>22} {'adjusted gap':>14} {'se':>10} {'p':>10}  verdict")
    print("-" * 78)

    verdicts = {}
    for r in rows:
        if r["mode"] == "shared":
            continue
        adj = r["mean_gap"] - placebo["mean_gap"]
        # Independent runs, so the standard errors of the two per-step gap means add in
        # quadrature; Welch on the two sets of per-step gaps gives the p-value.
        w = welch(r["gaps"], placebo["gaps"])
        if w is None:
            print(f"{r['name']:>22} {adj:>+14.6f} {'--':>10} {'--':>10}  "
                  f"too few updates to test")
            continue
        _, t, df, p = w
        se = math.sqrt(r["se"] ** 2 + placebo["se"] ** 2)
        sig = p < alpha
        verdict = "LEAK DETECTED" if (sig and adj > 0) else "no detectable leak"
        if sig and adj < 0:
            verdict = "negative (investigate)"
        verdicts[r["mode"]] = (adj, p, sig and adj > 0)
        print(f"{r['name']:>22} {adj:>+14.6f} {se:>10.6f} {p:>10.4g}  {verdict}")
    print()

    print("Reading it")
    print("-" * 92)
    leaky = verdicts.get("skip")
    fixed = verdicts.get("private-skip")
    if leaky is not None and fixed is not None:
        if leaky[2] and not fixed[2]:
            print("  skip leaks, private-skip does not. Privatizing the subspace removed the")
            print("  detectable per-record response -- the result the experiment is designed")
            print("  to produce, and the empirical counterpart of eps = inf vs finite eps.")
        elif leaky[2] and fixed[2]:
            print("  Both leak. Check the private-skip mechanism is actually binding:")
            print("  subspace/clipped_fraction in the training log should be ~1.0. If it is")
            print("  well below, C_sub is too large, almost nothing is being clipped, and the")
            print("  'privatized' gradient is close to the bare one.")
        elif not leaky[2]:
            print("  No leak detected even in skip. Either the probe is underpowered (raise")
            print("  --num_canaries, or lower --subspace_T for more updates), or the subspace")
            print("  is not tracking anything -- check cap/chance above and mean_rotation_deg")
            print("  in the training log. A gap measured under a subspace that is not moving")
            print("  is not evidence about a subspace that is.")
    else:
        missing = [m for m in ("skip", "private-skip") if m not in verdicts]
        print(f"  Missing arm(s): {', '.join(missing)}. skip vs private-skip is the")
        print("  comparison the experiment exists to make.")
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
        row = report_run(path, manifest, by_step, verbose=not args.quiet)
        if row is not None:
            rows.append(row)
    report_comparison(rows, args.alpha)

    if _scipy_stats is None:
        print("note: scipy not installed -- p-values use a normal approximation.")


if __name__ == "__main__":
    main()
