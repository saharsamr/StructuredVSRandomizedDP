#!/bin/bash
#
# One-shot setup for the RoBERTa-Large few-shot experiments.
#
#   bash setup_data.sh
#
# Runs from anywhere -- it locates itself. Idempotent: safe to re-run, it skips whatever
# is already done. Does three things:
#
#   1. Downloads the LM-BFF dataset tarball (~1.9 GB) into data/ and extracts data/original.
#   2. Generates the k-shot splits every run script reads from
#      data/k-shot-1k-test/<TASK>/<K>-<SEED>/.
#   3. Installs the local dpgrape package so `import dpgrape.low_rank_projector_dp` resolves
#      (DP-GaLore / DPTrack-Oracle need it; DP-GRAPE alone does not, so a stale install
#      fails only the two new methods). Set INSTALL_PACKAGE=0 to skip and do it yourself.
#
# Defaults cover the paper's protocol for the six tasks in this study: K=512, seeds
# 13/21/42 (final results) plus 100 (the C search seed).
#
# Overrides:
#   TASKS="SST-2 SNLI"   subset of tasks
#   KS="16 512"          subspace of k values (upstream also builds 16; nothing here uses it)
#   SEEDS="13 21 42"     subset of seeds
#   PYTHON=python        interpreter to use for generation and the package install
#   INSTALL_PACKAGE=0    skip step 3
#
# NOTE: SST-5 is spelled `sst-5` (lowercase) everywhere downstream -- the data directory,
# roberta_finetuning_fewshot.sh's case statement, and the TASK= you pass to the run scripts.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR/data"
ORIG_DIR="$DATA_DIR/original"
TARBALL="$DATA_DIR/datasets.tar"
URL="https://nlp.cs.princeton.edu/projects/lm-bff/datasets.tar"

PYTHON=${PYTHON:-python3}
TASKS=${TASKS:-"SST-2 sst-5 SNLI MNLI trec RTE"}
KS=${KS:-"512"}
SEEDS=${SEEDS:-"13 21 42 100"}
INSTALL_PACKAGE=${INSTALL_PACKAGE:-1}

if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "ERROR: '$PYTHON' not found on PATH. Activate your env, or set PYTHON=<interpreter>." >&2
    exit 1
fi

echo "=============================================================="
echo " tasks   : $TASKS"
echo " k       : $KS"
echo " seeds   : $SEEDS"
echo " python  : $PYTHON  ($($PYTHON -c 'import sys; print(sys.executable)'))"
echo " data dir: $DATA_DIR"
echo "=============================================================="

# ------------------------------------------------------------------ 1. download + extract

mkdir -p "$DATA_DIR"

if [ -d "$ORIG_DIR/SST-2" ]; then
    echo "==> data/original already extracted, skipping download"
else
    if [ ! -f "$TARBALL" ]; then
        echo "==> downloading datasets.tar (~1.9 GB, resumable)"
        if command -v curl >/dev/null 2>&1; then
            curl -fL -C - -o "$TARBALL.part" "$URL"
        elif command -v wget >/dev/null 2>&1; then
            wget -c -O "$TARBALL.part" "$URL"
        else
            echo "ERROR: need curl or wget on PATH" >&2
            exit 1
        fi
        mv "$TARBALL.part" "$TARBALL"
    else
        echo "==> datasets.tar already downloaded"
    fi

    echo "==> extracting"
    tar xf "$TARBALL" -C "$DATA_DIR"
fi

# The tarball ships two SST-2 variants; the paper uses the GLUE one.
if [ -d "$ORIG_DIR/GLUE-SST-2" ] && [ ! -d "$ORIG_DIR/SST-2-original" ]; then
    echo "==> using GLUE-SST-2 as SST-2"
    mv "$ORIG_DIR/SST-2" "$ORIG_DIR/SST-2-original"
    mv "$ORIG_DIR/GLUE-SST-2" "$ORIG_DIR/SST-2"
fi

# ------------------------------------------------------------------ 2. k-shot splits

for K in $KS; do
    echo "==> generating k-shot splits for K=$K"
    # shellcheck disable=SC2086
    "$PYTHON" "$SCRIPT_DIR/roberta_utils/tools/generate_k_shot_data.py" \
        --mode k-shot-1k-test \
        --k "$K" \
        --task $TASKS \
        --seed $SEEDS \
        --data_dir "$ORIG_DIR" \
        --output_dir "$DATA_DIR"
done

# ------------------------------------------------------------------ 3. dpgrape package

PKG_DIR="$(cd -- "$SCRIPT_DIR/../dpgrape" && pwd)"

if [ "$INSTALL_PACKAGE" = "1" ]; then
    if "$PYTHON" -c "import dpgrape.low_rank_projector_dp" >/dev/null 2>&1; then
        echo "==> dpgrape already importable, skipping install"
    else
        echo "==> installing dpgrape (editable) from $PKG_DIR"
        "$PYTHON" -m pip install -e "$PKG_DIR" || {
            echo "    install failed -- data is fine, fix the package by hand:" >&2
            echo "    $PYTHON -m pip install -e $PKG_DIR" >&2
        }
    fi
else
    echo "==> skipping package install (INSTALL_PACKAGE=0)"
fi

# ------------------------------------------------------------------ verify

echo
echo "=============================================================="
echo " verifying"
echo "=============================================================="

missing=0
for K in $KS; do
    for T in $TASKS; do
        for S in $SEEDS; do
            d="$DATA_DIR/k-shot-1k-test/$T/$K-$S"
            if [ -d "$d" ] && compgen -G "$d/train.*" >/dev/null; then
                n=$(find "$d" -maxdepth 1 -type f | wc -l | tr -d ' ')
                printf "  ok    %-6s K=%-4s seed=%-4s  %s files\n" "$T" "$K" "$S" "$n"
            else
                printf "  MISS  %-6s K=%-4s seed=%-4s  %s\n" "$T" "$K" "$S" "$d"
                missing=$((missing + 1))
            fi
        done
    done
done

echo
if "$PYTHON" -c "import dpgrape.low_rank_projector_dp" >/dev/null 2>&1; then
    echo "  ok    dpgrape package (low_rank_projector_dp importable)"
else
    echo "  MISS  dpgrape package is absent or stale -- DP-GaLore / DPTrack will ImportError."
    echo "        fix:  $PYTHON -m pip install -e $PKG_DIR"
    missing=$((missing + 1))
fi

echo
if [ "$missing" -eq 0 ]; then
    echo "All good. Run experiments from $SCRIPT_DIR, e.g."
    echo "  export CUDA_VISIBLE_DEVICES=0"
    echo "  TASK=SST-2 SEED=42 PRIVACY_EPS=6.0 bash roberta_finetuning_dpgrape.sh"
    echo "  TASK=SST-2 SEED=42 PRIVACY_EPS=6.0 bash roberta_finetuning_dptrack.sh"
    echo
    echo "datasets.tar (~1.9 GB) is kept for re-runs; delete it if you need the space:"
    echo "  rm $TARBALL"
else
    echo "$missing item(s) missing -- see above."
    exit 1
fi
