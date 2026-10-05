#!/usr/bin/env bash
# A real run of the suite on this laptop, before the DGX: one dataset, the models you name, every step from
# `prepare` to the reports. Quick by default: it checks that the whole chain works on real data, it does not
# produce results.
#
#   scripts/local-run.sh DATASET MODEL [MODEL ...]
#   scripts/local-run.sh ml20m elsa gru
#   SWEEP=history_length_inference scripts/local-run.sh ml20m elsa gru    # and one ablation sweep
#   FULL=1 scripts/local-run.sh ml20m gru                                # protocol.local.toml as it is
#   DRY=1 scripts/local-run.sh ml20m gru                                 # print the commands, run nothing
#
# Quick means a protocol of its own, written next to the results: TRIALS trials per model, SEEDS (a sweep with
# seeds of its own runs under the first), and EPOCHS for every trained model (ELSA, GRU, SASRec); everything
# else as in protocol.local.toml. Its fingerprints are its own, so nothing here can be mistaken for, or mixed
# with, a real run. FULL=1 copies protocol.local.toml unchanged instead (10 trials, 3 seeds: hours per model on
# real data).
#
# Settings, from the environment:
#   WORK       results                (default: work-local/<dataset>, or work-local-full/<dataset> with FULL=1)
#   DATA_DIR   raw data and caches    (default: $COMPRESSO_DATA_DIR, else ~/Documents/recombee/compresso-recsys/data)
#   CR_SRC     a library source folder to use ahead of the .venv's (default: none; `uv sync` installs the
#              vendored copy, vendor/compresso-recsys)
#   DEVICE     training device        (default: cuda when torch sees one, else cpu)
#   MEM        memory cap             (default: 6G: past it the run is killed, not the laptop; none: no cap)
#   REFERENCE  model the report compares every other with (default: elsa)
#   SWEEP      one [ablations.*] sweep to run as well (default: none)
#   TRIALS, SEEDS, EPOCHS             quick settings (defaults: 2, "0", 1)
#   OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, MKL_NUM_THREADS   numpy's threads (default: 4 each)
set -euo pipefail

if [[ $# -lt 2 ]]; then
    sed -n '2,/^set -euo/{/^set -euo/d; s/^# \{0,1\}//; p}' "$0"
    exit 2
fi
DATASET=$1
shift
MODELS=("$@")

HERE=$(cd "$(dirname "$0")/.." && pwd)
CR_SRC=${CR_SRC:-}
DATA_DIR=${DATA_DIR:-${COMPRESSO_DATA_DIR:-$HOME/Documents/recombee/compresso-recsys/data}}
MEM=${MEM:-6G}
REFERENCE=${REFERENCE:-elsa}
FULL=${FULL:-0}
TRIALS=${TRIALS:-2}
SEEDS=${SEEDS:-0}
EPOCHS=${EPOCHS:-1}
if [[ $FULL == 1 ]]; then
    WORK=${WORK:-$HERE/work-local-full/$DATASET}
else
    WORK=${WORK:-$HERE/work-local/$DATASET}
fi
DEVICE_ARGS=()
if [[ -n ${DEVICE:-} ]]; then
    DEVICE_ARGS=(--device "$DEVICE")
fi
# numpy's BLAS/LAPACK (EASE's inverse, the baselines) ignores torch's thread setting and takes every core
# unless told otherwise; it reads these once, at start-up (review B7)
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}

[[ -z $CR_SRC || -f $CR_SRC/compresso_recsys/__init__.py ]] || { echo "no compresso-recsys source at $CR_SRC" >&2; exit 2; }
mkdir -p "$WORK"
PROTOCOL=$WORK/protocol.toml
if [[ $FULL == 1 ]]; then
    cp "$HERE/protocol.local.toml" "$PROTOCOL"
else
    # stage 1 runs under SEEDS; a sweep with seeds of its own (a retraining sweep: one seed) under the first
    first_seed=${SEEDS%%,*}
    first_seed=${first_seed// /}
    sed -e "s/^seeds = \[[^]]*\]/seeds = [$first_seed]/" \
        -e "/^\[protocol\]/,/^\[/ s/^seeds = \[[^]]*\]/seeds = [$SEEDS]/" \
        -e "s/^trials_per_model = [0-9]*/trials_per_model = $TRIALS/" \
        -e "s/^epochs = { choice = \[[^]]*\] }/epochs = { choice = [$EPOCHS] }/" \
        "$HERE/protocol.local.toml" > "$PROTOCOL.new"
    # every substitution must have happened, or this would not be the quick run it says it is
    if [[ $(sed -n '/^\[protocol\]/,/^\[/p' "$PROTOCOL.new" | grep -c "^seeds = \[$SEEDS\]") != 1 ||
          $(grep -c "^trials_per_model = $TRIALS\b" "$PROTOCOL.new") != 1 ||
          $(grep -c "^epochs = " "$PROTOCOL.new") != $(grep -c "^epochs = { choice = \[$EPOCHS\] }" "$PROTOCOL.new") ]]; then
        echo "protocol.local.toml no longer has the lines this script shortens (seeds, trials_per_model, epochs)" >&2
        rm -f "$PROTOCOL.new"
        exit 1
    fi
    mv "$PROTOCOL.new" "$PROTOCOL"
fi

run() {
    local command=(env ${CR_SRC:+PYTHONPATH="$CR_SRC"} "$HERE/.venv/bin/seqrec-eval" --protocol "$PROTOCOL" --work-dir "$WORK" "$@")
    echo "== seqrec-eval $*" >&2
    if [[ ${DRY:-0} == 1 ]]; then
        return 0
    fi
    if [[ $MEM == none ]]; then
        "${command[@]}"
    else
        systemd-run --user --scope -q -p MemoryMax="$MEM" -p MemorySwapMax=0 "${command[@]}"
    fi
}

SELECT=(--dataset "$DATASET" --model "${MODELS[@]}")
run plan                                                           # the protocol loads and checks
run prepare --dataset "$DATASET" --data-dir "$DATA_DIR"
run analyse --dataset "$DATASET" --sweep none                      # profile, baselines, the floor (CPU)
run search "${SELECT[@]}" "${DEVICE_ARGS[@]}"
run final "${SELECT[@]}" "${DEVICE_ARGS[@]}"
run latency "${SELECT[@]}" --threads 4
run report "${SELECT[@]}" --reference "$REFERENCE"
run repeat-strata "${SELECT[@]}"
if [[ -n ${SWEEP:-} ]]; then
    run ablate "${SELECT[@]}" --sweep "$SWEEP" "${DEVICE_ARGS[@]}"
    run analyse --dataset "$DATASET" --sweep "$SWEEP"              # the floor at every condition
    run ablation-report "${SELECT[@]}" --sweep "$SWEEP"
fi
run status "${SELECT[@]}"
echo "done: reports in $WORK/reports, protocol $PROTOCOL" >&2
