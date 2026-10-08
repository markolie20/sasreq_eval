#!/usr/bin/env bash
# The whole suite on the DGX, in order of importance, with several processes sharing each step
# (review-plan/plans/week-budget.md, B8).
#
#   scripts/dgx-run.sh                         every dataset, model and sweep of protocol.toml, one GPU process
#   GPUS="cuda:0 cuda:0" scripts/dgx-run.sh    two processes on GPU 0 (they share the runs through their locks)
#   DRY=1 scripts/dgx-run.sh                   print the steps, run nothing
#   MODELS_ONLY=1 GPUS="cpu cpu cpu cpu" GPU_MODELS=sansa CPU_MODELS="" scripts/dgx-run.sh
#                                              one slow model beside the main run, in 4 processes (see below)
#
# Order: prepare; search and final (stage 1: the main result) with the full data's analysis beside them; the
# finals' diversity; the stage-1 report; then the sweeps one by one, each sweep's analysis beside the next; latency last, alone, since
# it times the CPU; then every report. A stop at any point leaves every finished run whole: rerun the script and
# it skips them. GPU models run on the GPU processes, EASE and popularity on one CPU process beside them.
#
# Exit codes, per step: 0 done; 3 work left over (another process held a run) repeats the step, up to ROUNDS
# times; anything else stops the script: 1 a run failed or was refused (see `seqrec-eval status`: rerun with
# --retry-failed, or accept unrunnable trials with `final --accept-failed`), 130 stopped.
# Stop it with `kill <pid of this script>` (the first line of its output gives the pid, as seen where it runs: in a
# container, `docker exec <container> kill <pid>`): every process it started stops cleanly, and nothing is counted
# as a failed attempt.
#
# MODELS_ONLY=1 runs only the steps that train and score the models named: search, final, their diversity, and
# each sweep's ablate, then their status. No prepare, analysis, latency or report: the main run does those. It is
# for a model too slow to share the main run's steps, which wait for their slowest process (DECISIONS §41): start
# it beside the main run, which leaves the model out, with its own LOGS. GPUS then names one device per process,
# `cpu` included; the run locks keep the processes apart, here and from the main run's.
#
# Settings, from the environment:
#   WORK        work dir                         (default: $SEQREC_EVAL_WORK)
#   DATA_DIR    raw data and caches              (default: $COMPRESSO_DATA_DIR)
#   PROTOCOL    the protocol                     (default: protocol.toml beside this folder)
#   SEQREC_EVAL_PROTOCOL_EXTRA  extra protocol files of [models.*] sections, ':'-separated, which every step
#               reads (DECISIONS §39); list their models in GPU_MODELS or CPU_MODELS to train them
#   GPUS        one torch device per GPU process (default: "cuda:0")
#   GPU_MODELS  models run on the GPU processes  (default: "elsa gru sasrec bert4rec")
#   CPU_MODELS  models run on the CPU process    (default: "popularity ease"; "" for none)
#   DATASETS    datasets                         (default: every one in the protocol)
#   SWEEPS      sweeps, in this order            (default: history_length_inference shuffle history_length density
#                                                 repeat_removal catalogue_top; "none": stage 1 only)
#   LATENCY_THREADS, LATENCY_CORES   latency's --threads (default 4) and --cores (default: none)
#   LOGS        a log per process                (default: $WORK/logs; with MODELS_ONLY=1, $WORK/logs-models)
#   MODELS_ONLY 1: only the model steps, for the models named (default: 0)
#   ROUNDS      repeats of a step with work left (default: 3)
#   SEQREC_EVAL the command                      (default: .venv/bin/seqrec-eval beside this folder)
set -euo pipefail

HERE=$(cd "$(dirname "$0")/.." && pwd)
WORK=${WORK:-${SEQREC_EVAL_WORK:-}}
DATA_DIR=${DATA_DIR:-${COMPRESSO_DATA_DIR:-}}
PROTOCOL=${PROTOCOL:-$HERE/protocol.toml}
read -r -a GPU_DEVICES <<< "${GPUS:-cuda:0}"
read -r -a GPU_MODEL_LIST <<< "${GPU_MODELS:-elsa gru sasrec bert4rec}"
read -r -a CPU_MODEL_LIST <<< "${CPU_MODELS-popularity ease}"
read -r -a DATASET_LIST <<< "${DATASETS:-}"
read -r -a SWEEP_LIST <<< "${SWEEPS:-history_length_inference shuffle history_length density repeat_removal catalogue_top}"
[[ ${SWEEP_LIST[*]} == none ]] && SWEEP_LIST=()  # stage 1 only
LATENCY_THREADS=${LATENCY_THREADS:-4}
LATENCY_CORES=${LATENCY_CORES:-}
ROUNDS=${ROUNDS:-3}
DRY=${DRY:-0}
MODELS_ONLY=${MODELS_ONLY:-0}
SEQREC_EVAL=${SEQREC_EVAL:-$HERE/.venv/bin/seqrec-eval}

[[ -n $WORK ]] || { echo "set WORK (or SEQREC_EVAL_WORK) to the work dir" >&2; exit 2; }
[[ -n $DATA_DIR ]] || { echo "set DATA_DIR (or COMPRESSO_DATA_DIR) to the raw data" >&2; exit 2; }
(( ${#GPU_DEVICES[@]} )) || { echo "GPUS names no device" >&2; exit 2; }
if [[ $MODELS_ONLY == 1 ]]; then LOGS=${LOGS:-$WORK/logs-models}; else LOGS=${LOGS:-$WORK/logs}; fi
[[ $DRY == 1 ]] || mkdir -p "$LOGS"
SELECT=()
(( ${#DATASET_LIST[@]} )) && SELECT=(--dataset "${DATASET_LIST[@]}")
BASE=("$SEQREC_EVAL" --protocol "$PROTOCOL" --work-dir "$WORK")

say() { echo "$(date '+%F %T') $*"; }

# every process started, so a stop reaches all of them; each stops cleanly on SIGTERM (DECISIONS §30, N31)
CHILDREN=()
stop() {
    say "stopping: SIGTERM to every process started"
    (( ${#CHILDREN[@]} )) && kill -TERM "${CHILDREN[@]}" 2>/dev/null || true
    wait || true
    exit 130
}
trap stop TERM INT HUP

# start one process in the background, its output in its own log, its pid in STARTED. Not called in a command
# substitution: that would make the process a subshell's child, which this shell could neither wait for nor stop.
STARTED=
start() {
    local label=$1; shift
    STARTED=
    if [[ $DRY == 1 ]]; then
        echo "== [$label] seqrec-eval ${*}"
        return 0
    fi
    "${BASE[@]}" "$@" > "$LOGS/$label.log" 2>&1 &
    STARTED=$!
    CHILDREN+=("$STARTED")
}

# wait for the given pids; 0 if all were 0, 3 if work is left over, else the first other code
collect() {
    local worst=0 code pid
    for pid in "$@"; do
        code=0
        wait "$pid" || code=$?
        if (( code != 0 && code != 3 )); then
            (( worst == 0 || worst == 3 )) && worst=$code
        elif (( code == 3 && worst == 0 )); then
            worst=3
        fi
    done
    return "$worst"
}

# one step, shared by the GPU processes (GPU models) and the CPU process (CPU models), repeated while work is
# left; LABEL names its logs: LABEL-r<round>-gpu<i>.log, LABEL-r<round>-cpu.log
shared() {
    local label=$1 step=$2; shift 2
    local round code pids i
    for (( round = 1; round <= ROUNDS; round++ )); do
        say "$step $* (round $round)"
        pids=()
        for i in "${!GPU_DEVICES[@]}"; do
            start "$label-r$round-gpu$i" "$step" "${SELECT[@]}" --model "${GPU_MODEL_LIST[@]}" \
                  --device "${GPU_DEVICES[$i]}" "$@"
            pids+=("$STARTED")
        done
        if (( ${#CPU_MODEL_LIST[@]} )); then
            start "$label-r$round-cpu" "$step" "${SELECT[@]}" --model "${CPU_MODEL_LIST[@]}" --device cpu "$@"
            pids+=("$STARTED")
        fi
        [[ $DRY == 1 ]] && return 0
        code=0
        collect "${pids[@]}" || code=$?
        case $code in
            0) return 0 ;;
            3) say "$step $*: work left over (a run another process held, or a model waiting); again" ;;
            *) say "$step $*: stopped with exit code $code; see $LOGS/$label-r$round-*.log and \`seqrec-eval status\`"
               exit "$code" ;;
        esac
    done
    say "$step $*: work still left after $ROUNDS rounds; see \`seqrec-eval status\` (a model blocked by a failed" \
        "trial needs \`search --retry-failed\` or \`final --accept-failed\`)"
    exit 3
}

# one process, waited for; any exit but 0 stops the script
alone() {
    local label=$1; shift
    say "$*"
    local code=0
    start "$label" "$@"
    [[ $DRY == 1 ]] && return 0
    collect "$STARTED" || code=$?
    (( code == 0 )) || { say "$*: exit code $code; see $LOGS/$label.log"; exit "$code"; }
}

# CPU work beside the GPU steps: started now, waited for by finish_background
BACKGROUND=()
beside() {
    local label=$1; shift
    say "beside: $*"
    start "$label" "$@"
    [[ $DRY == 1 ]] || BACKGROUND+=("$STARTED")
}
finish_background() {
    [[ $DRY == 1 ]] && return 0
    (( ${#BACKGROUND[@]} )) || return 0
    local code=0
    collect "${BACKGROUND[@]}" || code=$?
    BACKGROUND=()
    (( code == 0 )) || { say "a background analysis ended with exit code $code; see $LOGS/analyse-*.log"; exit "$code"; }
}

say "seqrec-eval suite (pid $$): protocol $PROTOCOL${SEQREC_EVAL_PROTOCOL_EXTRA:+ with the models of $SEQREC_EVAL_PROTOCOL_EXTRA}," \
    "work $WORK, GPU processes ${GPU_DEVICES[*]}, logs $LOGS"

if [[ $MODELS_ONLY == 1 ]]; then
    # the named models through their own steps, beside the main run (DECISIONS §41)
    MODELS=("${GPU_MODEL_LIST[@]}" "${CPU_MODEL_LIST[@]}")
    say "models only: ${MODELS[*]}"
    shared search search
    shared final final
    alone diversity diversity "${SELECT[@]}" --model "${MODELS[@]}" --device "${GPU_DEVICES[0]}"
    for sweep in "${SWEEP_LIST[@]}"; do
        shared "ablate-$sweep" ablate --sweep "$sweep"
    done
    alone status status "${SELECT[@]}" --model "${MODELS[@]}"
    say "done: ${MODELS[*]} searched, finalised and ablated; latency and the reports come from the main run"
    exit 0
fi

alone plan plan
alone prepare prepare "${SELECT[@]}" --data-dir "$DATA_DIR"

# stage 1, with the full data's analysis (profile, baselines, floor: CPU) beside it
beside analyse-full analyse "${SELECT[@]}" --sweep none
shared search search
shared final final
# coverage and intra-list diversity of the finals: one more test scoring each, on the GPU (diagnostics)
alone diversity diversity "${SELECT[@]}" --device "${GPU_DEVICES[0]}"
finish_background
alone report-stage1 report "${SELECT[@]}" --reference elsa

# the sweeps, in order of importance; each one's analysis beside the next
for sweep in "${SWEEP_LIST[@]}"; do
    shared "ablate-$sweep" ablate --sweep "$sweep"
    beside "analyse-$sweep" analyse "${SELECT[@]}" --sweep "$sweep"
done
finish_background

# latency last and alone: it times the CPU, which every other step loads
latency=(latency "${SELECT[@]}" --threads "$LATENCY_THREADS")
[[ -n $LATENCY_CORES ]] && latency+=(--cores "$LATENCY_CORES")
alone latency "${latency[@]}"

alone report report "${SELECT[@]}" --reference elsa
alone analysis-report analysis-report "${SELECT[@]}"
alone repeat-strata repeat-strata "${SELECT[@]}"
(( ${#SWEEP_LIST[@]} )) && alone ablation-report ablation-report "${SELECT[@]}" --sweep "${SWEEP_LIST[@]}"
alone status status "${SELECT[@]}"
say "done: reports in $WORK/reports"
