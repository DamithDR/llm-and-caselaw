#!/bin/bash
# ==============================================================================
# Submit the whole experiment: every open-weight model, the commercial APIs,
# and a report job that runs once they have all finished.
#
# Run from the REPO ROOT — SLURM_SUBMIT_DIR is what the jobs cd into:
#
#   bash slurm/submit_all.sh                # submit everything
#   bash slurm/submit_all.sh --dry-run      # print the sbatch lines, submit nothing
#   bash slurm/submit_all.sh --check        # validate every request against the
#                                           # scheduler (sbatch --test-only), queue nothing
#   bash slurm/submit_all.sh --max-params 15
#   bash slurm/submit_all.sh --smoke        # 5 jurisdictions per model, a quick check
#   bash slurm/submit_all.sh --no-api       # open models only
#   bash slurm/submit_all.sh --no-astro     # H200 only, ignore the astro pool
#
# Resources are derived per model from its parameter count. The cluster has only
# 4 H200 NVL cards (141 GB, gpu11-12) but 23 48 GB cards on the `astro` queue
# (15 L40S on gpu13-17, 8 L40 on gpu09-10), so anything that fits 48 GB goes there
# and the H200s are kept for the models that genuinely need them:
#     <= 15B   1 GPU    astro        TP=1
#     16-35B   2 GPUs   astro        TP=2   (one node; the smallest astro node has 3)
#     36-47B   1 H200   gpu-short    TP=1   (Mixtral is ~107 GB — needs the big card)
#     >  47B   2 H200   gpu-medium   TP=2
#
# --no-astro reverts to the old H200-only sizing (<=47B: 1 GPU, >47B: 2 GPUs).
#
# Each model writes results/<model_slug>.jsonl and resumes if re-submitted,
# so a job that dies partway costs only the jurisdictions it had not reached.
# ==============================================================================

set -euo pipefail

DRY_RUN=0
CHECK_ONLY=0
SMOKE=""
NO_API=0
ASTRO=1
MAX_PARAMS=""
MIN_PARAMS=""
EXTRA_EXPORT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)    DRY_RUN=1; shift ;;
        --check)      CHECK_ONLY=1; shift ;;
        --smoke)      SMOKE="5"; shift ;;
        --limit)      SMOKE="$2"; shift 2 ;;
        --no-api)     NO_API=1; shift ;;
        --no-astro)   ASTRO=0; shift ;;
        --max-params) MAX_PARAMS="$2"; shift 2 ;;
        --min-params) MIN_PARAMS="$2"; shift 2 ;;
        --export)     EXTRA_EXPORT=",$2"; shift 2 ;;
        -h|--help)    sed -n '2,30p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 1 ;;
    esac
done

if [[ ! -d experiments || ! -d prompts ]]; then
    echo "ERROR: run this from the repo root (experiments/ and prompts/ must be here)" >&2
    exit 1
fi
mkdir -p logs results

# Find an interpreter for the registry query below. This runs on the LOGIN node,
# where the conda env is not active — slurm/_env.sh only activates it inside the
# jobs — so plain `python` is usually not on PATH. Nothing here needs the env:
# experiments/models_hf.py is stdlib-only, so any python3 will do. Override with
# PYTHON=/path/to/python if none of these is right.
: "${PROJECT_ENV:=/storage/hpc/41/dolamull/envs/teacher}"
# Partition for the two CPU-only jobs (the API sweep and the report). `serial` is
# this cluster's single-core queue; override if your site names it differently:
#   CPU_PARTITION=short bash slurm/submit_all.sh
CPU_PART="${CPU_PARTITION:-serial}"
# Test that each candidate actually RUNS and is Python 3 — merely existing on PATH
# is not enough (a stub shim or a dangling symlink passes `command -v` and then
# fails at the point of use).
PY_BIN=""
for c in "${PYTHON:-}" "${PROJECT_ENV}/bin/python" python3 python; do
    [[ -z "$c" ]] && continue
    if "$c" -c 'import sys; sys.exit(0 if sys.version_info[0] == 3 else 1)' >/dev/null 2>&1; then
        PY_BIN="$c"; break
    fi
done
if [[ -z "$PY_BIN" ]]; then
    echo "ERROR: no python found on PATH (tried \$PYTHON, ${PROJECT_ENV}/bin/python, python3, python)." >&2
    echo "       The registry query needs stdlib python3 only. Either load the module:" >&2
    echo "           module load miniforge/20251003" >&2
    echo "       or point at an interpreter directly:" >&2
    echo "           PYTHON=${PROJECT_ENV}/bin/python bash slurm/submit_all.sh" >&2
    exit 1
fi

# Ask the registry which models to run, so this script never drifts from
# experiments/models_hf.py. Gated repos are included only if HF_TOKEN is set.
[ -f .env ] && { set -a; . ./.env; set +a; }
INCLUDE_GATED=$([ -n "${HF_TOKEN:-}" ] && echo 1 || echo 0)

MODEL_LINES="$("$PY_BIN" - "$MAX_PARAMS" "$MIN_PARAMS" "$INCLUDE_GATED" "$ASTRO" <<'PY'
import os, sys
sys.path.insert(0, ".")
from experiments import models_hf

mx = float(sys.argv[1]) if sys.argv[1] else None
mn = float(sys.argv[2]) if sys.argv[2] else None
gated_ok = sys.argv[3] == "1"
astro = sys.argv[4] == "1"

H200 = "nvidia_h200_nvl"   # 141 GB, 4 cards total (gpu11-12, 2 x 2) — scarce
# astro holds two card types, both 48 GB: 15 L40S (gpu13-17, 3/node) and 8 L40
# (gpu09-10, 4/node). Requesting an untyped GPU lets a job take whichever is free
# instead of queueing for one type while the other idles — the sizing below is the
# same either way. Pin it with ASTRO_GRES=nvidia_l40s if that ever stops being true.
ASTRO = os.environ.get("ASTRO_GRES", "-")   # "-" = untyped; --nodes=1 keeps a TP pair
                                            # on one node, so never a mixed pair

for m in models_hf.select(max_params=mx, min_params=mn, exclude_gated=not gated_ok):
    p = m["params_b"]
    # vram_gb_bf16 in the registry is params * 2 * 1.15; usable VRAM is the card
    # times GPU_MEMORY_UTIL (0.90), i.e. 43 GB per astro card and 127 GB per H200.
    if p > 47:            # 70-72B is ~165 GB: two H200 (254 GB usable)
        tp, gpus, gres, part, walltime = 2, 2, H200, "gpu-medium", "06:00:00"
    elif p > 35:          # Mixtral-8x7B is ~107 GB: one H200. Three 48 GB cards
                          # would need TP=3, and its 32 heads do not divide by 3
        tp, gpus, gres, part, walltime = 1, 1, H200, "gpu-short", "03:00:00"
    elif p > 15 and astro:   # 27-33B is 63-76 GB: two 48 GB cards (86 GB), one node
        tp, gpus, gres, part, walltime = 2, 2, ASTRO, "astro", "04:00:00"
    elif p > 15:
        tp, gpus, gres, part, walltime = 1, 1, H200, "gpu-short", "03:00:00"
    elif astro:           # <= 15B is <= 34 GB: fits one 48 GB card, KV cache included
        tp, gpus, gres, part, walltime = 1, 1, ASTRO, "astro", "03:00:00"
    else:
        tp, gpus, gres, part, walltime = 1, 1, H200, "gpu-short", "02:00:00"
    print("%s\t%s\t%d\t%d\t%s\t%s\t%s" % (m["id"], p, tp, gpus, gres, part, walltime))
PY
)"

if [[ -z "$MODEL_LINES" ]]; then
    echo "no models matched the filters" >&2
    exit 1
fi

echo "=============================================================================="
echo "Submitting jurisdiction-advice sweep"
echo "  repo        : $PWD"
echo "  models      : $(echo "$MODEL_LINES" | wc -l)"
echo "  gated repos : $([ "$INCLUDE_GATED" = 1 ] && echo "included (HF_TOKEN set)" || echo "skipped (no HF_TOKEN)")"
echo "  jurisdiction: ${SMOKE:-all 195}"
echo "  gpu pool    : $([ "$ASTRO" = 1 ] && echo "astro 48GB (L40/L40S) for <=35B, H200 above" || echo "H200 only (--no-astro)")"
echo "  python      : $(command -v "$PY_BIN") (registry query only; jobs use PROJECT_ENV)"
echo "  cpu jobs    : partition $CPU_PART (api + report)"
echo "  mode        : $([ "$DRY_RUN" = 1 ] && echo "dry run" || { [ "$CHECK_ONLY" = 1 ] && echo "check (sbatch --test-only)" || echo "submit"; })"
echo "=============================================================================="

JOB_IDS=()

submit() {   # submit <description> <sbatch args...>
    local desc="$1"; shift
    if [[ "$DRY_RUN" = 1 ]]; then
        printf '  [dry-run] %-58s sbatch %s\n' "$desc" "$*"
        return 0
    fi
    if [[ "${CHECK_ONLY:-0}" = 1 ]]; then
        # Validate the request against the scheduler without queueing anything.
        if sbatch --test-only "$@" >/dev/null 2>&1; then
            printf '  ok       %-58s\n' "$desc"
        else
            printf '  FAILS    %-58s %s\n' "$desc" \
                   "$(sbatch --test-only "$@" 2>&1 | tail -1)"
        fi
        return 0
    fi
    local out
    out="$(sbatch "$@")" || { echo "  !! failed to submit $desc" >&2; return 1; }
    local id="${out##* }"
    JOB_IDS+=("$id")
    printf '  submitted %-58s job %s\n' "$desc" "$id"
}

# ── one job per open-weight model ─────────────────────────────────────────────
while IFS=$'\t' read -r MODEL PARAMS TP GPUS GRES PART WALLTIME; do
    [[ -z "$MODEL" ]] && continue
    SLUG="$(echo "${MODEL##*/}" | tr '[:upper:]' '[:lower:]')"
    EXPORTS="ALL,MODEL=${MODEL},TENSOR_PARALLEL_SIZE=${TP}"
    [[ -n "$SMOKE" ]] && EXPORTS="${EXPORTS},LIMIT=${SMOKE}"
    EXPORTS="${EXPORTS}${EXTRA_EXPORT}"
    # "-" means any GPU in the partition; otherwise pin the card type.
    if [[ "$GRES" == "-" ]]; then
        GRES_ARG="gpu:${GPUS}"; CARD="gpu"
    else
        GRES_ARG="gpu:${GRES}:${GPUS}"
        CARD="${GRES##nvidia_}"; CARD="${CARD%%_nvl}"
    fi
    submit "${MODEL} (${PARAMS}B, ${GPUS}x${CARD}, ${PART})" \
        --job-name="ldv_${SLUG}" \
        --partition="$PART" \
        --gres="$GRES_ARG" \
        --nodes=1 \
        --time="$WALLTIME" \
        --export="$EXPORTS" \
        slurm/run_model.sh
done <<< "$MODEL_LINES"

# ── one CPU job for the commercial APIs ───────────────────────────────────────
if [[ "$NO_API" = 0 ]]; then
    API_EXPORTS="ALL"
    [[ -n "$SMOKE" ]] && API_EXPORTS="${API_EXPORTS},LIMIT=${SMOKE}"
    API_EXPORTS="${API_EXPORTS}${EXTRA_EXPORT}"
    if [[ -z "${OPENAI_API_KEY:-}${ANTHROPIC_API_KEY:-}${DEEPSEEK_API_KEY:-}" ]]; then
        echo "  note: no API keys in .env — skipping the API job"
    else
        submit "commercial APIs (openai/anthropic/deepseek)" \
            --partition="$CPU_PART" --export="$API_EXPORTS" slurm/run_api.sh
    fi
fi

# ── report, after everything else finishes (ok even if some jobs failed) ───────
# The report is the LAST thing submitted, so a bad partition here used to surface
# only after every model job was already queued — hence --check.
if [[ "$DRY_RUN" = 1 ]]; then
    echo "  [dry-run] report                                 sbatch --partition=$CPU_PART slurm/report.sh"
elif [[ "$CHECK_ONLY" = 1 ]]; then
    if sbatch --test-only --partition="$CPU_PART" slurm/report.sh >/dev/null 2>&1; then
        printf '  ok       %-58s\n' "report (partition $CPU_PART)"
    else
        printf '  FAILS    %-58s %s\n' "report (partition $CPU_PART)" \
               "$(sbatch --test-only --partition="$CPU_PART" slurm/report.sh 2>&1 | tail -1)"
    fi
elif [[ ${#JOB_IDS[@]} -gt 0 ]]; then
    DEP="afterany:$(IFS=:; echo "${JOB_IDS[*]}")"
    out="$(sbatch --dependency="$DEP" --kill-on-invalid-dep=yes \
                  --partition="$CPU_PART" slurm/report.sh)"
    echo "  submitted report (waits on ${#JOB_IDS[@]} job(s))    job ${out##* }"
fi

echo "=============================================================================="
[[ "$DRY_RUN" = 1 || "$CHECK_ONLY" = 1 ]] || {
    echo "  squeue -u \$USER            # watch progress"
    echo "  tail -f logs/ldv_*.out     # follow a job"
    echo "  $PY_BIN experiments/report.py   # statistics at any time"
}
echo "=============================================================================="
