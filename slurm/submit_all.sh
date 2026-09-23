#!/bin/bash
# ==============================================================================
# Submit the whole experiment: every open-weight model, the commercial APIs,
# and a report job that runs once they have all finished.
#
# Run from the REPO ROOT — SLURM_SUBMIT_DIR is what the jobs cd into:
#
#   bash slurm/submit_all.sh                # submit everything
#   bash slurm/submit_all.sh --dry-run      # print the sbatch lines, submit nothing
#   bash slurm/submit_all.sh --max-params 15
#   bash slurm/submit_all.sh --smoke        # 5 jurisdictions per model, a quick check
#   bash slurm/submit_all.sh --no-api       # open models only
#   bash slurm/submit_all.sh --no-astro     # H200 only, ignore the L40S pool
#
# Resources are derived per model from its parameter count. The cluster has only
# 4 H200 NVL cards (141 GB, 2 nodes x 2) but 15 L40S (48 GB, 5 nodes x 3) on the
# `astro` queue, so anything that fits 48 GB goes there and the H200s are kept for
# the models that genuinely need them:
#     <= 15B   1 L40S   astro        TP=1
#     16-35B   2 L40S   astro        TP=2   (one node; L40S nodes hold 3)
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
SMOKE=""
NO_API=0
ASTRO=1
MAX_PARAMS=""
MIN_PARAMS=""
EXTRA_EXPORT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)    DRY_RUN=1; shift ;;
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

# Ask the registry which models to run, so this script never drifts from
# experiments/models_hf.py. Gated repos are included only if HF_TOKEN is set.
[ -f .env ] && { set -a; . ./.env; set +a; }
INCLUDE_GATED=$([ -n "${HF_TOKEN:-}" ] && echo 1 || echo 0)

MODEL_LINES="$(python - "$MAX_PARAMS" "$MIN_PARAMS" "$INCLUDE_GATED" "$ASTRO" <<'PY'
import sys
sys.path.insert(0, ".")
from experiments import models_hf

mx = float(sys.argv[1]) if sys.argv[1] else None
mn = float(sys.argv[2]) if sys.argv[2] else None
gated_ok = sys.argv[3] == "1"
astro = sys.argv[4] == "1"

H200 = "nvidia_h200_nvl"   # 141 GB, 4 cards total (2 nodes x 2) — scarce
L40S = "nvidia_l40s"       # 48 GB, 15 cards (5 nodes x 3) on the astro queue

for m in models_hf.select(max_params=mx, min_params=mn, exclude_gated=not gated_ok):
    p = m["params_b"]
    # vram_gb_bf16 in the registry is params * 2 * 1.15; usable VRAM is the card
    # times GPU_MEMORY_UTIL (0.90), i.e. 43 GB per L40S and 127 GB per H200.
    if p > 47:            # 70-72B is ~165 GB: two H200 (254 GB usable)
        tp, gpus, gres, part, walltime = 2, 2, H200, "gpu-medium", "06:00:00"
    elif p > 35:          # Mixtral-8x7B is ~107 GB: one H200, but 3x L40S would
                          # need TP=3 and its 32 heads do not divide by 3
        tp, gpus, gres, part, walltime = 1, 1, H200, "gpu-short", "03:00:00"
    elif p > 15 and astro:   # 27-33B is 63-76 GB: two L40S (86 GB usable), one node
        tp, gpus, gres, part, walltime = 2, 2, L40S, "astro", "04:00:00"
    elif p > 15:
        tp, gpus, gres, part, walltime = 1, 1, H200, "gpu-short", "03:00:00"
    elif astro:           # <= 15B is <= 34 GB: fits one L40S with room for the KV cache
        tp, gpus, gres, part, walltime = 1, 1, L40S, "astro", "03:00:00"
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
echo "  gpu pool    : $([ "$ASTRO" = 1 ] && echo "astro L40S for <=35B, H200 above" || echo "H200 only (--no-astro)")"
echo "  dry run     : $([ "$DRY_RUN" = 1 ] && echo yes || echo no)"
echo "=============================================================================="

JOB_IDS=()

submit() {   # submit <description> <sbatch args...>
    local desc="$1"; shift
    if [[ "$DRY_RUN" = 1 ]]; then
        printf '  [dry-run] %-58s sbatch %s\n' "$desc" "$*"
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
    # Short label for the GPU type, for the submission line only.
    CARD="${GRES##nvidia_}"; CARD="${CARD%%_nvl}"
    submit "${MODEL} (${PARAMS}B, ${GPUS}x${CARD}, ${PART})" \
        --job-name="ldv_${SLUG}" \
        --partition="$PART" \
        --gres="gpu:${GRES}:${GPUS}" \
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
            --export="$API_EXPORTS" slurm/run_api.sh
    fi
fi

# ── report, after everything else finishes (ok even if some jobs failed) ───────
if [[ "$DRY_RUN" = 1 ]]; then
    echo "  [dry-run] report                                 sbatch slurm/report.sh"
elif [[ ${#JOB_IDS[@]} -gt 0 ]]; then
    DEP="afterany:$(IFS=:; echo "${JOB_IDS[*]}")"
    out="$(sbatch --dependency="$DEP" --kill-on-invalid-dep=yes slurm/report.sh)"
    echo "  submitted report (waits on ${#JOB_IDS[@]} job(s))    job ${out##* }"
fi

echo "=============================================================================="
[[ "$DRY_RUN" = 1 ]] || {
    echo "  squeue -u \$USER            # watch progress"
    echo "  tail -f logs/ldv_*.out     # follow a job"
    echo "  python experiments/report.py   # statistics at any time"
}
echo "=============================================================================="
