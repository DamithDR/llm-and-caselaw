#!/bin/bash
# ==============================================================================
# SLURM job — the commercial APIs (OpenAI, Anthropic/Claude, DeepSeek).
#
# No GPU: this job only makes HTTPS calls, so it runs on a CPU partition.
# Keys come from the gitignored .env at the repo root, loaded by slurm/_env.sh.
#
# OPTIONAL (export to override defaults):
#   PROVIDERS  — subset of: openai anthropic deepseek   (default: all three)
#   MODELS     — explicit model ids (overrides PROVIDERS)
#   REGIONS    — space-separated region filter          (default: all)
#   LIMIT      — first N jurisdictions (smoke test)
#   REPEATS    — samples per jurisdiction               (default: 1)
#   SLEEP      — seconds between calls, for rate limits (default: 0.2)
#   RUN_TAG    — suffix for the results files
#
# Submit examples
# ---------------
#   sbatch slurm/run_api.sh
#   sbatch --export=ALL,PROVIDERS="openai anthropic" slurm/run_api.sh
#   sbatch --export=ALL,MODELS="gpt-4o-mini",LIMIT=5 slurm/run_api.sh
# ==============================================================================
#SBATCH -J ldv_api
#SBATCH -p cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail

PROVIDERS="${PROVIDERS:-openai anthropic deepseek}"
REPEATS="${REPEATS:-1}"
SLEEP="${SLEEP:-0.2}"
RUN_TAG="${RUN_TAG:-}"

cd "${SLURM_SUBMIT_DIR:-$PWD}"

source slurm/_env.sh

echo "========================================"
echo "Job      : ${SLURM_JOB_NAME:-local} (${SLURM_JOB_ID:-none})"
echo "Node     : ${SLURMD_NODENAME:-$(hostname)}"
echo "Started  : $(date)"
echo "========================================"

python experiments/run_api.py --list

ARGS=(--repeats "$REPEATS" --sleep "$SLEEP")
if [[ -n "${MODELS:-}" ]]; then
    ARGS+=(--models ${MODELS})
else
    ARGS+=(--providers ${PROVIDERS})
fi
[[ -n "${REGIONS:-}" ]] && ARGS+=(--regions ${REGIONS})
[[ -n "${LIMIT:-}"   ]] && ARGS+=(--limit "$LIMIT")
[[ -n "$RUN_TAG"     ]] && ARGS+=(--run-tag "$RUN_TAG")

python experiments/run_api.py "${ARGS[@]}"

echo "========================================"
echo "Finished : $(date)"
echo "========================================"
