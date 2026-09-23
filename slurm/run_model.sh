#!/bin/bash
# ==============================================================================
# Parameterised SLURM job — one open-weight model over all 195 jurisdictions.
#
# REQUIRED (pass via --export at sbatch time):
#   MODEL                — HuggingFace model ID from experiments/models_hf.py
#
# OPTIONAL (export to override defaults):
#   TENSOR_PARALLEL_SIZE — GPUs for tensor parallelism      (default: 1)
#   GPU_MEMORY_UTIL      — fraction of GPU memory for vLLM  (default: 0.90)
#   MAX_MODEL_LEN        — vLLM context length              (default: 4096)
#   BACKEND              — vllm | local                     (default: vllm)
#   REGIONS              — space-separated region filter     (default: all)
#   LIMIT                — first N jurisdictions (smoke test)
#   REPEATS              — samples per jurisdiction          (default: 1)
#   RUN_TAG              — suffix for the results file
#
# Output: results/<model_slug>.jsonl  (append-only, resumes if re-run)
#
# The default #SBATCH resources are one L40S (48 GB) on the `astro` queue, which
# suits everything up to ~15B and leaves the cluster's 4 H200 cards free for the
# models that need them. Command-line flags take precedence over the defaults
# below, so size up per model:
#
#   27-33B   --partition=astro --gres=gpu:nvidia_l40s:2   TENSOR_PARALLEL_SIZE=2
#   36-47B   --partition=gpu-short  --gres=gpu:nvidia_h200_nvl:1   TP=1
#   70B+     --partition=gpu-medium --gres=gpu:nvidia_h200_nvl:2   TP=2
#
# Submit examples
# ---------------
#   sbatch --export=ALL,MODEL=Qwen/Qwen2.5-7B-Instruct slurm/run_model.sh
#
#   sbatch --partition=astro --gres=gpu:nvidia_l40s:2 --time=04:00:00 \
#          --export=ALL,MODEL=Qwen/Qwen2.5-32B-Instruct,TENSOR_PARALLEL_SIZE=2 \
#          slurm/run_model.sh
#
#   sbatch --partition=gpu-medium --gres=gpu:nvidia_h200_nvl:2 --time=06:00:00 \
#          --export=ALL,MODEL=Qwen/Qwen2.5-72B-Instruct,TENSOR_PARALLEL_SIZE=2 \
#          slurm/run_model.sh
#
#   sbatch --export=ALL,MODEL=Qwen/Qwen2.5-0.5B-Instruct,LIMIT=5 slurm/run_model.sh
# ==============================================================================
#SBATCH -J ldv_model
#SBATCH -p astro
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=03:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail

if [[ -z "${MODEL:-}" ]]; then
    echo "ERROR: MODEL is not set. Pass it via --export=ALL,MODEL=<model-id>" >&2
    exit 1
fi

TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
BACKEND="${BACKEND:-vllm}"
REPEATS="${REPEATS:-1}"
RUN_TAG="${RUN_TAG:-}"

# SLURM copies the script to its spool dir before execution, so $0 does not point
# at the repo. SLURM_SUBMIT_DIR is the directory sbatch was invoked from — run
# sbatch from the repo root.
cd "${SLURM_SUBMIT_DIR:-$PWD}"

source slurm/_env.sh

echo "========================================"
echo "Job      : ${SLURM_JOB_NAME:-local} (${SLURM_JOB_ID:-none})"
echo "Node     : ${SLURMD_NODENAME:-$(hostname)}"
echo "GPUs     : ${CUDA_VISIBLE_DEVICES:-none}"
echo "Model    : $MODEL"
echo "Backend  : $BACKEND  (TP=$TENSOR_PARALLEL_SIZE, util=$GPU_MEMORY_UTIL)"
echo "Started  : $(date)"
echo "========================================"

ARGS=(--backend "$BACKEND" --ids "$MODEL" --repeats "$REPEATS")
[[ "$BACKEND" == "vllm" ]] && ARGS+=(--tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
                                     --gpu-memory-utilization "$GPU_MEMORY_UTIL"
                                     --max-model-len "$MAX_MODEL_LEN")
[[ -n "${REGIONS:-}" ]] && ARGS+=(--regions ${REGIONS})
[[ -n "${LIMIT:-}"   ]] && ARGS+=(--limit "$LIMIT")
[[ -n "$RUN_TAG"     ]] && ARGS+=(--run-tag "$RUN_TAG")
# The registry skips licence-gated repos by default; allow them when a token is set.
[[ -n "${HF_TOKEN:-}" ]] && ARGS+=(--include-gated)

python experiments/run_hf.py "${ARGS[@]}"

echo "========================================"
echo "Finished : $(date)"
echo "========================================"
