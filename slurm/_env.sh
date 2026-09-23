# Shared environment activation — SOURCED (not executed) by the SLURM jobs.
# Loads the cluster modules and activates the conda env. Every value can be
# overridden from the environment; the submit scripts forward them via --export=ALL.

# UTF-8 locale — required here, not optional: the jurisdiction list and prompts carry
# non-ASCII jurisdiction names (Côte d'Ivoire, Türkiye) and the default latin-1
# on this cluster raises UnicodeEncodeError on print.
export LANG="${LANG:-C.UTF-8}"
export LC_ALL=C.UTF-8
export PYTHONUTF8=1

: "${MODULE_MINIFORGE:=miniforge/20251003}"
# Do NOT load a system CUDA module by default: vLLM's torch wheel bundles its own
# CUDA + NCCL, and a system libnccl on LD_LIBRARY_PATH shadows it (undefined
# symbol: ncclCommWindowDeregister). Only the GPU driver is needed. Set
# MODULE_CUDA=cuda/12.9 to override if you ever need the system toolkit.
: "${MODULE_CUDA:=}"
# GCC >= 13.1 is required for two reasons:
#   1. FlashInfer JIT needs -std=c++20; older nvcc silently drops it and libcu++ fails.
#   2. vLLM's bundled libicui18n.so.78 requires CXXABI_1.3.15 (GCC >= 13.1) from
#      libstdc++.so.6. Without this the import chain fails at sqlite3/_sqlite3.
: "${MODULE_GCC:=gcc/14.3.0}"
: "${PROJECT_ENV:=/storage/hpc/41/dolamull/envs/teacher}"   # conda env name or -p prefix path

# `source /etc/profile` makes the `module` command available in a non-interactive
# batch shell. These init scripts aren't `set -u` clean, so relax nounset around them.
set +u
source /etc/profile 2>/dev/null || true
module purge 2>/dev/null || true
[ -n "${MODULE_MINIFORGE}" ] && module load "${MODULE_MINIFORGE}"
[ -n "${MODULE_CUDA}" ]      && module load "${MODULE_CUDA}"
[ -n "${MODULE_GCC}" ]       && module load "${MODULE_GCC}"
# `conda activate` is unreliable in a non-interactive batch shell here, so run it
# best-effort AND prepend the env's bin to PATH — the latter guarantees the right
# python/packages regardless of whether conda activate took effect.
eval "$(conda shell.bash hook)" 2>/dev/null || true
conda activate "${PROJECT_ENV}" 2>/dev/null || true
[[ "${PROJECT_ENV}" == /* && -d "${PROJECT_ENV}/bin" ]] && export PATH="${PROJECT_ENV}/bin:${PATH}"
set -u

# Fail fast (clear message) if python still isn't from the env.
if [[ "${PROJECT_ENV}" == /* ]] && [[ "$(command -v python)" != "${PROJECT_ENV}"/* ]]; then
  echo "ERROR: env not found at ${PROJECT_ENV} (python=$(command -v python)). Check PROJECT_ENV / that the env exists." >&2
  exit 1
fi

# Caches on scratch (fast, large), all under one XDG base. Override via env.
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/scratch/hpc/41/dolamull/.cache}"
export HF_HOME="${HF_HOME:-$XDG_CACHE_HOME/huggingface}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$XDG_CACHE_HOME/pip}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$XDG_CACHE_HOME/triton}"
mkdir -p "$XDG_CACHE_HOME" "$HF_HOME" "$PIP_CACHE_DIR" "$TRITON_CACHE_DIR" 2>/dev/null || true
export TOKENIZERS_PARALLELISM=false

# Results and logs live in the repo; jobs run with cwd = repo root.
mkdir -p results logs 2>/dev/null || true

# Load secrets (HF_TOKEN, OPENAI_API_KEY, ANTHROPIC_API_KEY, DEEPSEEK_API_KEY)
# from the gitignored repo .env. Never commit .env.
if [ -f .env ]; then
  set -a; . ./.env; set +a
fi
[ -n "${HF_TOKEN:-}" ] && export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"

# One-line sanity banner so a failed job's log says immediately what was wrong.
python - <<'PY' 2>/dev/null || echo "WARNING: python sanity check failed"
import importlib, sys
have = []
for m in ("torch", "transformers", "vllm", "openai", "anthropic", "huggingface_hub"):
    try:
        mod = importlib.import_module(m)
        have.append("%s=%s" % (m, getattr(mod, "__version__", "?")))
    except Exception:
        have.append("%s=MISSING" % m)
print("deps: " + " | ".join(have))
PY

echo "env: $(python -V 2>&1) @ $(command -v python) | conda=${PROJECT_ENV} | cuda module=${MODULE_CUDA:-none} | hf_token=$([ -n "${HF_TOKEN:-}" ] && echo set || echo unset)"
