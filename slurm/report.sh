#!/bin/bash
# ==============================================================================
# SLURM job — aggregate every results/*.jsonl into the YES/NO statistics.
#
# submit_all.sh chains this with --dependency=singleton so it runs after the
# model and API jobs finish. It is also safe to run standalone at any point:
# partial results still aggregate.
#
#   sbatch slurm/report.sh
# ==============================================================================
#SBATCH -J ldv_report
#SBATCH -p serial
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

source slurm/_env.sh

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="results/report_${STAMP}.txt"

{
    echo "generated: $(date)"
    echo
    python experiments/report.py
    echo
    python experiments/report.py --by-region
    echo
    python experiments/report.py --reasons
    echo
    python experiments/report.py --jurisdictions
} | tee "$OUT"

python experiments/report.py --csv "results/summary_${STAMP}.csv"

echo
echo "wrote $OUT and results/summary_${STAMP}.csv"
