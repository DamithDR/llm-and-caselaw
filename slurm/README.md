# SLURM jobs

Runs the jurisdiction-advice sweep on the cluster: every open-weight model over all
195 jurisdictions, the commercial APIs, then the YES/NO report.

```
slurm/_env.sh          sourced by every job — modules, conda env, caches, .env secrets
slurm/run_model.sh     one open-weight model, all jurisdictions (GPU, vLLM)
slurm/run_api.sh       OpenAI + Anthropic + DeepSeek (CPU, no GPU needed)
slurm/report.sh        aggregates results/*.jsonl into the statistics
slurm/submit_all.sh    submits all of the above, with the report chained last
```

## Run everything

Always submit **from the repo root** — the jobs `cd` into `SLURM_SUBMIT_DIR`.

```bash
bash slurm/submit_all.sh --dry-run     # see the sbatch lines first
bash slurm/submit_all.sh --check       # ask the scheduler to validate them, queue nothing
bash slurm/submit_all.sh --smoke       # 5 jurisdictions per model, ~15 min end to end
bash slurm/submit_all.sh               # the real sweep
```

`--check` runs every request through `sbatch --test-only`, so a bad partition or an
impossible `--gres` is caught before anything is queued. Worth doing after any change to
partitions or sizing: the report job is submitted last, so without it a mistake there
only surfaces once all 13 model jobs are already in the queue.

Useful variants:

```bash
bash slurm/submit_all.sh --max-params 15    # skip the big models
bash slurm/submit_all.sh --min-params 20    # only the big models
bash slurm/submit_all.sh --no-api           # open weights only
bash slurm/submit_all.sh --no-astro         # H200 only, do not use the L40S pool
```

### python on the login node

`submit_all.sh` runs a short stdlib-only query against `experiments/models_hf.py` to
decide each model's resources, so it needs an interpreter **on the login node** — where
the conda env is not active, since `_env.sh` only activates it inside the jobs. It tries
`$PYTHON`, then `$PROJECT_ENV/bin/python`, then `python3`, then `python`, and checks each
one actually runs rather than just resolving on `PATH`. If it finds none:

```bash
module load miniforge/20251003            # or
PYTHON=/storage/hpc/41/dolamull/envs/teacher/bin/python bash slurm/submit_all.sh
```

The banner prints which interpreter it picked. It is used only for that query — the jobs
themselves always source `slurm/_env.sh` and run from `PROJECT_ENV`.

### CPU jobs

`run_api.sh` (HTTPS calls only) and `report.sh` (aggregation) need no GPU and go to
`serial`, this cluster's single-core queue. If your site names it differently:

```bash
CPU_PARTITION=short bash slurm/submit_all.sh
```

`sinfo -s` lists the real partition names if `serial` is ever rejected.

## Resources

The [cluster's GPUs](https://lancaster-hec.readthedocs.io/en/latest/gpu.html) are not
evenly stocked, and that drives the whole sizing policy:

| Card | VRAM | Cards | Per node | Queues | Max GPUs/user |
|---|---|---|---|---|---|
| H200 NVL | 141 GB | **4** (2 nodes) | 2 | `gpu-short`, `gpu-medium`, `gpu-long` | unlimited / 6 / 2 |
| L40S | 48 GB | **15** (5 nodes) | 3 | `astro` (24 h) | 15 |
| L40 | 48 GB | 8 (2 nodes) | 4 | `astro` (24 h) | 15 |
| V100 | 32 GB | 24 (8 nodes) | 3 | `gpu-short`, `gpu-medium`, `gpu-long` | as above |

There are only four H200 cards on the whole machine, so putting all twenty models on
them serialises the sweep. Fifteen of the twenty are ≤ 15B and fit one 48 GB L40S, so
they go to `astro` and the H200s are reserved for the models that actually need them:

| Model size | GPUs | TP | Partition | Card | Walltime |
|---|---|---|---|---|---|
| ≤ 15B | 1 | 1 | `astro` | L40S | 03:00:00 |
| 16–35B | 2 | 2 | `astro` | L40S | 04:00:00 |
| 36–47B | 1 | 1 | `gpu-short` | H200 | 03:00:00 |
| > 47B | 2 | 2 | `gpu-medium` | H200 | 06:00:00 |

Usable VRAM is the card times `GPU_MEMORY_UTIL` (0.90): 43 GB per L40S, 127 GB per H200.
Against the registry's `vram_gb_bf16` (params × 2 × 1.15) that gives:

- **≤ 15B** — phi-4 and Qwen2.5-14B are the ceiling at ~34 GB; comfortable on one L40S.
- **16–35B** — gemma-2-27b is ~63 GB and Qwen2.5-32B ~75 GB, so TP=2 (86 GB). Both have
  head counts divisible by 2, and `--nodes=1` keeps the pair on one L40S node.
- **36–47B** — Mixtral-8x7B is ~107 GB. It would need TP=3 on L40S, but its 32 attention
  heads do not divide by 3, so it takes a single H200 instead.
- **> 47B** — 70B/72B are ~165 GB: two H200 (254 GB). Out of reach for a 3-GPU L40S node.

With gated repos included the full sweep asks for 19 L40S at once against a 15 GPU/user
cap, so the last few jobs simply pend on `AssocGrpGRES` until the earlier ones finish —
that is a queueing delay, not a rejection.

`--no-astro` reverts to the old H200-only sizing if you cannot get onto that queue.
Access is via FairShare for everyone; only the Observational Astrophysics Group gets the
priority `astro-tier1` / `astro-tier2` queues, which this project does not use.

Generation itself is quick: 195 prompts × ~220 tokens batched through vLLM. Wall time
is dominated by downloading and loading the weights, which is why the walltimes are
generous relative to the work. The L40S walltimes are a little longer than the H200
equivalents would be: the card is roughly a third of an H200 in bf16 throughput, and
the TP=2 pairs talk over PCIe rather than NVLink.

## Single jobs

`run_model.sh` defaults to one L40S on `astro`, so small models need no flags:

```bash
sbatch --export=ALL,MODEL=Qwen/Qwen2.5-7B-Instruct slurm/run_model.sh

sbatch --partition=astro --gres=gpu:nvidia_l40s:2 --time=04:00:00 \
       --export=ALL,MODEL=Qwen/Qwen2.5-32B-Instruct,TENSOR_PARALLEL_SIZE=2 \
       slurm/run_model.sh

sbatch --partition=gpu-medium --gres=gpu:nvidia_h200_nvl:2 --time=06:00:00 \
       --export=ALL,MODEL=Qwen/Qwen2.5-72B-Instruct,TENSOR_PARALLEL_SIZE=2 \
       slurm/run_model.sh

sbatch --export=ALL,PROVIDERS="openai anthropic" slurm/run_api.sh
sbatch slurm/report.sh
```

## Secrets

`slurm/_env.sh` sources a gitignored `.env` at the repo root:

```bash
HF_TOKEN=hf_...              # gated repos (Llama, Gemma); without it they are skipped
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
DEEPSEEK_API_KEY=sk-...
```

Copy `.env.example` to `.env` and fill in whatever you have. Missing keys are not
fatal: `submit_all.sh` skips the API job if none are set, and `run_model.sh` only
passes `--include-gated` when `HF_TOKEN` is present.

## Environment

Inherited from the existing cluster setup: `miniforge/20251003`, `gcc/14.3.0`, no
system CUDA module (vLLM's torch wheel bundles its own — a system `libnccl` on
`LD_LIBRARY_PATH` shadows it), conda env at `PROJECT_ENV`, caches under
`/scratch/hpc/41/dolamull/.cache`.

Override any of them at submit time:

```bash
sbatch --export=ALL,MODEL=...,PROJECT_ENV=/path/to/other/env slurm/run_model.sh
```

`_env.sh` prints a dependency banner at the top of every job log:

```
deps: torch=2.x | transformers=4.x | vllm=0.x | openai=MISSING | anthropic=MISSING | huggingface_hub=0.x
```

If `openai`/`anthropic` show `MISSING`, the GPU jobs are unaffected — only
`run_api.sh` needs them:

```bash
python -m pip install openai anthropic
```

That env is shared with your other project, so install deliberately rather than
letting a job do it.

## Monitoring

```bash
squeue -u $USER
tail -f logs/ldv_*.out
python experiments/report.py        # works on partial results at any time
```

## Resume

Each model appends to `results/<model_slug>.jsonl` and skips jurisdictions already
recorded. A job that hits the walltime or dies mid-sweep can simply be resubmitted —
it picks up where it stopped. To force a clean rerun, delete that model's file.
