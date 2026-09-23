# Jurisdiction-advice experiment

Asks every model, for each of the 195 jurisdictions, whether it can give employment-law
advice specific to that jurisdiction. One fixed dispute; the jurisdiction is the only
variable. The model replies with a four-field JSON object and the runners record
YES / NO plus the reason code.

```
prompts/jurisdiction_advice_prompt.py   the prompt (jurisdiction substituted at call time)
experiments/common.py                   parsing, result storage with resume, statistics
experiments/models_hf.py                open-model registry, 0.5B -> 72.7B
experiments/run_hf.py                   runner for open weights (local or hosted)
experiments/run_api.py                  runner for OpenAI / Anthropic / DeepSeek
experiments/report.py                   YES/NO statistics
results/*.jsonl                         one file per model, git-ignored
```

## Install

```bash
pip install transformers torch accelerate          # --backend local
pip install huggingface_hub                        # --backend api
pip install openai anthropic                       # commercial APIs
```

## Open-weight models

```bash
python experiments/models_hf.py                    # registry + VRAM estimates

# smoke test: one small model, 5 jurisdictions
python experiments/run_hf.py --ids Qwen/Qwen2.5-0.5B-Instruct --limit 5

# everything that fits in ~8 GB
python experiments/run_hf.py --max-params 4

# 4-bit quantisation stretches local inference to ~14B on 12 GB
python experiments/run_hf.py --max-params 15 --load-in-4bit

# the large end, through hosted inference
set HF_TOKEN=hf_...
python experiments/run_hf.py --backend api --min-params 20
```

Licence-gated repos (Llama, Gemma) are **skipped by default** — they need the terms
accepted on the Hub and `huggingface-cli login`, and without that they fail as 403s
mid-sweep. Once you have accepted them, add `--include-gated`.

## Commercial APIs

```bash
set OPENAI_API_KEY=sk-...
set ANTHROPIC_API_KEY=sk-ant-...
set DEEPSEEK_API_KEY=sk-...

python experiments/run_api.py --list                      # check which keys are set
python experiments/run_api.py --models gpt-4o-mini --limit 5
python experiments/run_api.py --providers openai anthropic deepseek
```

Model ids live in `API_MODELS` at the top of `run_api.py` — edit that list as
providers change.

## Statistics

```bash
python experiments/report.py                  # YES/NO per model
python experiments/report.py --by-region      # coverage by region
python experiments/report.py --reasons        # reason-code and confidence breakdown
python experiments/report.py --jurisdictions  # which countries models decline
python experiments/report.py --csv results/summary.csv
```

## Notes

- **Greedy / temperature 0** everywhere. The question is the model's prior, not a sample
  from it. Use `--repeats N` if you want to measure instability deliberately.
- **Resume is automatic.** Re-running skips jurisdictions already recorded for that
  model, so an interrupted sweep costs nothing to restart. Delete the model's
  `results/*.jsonl` to force a rerun.
- **Unparseable replies are counted, not dropped** (`parse_ok: false`). Small models
  often ignore the JSON contract, and that is itself a finding — treating it as missing
  data would flatter them.
- **`raw` is stored** (first 4000 chars) so a claimed YES can be audited later against
  primary sources.
