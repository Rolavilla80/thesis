# RQ2 — different angle

This is an isolated, intrinsic replication study of the interaction between
SuperBPE and parity-aware BPE. It deliberately has no dependency on sibling
experiments in the parent repository.

## Question

For a target vocabulary `V`, we expose two vocabulary-index transition points:

```text
256 ── t_super ── t_parity ── V
      Csub          Csuper       PAsuper
```

- `Csub`: classical merge selection; whitespace blocks merging.
- `Csuper`: classical merge selection; merges may cross whitespace.
- `PAsuper`: parity-aware selection; merges may cross whitespace.

The primary local pilot is `V=20,000`, `t_super=10,000`, and
`t_parity=15,000`. Since byte BPE starts with 256 base symbols, that schedule
means 9,744 Csub merges, 5,000 Csuper merges, and 5,000 PAsuper merges.

## Design principles

- All training, development, evaluation, and generated artifacts belong under
  this directory.
- Parity selection uses a parallel development corpus. Evaluation keeps two
  intentionally distinct equality measures: compression equality (Gini of
  bytes per token, which is the quantity the current parity selector
  equalizes) and token workload equality (Gini of tokens per aligned
  segment). UTF-8 script width makes them answer different questions.
- Overall encoding efficiency is measured in bytes per token.
- The smoke test uses generated text only to validate mechanics. It is not a
  scientific result.

## Layout

```text
src/different_angle/   Independent byte-BPE learner and metrics
scripts/               Smoke and later pilot entry points
configs/               Reproducible schedule/data specifications
data/                  Fresh local corpora only (git-ignored)
results/               Generated merges and measurements (git-ignored)
```

## Run the smoke test

The current Python environment needs the `regex` package.

```powershell
$env:PYTHONPATH = 'src'
python scripts/smoke.py
```

The next data-dependent pilot is described in
[`configs/pilot_20k.json`](configs/pilot_20k.json).

## Commit-compatible local path

The linked parity-aware-BPE commit is represented explicitly by
`pretokenizer: "commit_bytelevel"`:

- phase 1 uses Hugging Face `Whitespace()` followed by
  `ByteLevel(use_regex=False)`;
- phase 2 uses `ByteLevel(use_regex=False)` only;
- `parity_reference: "phase"` measures relative compression from the start
  of the parity-learning invocation, as the original `--ratio 1` continuation
  does after it preloads previous merges.

`parity_reference: "global"` is retained as the separate research variant:
it measures relative to immutable original byte lengths across the whole
curriculum. It is not the direct-continuation setting.

Install the project dependencies once:

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
```

Then run the local study in order from this directory:

```powershell
# 1. Check the commit-compatible pretokenization, parity mode, and schedules.
.\.venv\Scripts\python.exe scripts/run_local_plan.py 1

# 2. Run the 20k SuperBPE control and 20k dual curriculum on 500 KB/language.
.\.venv\Scripts\python.exe scripts/run_local_plan.py 2

# 3. Run the 80k SuperBPE control and 80k tSuper=40k, tParity=60k dual curriculum on 1 MB/language.
.\.venv\Scripts\python.exe scripts/run_local_plan.py 3

# 4. Sweep parity timing at tParity=70k and 50k, with tSuper fixed at 40k.
.\.venv\Scripts\python.exe scripts/run_local_plan.py 4

# 5. Sweep SuperBPE timing at tSuper=30k and 50k, with tParity fixed at 60k.
.\.venv\Scripts\python.exe scripts/run_local_plan.py 5
```

Steps 2 and 3 save merge-level checkpoints under `checkpoints/` every 1,000
merges. To resume after an interruption, rerun the same command with
`--resume`; a condition without a checkpoint starts normally.
