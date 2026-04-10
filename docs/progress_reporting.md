# Progress Reporting

## Overview

Long-running loops across all five pipeline modules emit tqdm-based progress bars.
The implementation uses the shared `ai_fields.common.progress` utilities, which are
TTY-aware and support explicit opt-in/opt-out via the `AI_FIELDS_PROGRESS` environment
variable or a `progress_enabled` keyword argument.

No data contract or manifest behavior is affected.  All progress instrumentation is
purely observational.

---

## Where progress bars appear

| Module | Function | Bar label | Unit | When visible |
|--------|----------|-----------|------|--------------|
| `module_prep_data` | `compute_and_save_patches()` | `prep_data: extract patches` | patch | patch extraction loop |
| `module_net_train` | `run_train_baseline()` | `net_train: epochs` | epoch | outer epoch loop |
| `module_net_train` | `train_one_epoch()` | `  train` | batch | per-epoch batch loop |
| `module_net_train` | `evaluate_one_epoch()` | `    val` | batch | per-epoch val loop |
| `module_target_predict` | `run_tiled_predict()` | `predict: tiles` | tile | sliding-window tile loop |
| `module_postprocess_vectorize` | `build_postprocess_polygons()` | `postprocess: polygonize instances` | parcel | instance polygonization loop |
| `module_eval` | `run_eval()` | `eval: stages` | stage | stages A-E |
| `module_eval` | `run_pairwise_comparison()` | `compare: stages` | stage | stages A-C |

### Nesting / leave policy

- Batch bars inside the epoch loop use `leave=False` — they clear after each epoch so the
  terminal shows only the epoch bar at the end of training.
- The epoch bar uses `leave=True` — it persists so the final training summary is visible.
- Tile, instance-polygonization, and stage bars use `leave=True` (tiles, epoch) or
  `leave=False` (batch, polygonize, eval stages) depending on whether the final count is
  useful after the step completes.

### Postfix fields

| Bar | Postfix fields |
|-----|---------------|
| epoch | `train=<loss>`, `val=<loss>`, `best=<metric>`, `lr=<lr>` |
| train batch | `loss=<current batch total loss>` |
| val batch | `loss=<current batch total loss>` |
| tile | `processed=<n>`, `skipped=<n_invalid_only>` |
| eval stage | `stage=<label>` |
| compare stage | `stage=<label>` |

---

## Enable / disable

### Default behavior

Progress bars are shown only when `stderr` is a TTY (interactive terminal).
In CI, log files, or when stdout/stderr is redirected, bars are suppressed automatically.

### Environment variable

```bash
AI_FIELDS_PROGRESS=1   # force enable
AI_FIELDS_PROGRESS=0   # force disable
```

Accepted truthy values: `1`, `true`, `yes`, `on`
Accepted falsy values: `0`, `false`, `no`, `off`

### Explicit argument

Every public entry-point that has a progress bar accepts:

```python
progress_enabled: bool | None = None
```

- `True`  → always show bars
- `False` → always suppress bars
- `None`  → auto-detect from TTY / env var (default)

### Propagation chain

```text
run_train_baseline(progress_enabled=...)
  -> train_one_epoch(progress_enabled=...)
  -> evaluate_one_epoch(progress_enabled=...)

run_predict_for_scene(progress_enabled=...)
  -> run_tiled_predict(progress_enabled=...)

run_postprocess_for_scene(progress_enabled=...)
  -> build_postprocess_polygons(progress_enabled=...)

run_eval(progress_enabled=...)
run_pairwise_comparison(progress_enabled=...)
```

The `compute_and_save_patches()` function in `module_prep_data` already accepted
`progress_enabled` before this instrumentation was added (unchanged).

---

## Speed / ETA semantics

tqdm computes rate and ETA from wall-clock time between `update()` calls.

- **Tile bar**: rate is tiles/s.  ETA is approximate because skipped tiles are much
  faster than processed tiles (no model forward pass).
- **Batch bar**: rate is batches/s.
- **Epoch bar**: rate is epochs/s; not useful alone but `best=` postfix shows training
  progress.
- **Stage bars** (eval, compare): each stage has very different cost — ETA is informative
  only for the stage currently in progress, not across stages.

---

## Implementation notes

All bars use the `progress_bar()` context manager (for manual `update()` / `set_postfix()`
calls) or `iter_progress()` wrapper (for automatic per-item updating) from
`ai_fields.common.progress`.  The `_NullProgress` stub is used when progress is disabled
so that `bar.update()` / `bar.set_postfix()` calls in the loop body remain valid without
any conditional branching in calling code.
