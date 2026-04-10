# `module_postprocess_vectorize`: practical runner

Каноничный запуск:

```bash
bash tools/run_module_postprocess_vectorize.sh \
  --predict-run-dir runs/module_target_predict/target-predict-20260408T133539Z \
  --output-dir runs/module_postprocess_vectorize
```

Runner использует Python entrypoint:

- `ai_fields.module_postprocess_vectorize.run_postprocess.run_postprocess_for_scene`

## 1. Режимы входа

Рекомендуемый режим:

- `--predict-run-dir <path>`

Автоматически резолвятся:
- `extent_prob.tif`
- `boundary_prob.tif`
- `distance_pred.tif`
- `valid.tif`
- `predict_manifest.json` (если найден рядом)

Явный режим (если нужно вручную):

```bash
bash tools/run_module_postprocess_vectorize.sh \
  --extent-prob <path/to/extent_prob.tif> \
  --boundary-prob <path/to/boundary_prob.tif> \
  --distance-pred <path/to/distance_pred.tif> \
  --valid <path/to/valid.tif> \
  --output-dir runs/module_postprocess_vectorize
```

Важно: нельзя смешивать `--predict-run-dir` и явные raster paths в одном запуске.

## 2. Основные опции

- `--output-dir <path>`
- `--run-id <id>`
- `--run-dir <path>` (приоритетнее `output-dir/run-id`)
- `--layer-name <name>` (по умолчанию `parcels`)
- `--aoi <path>`
- `--aoi-suppression-enabled | --no-aoi-suppression`
- `--boundary-repair-enabled | --no-boundary-repair`
- `--boundary-repair-closing-radius <int>`
- `--workers <int>` (alias `--num-workers`) для Stage D polygonization
- `--progress-enabled | --no-progress`
- `--dry-run`

Есть и узкие policy-overrides (marker/watershed/polygonization), но для первого baseline запуска лучше оставить дефолты runner’а.

## 3. Dry-run

```bash
bash tools/run_module_postprocess_vectorize.sh \
  --predict-run-dir runs/module_target_predict/target-predict-20260408T133539Z \
  --output-dir runs/module_postprocess_vectorize \
  --dry-run
```

## 4. Что создаётся в run dir

В `<output-dir>/<run-id>`:

- `parcel_instance.tif`
- `parcels.gpkg`
- `postprocess_manifest.json`
- `summary.json`
- `config_used.yaml`

Дополнительно (если включены соответствующие режимы):
- `effective_valid.tif` (AOI suppression)
- `repaired_boundary_prob.tif` (boundary repair)

## 5. Границы ответственности

Этот runner делает только postprocess/vectorization для одного predict-run.
Он не запускает eval, training или predict.
