# module_eval: практический runner

Каноничный запуск `module_eval`:

```bash
bash tools/run_module_eval.sh \
  --prep-run-dir runs/module_prep_data/prep-raw8-256-ps256 \
  --predict-run-dir runs/module_target_predict/target-predict-20260408T133539Z \
  --postprocess-run-dir runs/module_postprocess_vectorize/postprocess-vectorize-20260409T092257Z \
  --output-dir runs/module_eval
```

## Что делает runner
- Использует реальный Python entrypoint: `ai_fields.module_eval.run_eval.run_eval`.
- Склеивает Stage A->E в одном запуске.
- В `run-dir` режиме автоматически резолвит входы:
  - из `prep-run-dir`: `extent.tif`, `boundary.tif`, `valid.tif`, `distance.tif` (если есть), `vector_in_raster_crs.gpkg`;
  - из `predict-run-dir`: `extent_prob.tif`, `boundary_prob.tif`, `distance_pred.tif`, `valid.tif`, `predict_manifest.json` (если есть);
  - из `postprocess-run-dir`: `parcel_instance.tif` (если есть), `parcels.gpkg`, `postprocess_manifest.json` (если есть).

## Минимальные обязательные входы
Для текущего исполняемого `run_eval` обязательны:
- GT raster: `extent`, `boundary`, `valid`;
- GT polygons (`--gt-parcels` или auto из `prep-run-dir`);
- predict raster: `extent_prob`, `boundary_prob`, `distance_pred`, `valid`;
- postprocess polygons (`--post-parcels-gpkg` или auto из `postprocess-run-dir`).

Если чего-то не хватает, runner падает fail-fast с явной ошибкой.

## Полезные опции
- `--run-id <id>` и `--output-dir <path>`
- `--run-dir <path>` (переопределяет `output-dir/run-id`)
- `--dry-run`
- `--progress-enabled` / `--no-progress`
- `--extent-threshold <float>` и threshold provenance overrides
- `--bucket-enabled` / `--no-bucket`

## Dry-run

```bash
bash tools/run_module_eval.sh \
  --prep-run-dir runs/module_prep_data/prep-raw8-256-ps256 \
  --predict-run-dir runs/module_target_predict/target-predict-20260408T133539Z \
  --postprocess-run-dir runs/module_postprocess_vectorize/postprocess-vectorize-20260409T092257Z \
  --dry-run
```

## Какие артефакты появляются
В `runs/module_eval/<run-id>/` текущий путь пишет как минимум:
- `eval_manifest.json`
- `summary.json`
- `config_used.yaml`
- `error_taxonomy.json`
- `metrics_aggregate.json`
- `scenes_included.json`
- `scenes_excluded.json`
- `source_runs.json`
- `visual_diagnostics.*` (если доступны зависимости визуализации)

Примечание: runner запускает single-scene eval orchestration. Pairwise comparison запускается отдельным `run_compare` path и в этот runner не включён.
