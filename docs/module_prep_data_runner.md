# Краткое использование `tools/run_module_prep_data.sh`

`tools/run_module_prep_data.sh` — каноничный shell-runner для полного или частичного запуска `module_prep_data` через реальный entrypoint:

- `python3 -m ai_fields.module_prep_data.run_pipeline`

## 1. Полный запуск

```bash
bash tools/run_module_prep_data.sh \
  --config configs/module_prep_data/baseline.raw8.yaml \
  --raster data/raw/scene_01_8band.tif \
  --vector data/labels/fields_01.gpkg \
  --output-dir runs/module_prep_data \
  --run-id prep_data_scene_01 \
  --runtime-compute-enabled \
  --overwrite
```

## 2. Запуск с конкретной стадии

Пример: продолжить с `05_make_patches` до конца.

```bash
bash tools/run_module_prep_data.sh \
  --config configs/module_prep_data/baseline.raw8.yaml \
  --raster data/raw/scene_01_8band.tif \
  --vector data/labels/fields_01.gpkg \
  --output-dir runs/module_prep_data \
  --run-id prep_data_scene_01 \
  --from-stage 05_make_patches \
  --to-stage 07_validate_outputs \
  --runtime-compute-enabled
```

## 3. Где менять дефолтные пути

Откройте `tools/run_module_prep_data.sh` и измените переменные в блоке вверху файла:

- `PYTHON_CMD`
- `CONFIG_PATH`
- `RASTER_PATH`
- `VECTOR_PATH`
- `AOI_PATH`
- `OUTPUT_DIR`
- `RUN_ID`

Также там можно менять дефолты флагов (`RUNTIME_COMPUTE_ENABLED`, `RUNTIME_PROBE_ENABLED` и т.д.).

## 4. Пример `raw8`

```bash
bash tools/run_module_prep_data.sh \
  --config configs/module_prep_data/baseline.raw8.yaml \
  --raster data/raw/scene_01_8band.tif \
  --vector data/labels/fields_01.gpkg \
  --runtime-compute-enabled
```

## 5. Пример `raw8_idx3`

```bash
bash tools/run_module_prep_data.sh \
  --config configs/module_prep_data/baseline.raw8_idx3.yaml \
  --raster data/raw/scene_01_8band.tif \
  --vector data/labels/fields_01.gpkg \
  --runtime-compute-enabled
```

## 6. Multi-size режим (256/384/512)

```bash
bash tools/run_module_prep_data.sh \
  --config configs/module_prep_data/baseline.raw8.yaml \
  --raster data/raw/scene_01_8band.tif \
  --vector data/labels/fields_01.gpkg \
  --runtime-compute-enabled \
  --patch-sizes 256,384,512 \
  --multi-size-export-root runs/module_prep_data/prep_data_for_train
```

Подробно: [docs/module_prep_data_multi_size.md](/home/cantlor/uzcosmos/new_proj/docs/module_prep_data_multi_size.md)

## Примечание по текущему коду

В `run_pipeline` по умолчанию `--runtime-compute-enabled` выключен, поэтому без явного включения стадии 02..07 работают как metadata/contract-driven и не материализуют полный train-ready export.
