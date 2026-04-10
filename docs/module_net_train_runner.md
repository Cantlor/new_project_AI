# Краткое использование `tools/run_module_net_train.sh`

`tools/run_module_net_train.sh` — каноничный shell-runner для `module_net_train`.
Он запускает реальный текущий entrypoint кода:

- `ai_fields.module_net_train.run_train.run_train_baseline(...)`

Важно: входом должен быть **train-ready export** из `module_prep_data` (split layout), а не сырые GeoTIFF/векторные данные.

## 1. Полный запуск (самый практичный режим)

Если есть готовый `module_prep_data` run (`.../06_split_dataset/...`), передайте только config + prep run dir:

```bash
bash tools/run_module_net_train.sh \
  --config configs/module_net_train/baseline.raw8.yaml \
  --prep-run-dir runs/module_prep_data/prep-data-20260407T064457Z \
  --patch-size 512 \
  --runs-root runs/module_net_train \
  --run-id net-train-raw8-scene01
```

Скрипт автоматически возьмёт:

- dataset root: `<prep-run-dir>/06_split_dataset/dataset`
- source manifest: `<prep-run-dir>/06_split_dataset/split_manifest.json`
- source run id: basename `prep-run-dir`

## 2. Запуск от конкретного шага

В текущем коде у `module_net_train` один операторский шаг: `train`.
Поэтому `--from-step/--to-step` поддерживают только `train`:

```bash
bash tools/run_module_net_train.sh \
  --config configs/module_net_train/baseline.raw8.yaml \
  --prep-run-dir runs/module_prep_data/prep-data-20260407T064457Z \
  --from-step train \
  --to-step train
```

## 3. Где менять дефолтные пути

Откройте `tools/run_module_net_train.sh` и измените блок переменных вверху файла:

- `PYTHON_CMD`
- `CONFIG_PATH`
- `PREP_RUN_DIR`
- `DATASET_ROOT`
- `DATASET_SOURCE_MANIFEST`
- `RUNS_ROOT`
- `RUN_ID`

## 4. Пример `raw8`

```bash
bash tools/run_module_net_train.sh \
  --config configs/module_net_train/baseline.raw8.yaml \
  --prep-run-dir runs/module_prep_data/prep-data-20260407T064457Z \
  --run-id net-train-raw8
```

## 5. Пример `raw8_idx3`

```bash
bash tools/run_module_net_train.sh \
  --config configs/module_net_train/baseline.raw8_idx3.yaml \
  --prep-run-dir runs/module_prep_data/prep-data-20260407T064457Z \
  --run-id net-train-raw8-idx3
```

## 6. Полезные override-параметры

- `--dataset-root`, `--train-split-dir`, `--val-split-dir` — если нужно вручную задать split paths.
- `--dataset-source-manifest` и `--dataset-source-run-id` — явный provenance override.
- `--run-dir` — полный путь run output (если не хотите `runs-root + run-id`).
- `--normalization-stats-source` — явное указание `stats_source` для train export metadata.
- `--patch-size` — строгая проверка `patch_size` выбранного dataset (`256|384|512`).

## 7. Multi-size usage

Отдельный практический гайд для запуска независимых run-ов по size-specific датасетам:

- [docs/module_net_train_multi_size_usage.md](/home/cantlor/uzcosmos/new_proj/docs/module_net_train_multi_size_usage.md)
