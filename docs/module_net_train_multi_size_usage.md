# `module_net_train`: запуск по отдельным patch-size датасетам

Этот режим поддерживает **несколько отдельных training run-ов** для разных размеров патча (`256` / `384` / `512`).

Важно:
- один training run использует **ровно один** выбранный dataset root;
- смешивание patch-size внутри одного run не поддерживается;
- вход должен быть train-ready export из `module_prep_data` (`06_split_dataset/dataset`), а не сырые данные.

Каноничный runner: `tools/run_module_net_train.sh`.

## 1. Режим через `--prep-run-dir` (рекомендуется)

```bash
bash tools/run_module_net_train.sh \
  --config configs/module_net_train/baseline.raw8.yaml \
  --prep-run-dir runs/module_prep_data/prep-raw8-256-ps256 \
  --patch-size 256 \
  --run-id net-train-raw8-ps256
```

```bash
bash tools/run_module_net_train.sh \
  --config configs/module_net_train/baseline.raw8.yaml \
  --prep-run-dir runs/module_prep_data/prep-raw8-384-ps384 \
  --patch-size 384 \
  --run-id net-train-raw8-ps384
```

```bash
bash tools/run_module_net_train.sh \
  --config configs/module_net_train/baseline.raw8.yaml \
  --prep-run-dir runs/module_prep_data/prep-raw8-512-ps512 \
  --patch-size 512 \
  --run-id net-train-raw8-ps512
```

Runner автоматически берёт:
- dataset root: `<prep-run-dir>/06_split_dataset/dataset`
- split manifest: `<prep-run-dir>/06_split_dataset/split_manifest.json`
- source run id: basename `prep-run-dir`

## 2. Режим через явный `--dataset-root`

```bash
bash tools/run_module_net_train.sh \
  --config configs/module_net_train/baseline.raw8_idx3.yaml \
  --dataset-root prep_data_for_train/raw8_idx3/384 \
  --dataset-source-manifest runs/module_prep_data/prep-raw8-idx3-384-ps384/06_split_dataset/split_manifest.json \
  --patch-size 384 \
  --run-id net-train-raw8-idx3-ps384
```

## 3. Политика выбора датасета (без двусмысленности)

Нужно выбрать **один** источник:
- либо `--prep-run-dir`,
- либо `--dataset-root` (+ `--dataset-source-manifest`).

Если передать оба режима одновременно, runner завершится с явной ошибкой.

## 4. Что валидируется до старта обучения

Перед train-run выполняется контрактная проверка выбранного dataset:
- наличие `train/` и `val/`;
- наличие обязательных слоёв `img/extent/boundary/distance/valid/meta`;
- согласованность spatial shape по всем слоям sample;
- фиксированный patch size внутри выбранного dataset;
- channel/feature-mode согласованность с train config.

## 5. Что фиксируется в train artifacts

В `train_manifest.json` / `summary.json` фиксируются:
- `dataset_source_run_id`
- `dataset_source_manifest_path`
- `dataset_root`
- `patch_size`
- `dataset_feature_mode`

Это делает provenance прозрачным для отдельных run-ов на `256` / `384` / `512`.
