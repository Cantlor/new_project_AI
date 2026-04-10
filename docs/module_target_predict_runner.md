# `module_target_predict`: практический runner

Канонический запуск:

```bash
bash tools/run_module_target_predict.sh \
  --checkpoint runs/module_net_train/net-train-20260408T110933Z/best.ckpt \
  --input-raster data/raw/clipped.tif \
  --output-dir runs/module_target_predict
```

## Что обязательно

- `--checkpoint` — путь к `.ckpt`
- `--input-raster` — путь к входному 8-band GeoTIFF

Runner автоматически пытается взять рядом с checkpoint:
- `checkpoint_metadata.json` (обязательно, fail-fast если не найден)
- `train_manifest.json` (опционально)
- `config_used.yaml` (опционально)

При необходимости можно передать явно:

```bash
--checkpoint-metadata <path>
--train-manifest <path>
--config-used <path>
```

По умолчанию runner использует `.venv/bin/python`, если он существует в корне репозитория; иначе `python3`.  
Можно переопределить через `--python-cmd`.

## Полезные runtime overrides

```bash
--run-id <id>
--run-dir <path>
--device-override <cpu|cuda|mps>
--tile-size <int>
--overlap <float>
--normalization-stats <path>
--progress-enabled | --no-progress
```

Проверка без выполнения:

```bash
bash tools/run_module_target_predict.sh \
  --checkpoint runs/module_net_train/net-train-20260408T110933Z/best.ckpt \
  --input-raster data/raw/clipped.tif \
  --output-dir runs/module_target_predict \
  --dry-run
```

## Что создаётся в run dir

По умолчанию run dir:

`<output-dir>/target-predict-<UTC timestamp>`

Артефакты:
- `extent_prob.tif`
- `boundary_prob.tif`
- `distance_pred.tif`
- `valid.tif`
- `predict_manifest.json`
- `summary.json`
- `config_used.yaml`

## Runtime note (память/диск)

Для больших сцен tiled inference теперь использует временные disk-backed буферы (memmap) вместо полного накопления в RAM.  
Это снижает риск OOM, но требует свободного места во временной директории ОС (обычно `/tmp`).

## Важно про контракт

- Predict запускается checkpoint-driven: `feature_mode`, `assembled_model_input`, `in_channels`, нормализация и target-head contract восстанавливаются из train artifacts.
- Поддерживаются baseline-контракты:
  - `raw8_valid` (9 каналов на вход модели)
  - `raw8_idx3_valid` (12 каналов на вход модели)
- Этот runner делает только raster predict. Постпроцессинг/векторизация/оценка в него не входят.

## Ограничение текущей исполняемой реализации

Флаг `--aoi` сейчас не поддержан в рабочем Python path (`run_predict_for_scene` работает в режиме full raster).  
Runner завершится с явной ошибкой, если передать `--aoi`, чтобы не было скрытых fallback.
