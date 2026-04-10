# module_prep_data: multi-size export (256 / 384 / 512)

## Что это делает

`module_prep_data` поддерживает:

1. Обычный single-size запуск (один `patch_size` из конфига).
2. Multi-size режим (`--patch-sizes`), где для каждого размера делается отдельный полный run, а train-ready датасеты публикуются отдельно:
   - `.../raw8/256`
   - `.../raw8/384`
   - `.../raw8/512`
   (или для `raw8_idx3` аналогично).

Смешивание размеров внутри одного train/val/test экспорта не допускается.

## Команды

### 1) Single-size (как раньше)

```bash
bash tools/run_module_prep_data.sh \
  --config configs/module_prep_data/baseline.raw8.yaml \
  --raster data/raw/scene.tif \
  --vector data/raw/labels.gpkg \
  --runtime-compute-enabled
```

Итоговый train-ready root:

`runs/module_prep_data/<run-id>/06_split_dataset/dataset`

### 2) Multi-size (256,384,512)

```bash
bash tools/run_module_prep_data.sh \
  --config configs/module_prep_data/baseline.raw8.yaml \
  --raster data/raw/scene.tif \
  --vector data/raw/labels.gpkg \
  --runtime-compute-enabled \
  --patch-sizes 256,384,512 \
  --multi-size-export-root runs/module_prep_data/prep_data_for_train
```

Итоговые опубликованные train-ready roots:

- `runs/module_prep_data/prep_data_for_train/raw8/256`
- `runs/module_prep_data/prep_data_for_train/raw8/384`
- `runs/module_prep_data/prep_data_for_train/raw8/512`

## Что внутри каждого size-specific root

Структура стабильная (как у обычного train-ready датасета):

- `train/img`, `train/extent`, `train/boundary`, `train/distance`, `train/valid`, `train/meta`
- `val/...`
- `test/...`
- `norm_stats.json`

## Контракт размеров

Для каждого отдельного размера:

- все патчи имеют строго `width == patch_size` и `height == patch_size`;
- все слои одного sample имеют одинаковый размер;
- частичные border-патчи не экспортируются;
- Stage 07 проверяет соответствие размеров `patch_size`.

## Артефакты multi-size режима

При `--patch-sizes` дополнительно пишется агрегированный manifest/summary:

- `<output-dir>/<run-id>__multi_size/multi_size_manifest.json`
- `<output-dir>/<run-id>__multi_size/summary.json`

Там фиксируются:

- режим `multi_size`,
- список размеров,
- `feature_mode`,
- per-size run ids,
- per-size source dataset dir и published export root,
- split counts и базовая диагностика.
