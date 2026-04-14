# Руководство по запуску `module_prep_data` (операторский runbook)

## 1. Назначение модуля

`module_prep_data` готовит **train-ready датасет** из сырых геоданных для downstream `module_net_train`.

Это не просто «нарезка патчей». Модуль фиксирует воспроизводимый контракт:
- spatial alignment;
- `valid`/NoData semantics;
- feature mode (`raw8` или `raw8_idx3`);
- target layers (`extent`, `boundary`, `distance`, `valid`);
- split/export + manifests.

## 2. Какие входы подготовить заранее

Обязательно:
- 8-бэндовый GeoTIFF (`.tif`) со сценой;
- вектор полей (`.gpkg`/другой поддерживаемый fiona формат) с `Polygon`/`MultiPolygon`.

Опционально:
- AOI polygon (`.gpkg`/другой поддерживаемый формат).

Что проверить до запуска:
- CRS: растр, вектор полей и AOI должны быть совместимы (в текущем коде ожидается фактическое совпадение CRS по строке после нормализации).
- Band count: строго 8 каналов.
- Band order: для runtime `raw8/raw8_idx3` код ожидает порядок  
  `coastal, blue, green, yellow, red, rededge, nir1, nir2`.
- NoData / valid: должна быть однозначная интерпретация (`has_valid_mask` и/или `nodata`).
- Геометрии: только polygon/multipolygon, `feature_count > 0`.

Важно про AOI:
- если используете AOI, в конфиге `aoi.enabled: true` и в CLI нужен `--aoi ...`;
- для `run_pipeline` желательно иметь sidecar `aoi.gpkg.meta.json` (подробно в разделе 4).

## 3. Рекомендуемая структура папок

Минимальный практичный вариант:

```text
<repo-root>/
  data/
    raw/
      scene_01_8band.tif
    labels/
      fields_01.gpkg
    aoi/
      aoi_01.gpkg                  # опционально
  configs/
    module_prep_data/
      baseline.raw8.yaml
      baseline.raw8_idx3.yaml
      run.scene_01.yaml            # ваш рабочий конфиг
  runs/
    module_prep_data/
      <run_id>/
        01_check_inputs/
        02_prepare_spatial_context/
        03_prepare_features/
        04_prepare_targets/
        05_make_patches/
        06_split_dataset/
        07_validate_outputs/
```

`run_pipeline` пишет в `<output-dir>/<run-id>/...`, поэтому удобно использовать `output-dir = runs/module_prep_data`.

## 4. Подготовка конфига

### 4.1. С чего начать

Скопируйте baseline:

```bash
cp configs/module_prep_data/baseline.raw8.yaml configs/module_prep_data/run.scene_01.yaml
```

или:

```bash
cp configs/module_prep_data/baseline.raw8_idx3.yaml configs/module_prep_data/run.scene_01.yaml
```

### 4.2. Ключи, которые нужно проверить в первую очередь

- `feature_mode`: `raw8` или `raw8_idx3`.
- `valid_policy.nodata_source`: обычно `metadata_then_config`.
- `valid_policy.compute_before_fill`: должно быть `true`.
- `aoi.enabled`, `aoi.aoi_path`, `aoi.buffer_m`.
- `patches.patch_size`: baseline v1 = `512`.
- `patches.halo_px`: обязательный halo для patch-local compute.
- `patches.sampling_policy`: `strategic` или `random`.
- `boundary.encoding`: `background_skeleton_buffer`.
- `distance.target`: `unsigned_distance_to_boundary`.
- `distance.distance_clip_px`: должен удовлетворять `<= patches.halo_px`.
- `normalization.*` (baseline robust percentiles `0.5/99.5`, scale `[0,1]`).
- `split.policy`, `split.random_seed`.

### 4.3. Как выбрать `feature_mode`

- `raw8`: если нужен базовый и самый безопасный первый прогон.
- `raw8_idx3`: если хотите сразу 8 каналов + `NDVI/SAVI/NDWI`.

### 4.4. Как учитывается `valid`

- `valid` сохраняется отдельным слоем (`valid.tif` и patch-level `*_valid.tif`).
- По контракту это также часть assembled input downstream (`raw8_valid`/`raw8_idx3_valid`), но не обязано быть физически склеено в `img`.

### 4.5. AOI в конфиге и на CLI

Если AOI включён:
- `aoi.enabled: true`;
- `aoi.aoi_path` укажите в YAML;
- в запуске обязательно передайте `--aoi`.

Практический нюанс текущего кода:
- для `run_pipeline` AOI metadata обычно должен прийти из sidecar `aoi.gpkg.meta.json`;
- минимально полезные поля sidecar: `crs`, `feature_count`, `geometry_types`, опционально `bounds`.

Пример:

```json
{
  "crs": "EPSG:32637",
  "feature_count": 1,
  "geometry_types": ["Polygon"],
  "bounds": [100.0, 200.0, 400.0, 700.0],
  "readable": true
}
```

### 4.6. Что не нужно «угадывать молча»

- порядок 8 каналов;
- CRS-перепривязки «на глаз»;
- смысл NoData/valid;
- структуру выходного датасета.

Если это неочевидно, фиксируйте явно в данных/sidecar и проверяйте stage manifests.

## 5. Точный порядок запуска стадий (01 → 07)

Ниже реальные команды через текущий entrypoint:
- `python3 -m ai_fields.module_prep_data.run_pipeline`

Подготовьте переменные:

```bash
export CFG="configs/module_prep_data/run.scene_01.yaml"
export RASTER="data/raw/scene_01_8band.tif"
export VECTOR="data/labels/fields_01.gpkg"
export OUT_ROOT="runs/module_prep_data"
export RUN_ID="prep_data_scene_01"
export PATCHES_DIR="$OUT_ROOT/$RUN_ID/05_make_patches/patches"
export DATASET_DIR="$OUT_ROOT/$RUN_ID/06_split_dataset/dataset"
# export AOI="data/aoi/aoi_01.gpkg"   # если используете AOI
```

Базовый префикс:

```bash
BASE="python3 -m ai_fields.module_prep_data.run_pipeline \
  --config \"$CFG\" --raster \"$RASTER\" --vector \"$VECTOR\" \
  --output-dir \"$OUT_ROOT\" --run-id \"$RUN_ID\""
```

Если AOI используется, добавляйте `--aoi "$AOI"` в каждую команду.

### 01_check_inputs

Что делает:
- проверяет входной контракт (пути, band count=8, geometry type/count, CRS, resolvable valid/nodata).

Команда:

```bash
eval "$BASE --start-from-stage 01_check_inputs --stop-after-stage 01_check_inputs --overwrite"
```

Ключевой артефакт:
- `runs/module_prep_data/$RUN_ID/01_check_inputs/check_inputs_manifest.json`

Что проверить:
- `status=success`;
- `checks.band_count_ok=true`;
- `checks.crs_compatible=true`;
- `checks.nodata_interpretation_resolved=true`.

### 02_prepare_spatial_context

Что делает:
- фиксирует spatial context и AOI policy; при runtime compute рассчитывает `effective_extent_bounds`.

Команда:

```bash
eval "$BASE --start-from-stage 02_prepare_spatial_context --stop-after-stage 02_prepare_spatial_context"
```

Ключевой артефакт:
- `.../02_prepare_spatial_context/aoi_manifest.json`

Что проверить:
- `aoi_present`/`aoi_policy_enabled` соответствуют ожиданию;
- `checks.contract_checks_passed=true`.

### 03_prepare_features

Что делает:
- фиксирует feature compute spec и assembled feature contract;
- в baseline режиме не обязан materialize full-scene `img.tif`/`valid.tif`.

Команда:

```bash
eval "$BASE --start-from-stage 03_prepare_features --stop-after-stage 03_prepare_features"
```

Ключевые артефакты:
- `.../03_prepare_features/features_manifest.json`

Что проверить:
- `feature_mode` правильный;
- `feature_channel_count` = 8 (`raw8`) или 11 (`raw8_idx3`);
- `materialization_mode=compute_spec_only` в baseline.

### 04_prepare_targets

Что делает:
- фиксирует target compute spec и target policies;
- в baseline режиме не обязан materialize full-scene target rasters.

Команда:

```bash
eval "$BASE --start-from-stage 04_prepare_targets --stop-after-stage 04_prepare_targets"
```

Ключевые артефакты:
- `.../04_prepare_targets/targets_manifest.json`

Что проверить:
- `boundary_encoding=background_skeleton_buffer`;
- `distance_target=unsigned_distance_to_boundary`;
- `materialization_mode=compute_spec_only` в baseline;
- `status=success`.

### 05_make_patches

Что делает:
- главный materializer train-ready samples из canonical source Stage 02;
- локально считает features/targets на `patch+halo` и экспортирует центральный patch.

Команда:

```bash
eval "$BASE --start-from-stage 05_make_patches --stop-after-stage 05_make_patches"
```

Ключевые артефакты:
- `.../05_make_patches/patches_manifest.json`
- `.../05_make_patches/patches/*.tif`
- `.../05_make_patches/patches/*_meta.json`

Что проверить:
- `written_total > 0` (если `0`, дальше split практически бессмысленен);
- `patch_contract_mode=patch_first_from_source`;
- `patch_runtime_mode=patch_first_from_source`.

### 06_split_dataset

Что делает:
- назначает `train/val/test`;
- материализует train-ready layout;
- считает `norm_stats.json` потоково (bounded-memory, exact fallback для малых наборов).

Команда:

```bash
eval "$BASE --start-from-stage 06_split_dataset --stop-after-stage 06_split_dataset --patches-dir \"$PATCHES_DIR\""
```

Ключевые артефакты:
- `.../06_split_dataset/split_manifest.json`
- `.../06_split_dataset/dataset/train|val|test/...`
- `.../06_split_dataset/dataset/norm_stats.json`

Что проверить:
- `export_layout_materialized=true`;
- в каждом split есть подпапки `img/extent/boundary/distance/valid/meta`;
- `norm_stats.json` существует;
- в `split_manifest.json` заполнен блок `normalization_stats`
  (`method`, `approximation`, `rng_seed`).

### 07_validate_outputs

Что делает:
- проверяет output contract и (в runtime mode) сканирует dataset.

Команда:

```bash
eval "$BASE --start-from-stage 07_validate_outputs --stop-after-stage 07_validate_outputs --dataset-dir \"$DATASET_DIR\""
```

Ключевой артефакт:
- `.../07_validate_outputs/validate_outputs_manifest.json`

Что проверить:
- `status=success`;
- `validation_runtime_executed=true`;
- `checks.contract_checks_passed=true`.

## 6. One-command / end-to-end запуск

В текущем репозитории **есть** единый wrapper-раннер:
- `python3 -m ai_fields.module_prep_data.run_pipeline`

E2E-команда:

```bash
python3 -m ai_fields.module_prep_data.run_pipeline \
  --config "$CFG" \
  --raster "$RASTER" \
  --vector "$VECTOR" \
  --output-dir "$OUT_ROOT" \
  --run-id "$RUN_ID" \
  --runtime-compute-enabled \
  --patches-dir "$PATCHES_DIR" \
  --dataset-dir "$DATASET_DIR" \
  --overwrite
```

Если есть AOI, добавьте `--aoi "$AOI"` и убедитесь, что AOI metadata резолвится (см. раздел 4).

## 7. Ожидаемые выходы после успешного завершения

Ключевой train-ready dataset находится в:
- `runs/module_prep_data/<run_id>/06_split_dataset/dataset/`

Ожидаемая структура:

```text
dataset/
  train/
    img/
    extent/
    boundary/
    distance/
    valid/
    meta/
  val/
    img/
    extent/
    boundary/
    distance/
    valid/
    meta/
  test/
    img/
    extent/
    boundary/
    distance/
    valid/
    meta/
  norm_stats.json
```

Также ожидайте manifests/summaries по стадиям:
- `01_check_inputs/check_inputs_manifest.json`
- `02_prepare_spatial_context/aoi_manifest.json`
- `03_prepare_features/features_manifest.json`
- `04_prepare_targets/targets_manifest.json`
- `05_make_patches/patches_manifest.json`
- `06_split_dataset/split_manifest.json`
- `07_validate_outputs/validate_outputs_manifest.json`
- и `summary.json` внутри каждой stage-папки.

## 8. Минимальная рекомендация для первого боевого прогона

Самый безопасный первый run:
- 1 сцена (`8-band .tif`);
- 1 файл полей (`.gpkg`);
- AOI опционально (если есть, добавляйте сразу корректно);
- `feature_mode: raw8`;
- `patch_size: 512`;
- сначала stage-by-stage (раздел 5), потом уже one-command.

## 9. Частые проблемы и что смотреть

### Неверное число каналов

Симптом:
- stage 01 падает с ошибкой про `band_count`.

Проверка:
- `01_check_inputs/check_inputs_manifest.json` → `checks.band_count_ok`.

### Неочевидный порядок каналов

Симптом:
- run формально успешен, но индексы/качество downstream плохие.

Проверка:
- убедиться, что вход реально в порядке  
  `coastal, blue, green, yellow, red, rededge, nir1, nir2`.

### CRS mismatch

Симптом:
- `SpatialContractError` на stage 01/02.

Проверка:
- манифесты stage 01/02 (`crs`, `checks.crs_compatible`).

### NoData/valid ambiguity

Симптом:
- ошибка `Unable to resolve valid/NoData policy`.

Проверка:
- наличие dataset mask/nodata в растре;
- при необходимости sidecar с `has_valid_mask`/`nodata`.

### Пустые/битые полигоны

Симптом:
- ошибки про `feature_count` или `geometry_types`.

Проверка:
- вектор должен содержать polygon/multipolygon, `feature_count > 0`.

### Не сгенерировались патчи

Симптом:
- stage 05 `written_total = 0`.

Причины:
- слишком маленькая сцена;
- почти всё invalid;
- фильтрация по `valid_ratio`.

Что делать:
- проверить `patches_manifest.json` (`written_total`, `rejection_stats`);
- при первом запуске использовать репрезентативный участок.

### Неполная структура выхода

Симптом:
- stage 07 ругается на missing dirs/layers.

Проверка:
- `06_split_dataset/dataset/{train,val,test}/{img,extent,boundary,distance,valid,meta}`;
- в stage 06 запуске был передан корректный `--patches-dir`;
- в stage 07 запуске был передан корректный `--dataset-dir`.

## 10. Финальный чеклист перед переходом к `module_net_train`

- `07_validate_outputs` завершился со `status=success`.
- `06_split_dataset/dataset/` существует.
- В `train/val/test` есть `img/extent/boundary/distance/valid/meta`.
- `norm_stats.json` существует.
- `feature_mode` в manifests соответствует вашему плану (`raw8` или `raw8_idx3`).
- `written_total > 0`, и в `train` есть реальные sample files.
- Нет `blocking_issues` в stage summaries/manifests.

## Примечание о разночтениях docs ↔ code (на текущий момент)

- В `MANIFEST_SCHEMAS.md` для `module_prep_data` описаны не все stage manifests, которые реально пишет код (`targets_manifest.json`, `validate_outputs_manifest.json` уже есть в реализации).
- В ТЗ упомянуты band order/band mapping как конфигурируемые; в текущей реализации runtime feature compute использует фиксированный порядок каналов, отдельного ключа band mapping в YAML schema нет.
- В `run_pipeline` по умолчанию `--runtime-compute-enabled` включен (`true`), а baseline Stage 03/04 работает как `compute_spec_only`; full-scene materialization включается только через `--diagnostic-full-scene-materialization`.
