# MANIFEST_SCHEMAS.md

## 1. Назначение документа

Этот документ фиксирует схемы manifest- и summary-артефактов проекта ИИ для полей.

Его задача:

- задать единые правила для manifests всех модулей;
- обеспечить воспроизводимость run-ов;
- зафиксировать provenance;
- исключить скрытые догадки между `module_prep_data`, `module_net_train`, `module_target_predict`, `module_postprocess_vectorize`, `module_eval`;
- обеспечить возможность forensic-сравнения run-ов и восстановления полного data/runtime contract.

Этот документ не заменяет `DATA_CONTRACT.md`.
`DATA_CONTRACT.md` отвечает на вопрос, что значат данные,
а `MANIFEST_SCHEMAS.md` отвечает на вопрос, как документируется конкретный запуск и его артефакты.

## 2. Общие принципы

### 2.1. Manifest-first policy

Каждый модуль обязан выпускать manifest-артефакты как часть baseline design.
Минимальный обязательный набор run-артефактов проекта:

- `config_used.*`
- `manifest.*`
- `summary.*`
- `diagnostics / logs`
- `versioned outputs`

### 2.2. Manifest must be sufficient

Manifest считается корректным, если по нему можно восстановить:

- что было прочитано;
- что было записано;
- какой контракт данных был использован;
- какие runtime-решения реально применились;
- какие policies действовали для `valid`, `AOI`, normalization, thresholds и comparison rules;
- из каких upstream run-ов произошли входные артефакты.

### 2.3. Запрет на скрытую магию

Manifest не должен быть декоративным.
Если критичное решение реально влияло на результат, оно должно быть либо явно записано в manifest, либо считаться потерянным provenance. Это особенно важно для:

- feature contract;
- assembled model input contract;
- valid semantics;
- normalization stats source;
- threshold provenance;
- AOI policy;
- runtime fallback-решений.

## 3. Общая структура любого manifest

Любой manifest проекта должен использовать общий верхнеуровневый каркас.

### 3.1. Обязательные верхнеуровневые поля

```yaml
schema_name: string
schema_version: string
module_name: string
module_version: string | null
data_contract_version: string
run_id: string
stage_name: string | null
created_at_utc: string
status: string
```

### 3.2. Семантика

- `schema_name` — каноническое имя схемы, например `prep_data.features_manifest`.
- `schema_version` — версия именно схемы manifest, не всего проекта.
- `module_name` — имя модуля, например `module_prep_data`.
- `module_version` — версия реализации модуля, если уже ведётся.
- `data_contract_version` — версия сквозного data contract, например `v1`.
- `run_id` — уникальный идентификатор запуска.
- `stage_name` — имя стадии, если manifest stage-specific.
- `created_at_utc` — время записи manifest в UTC.
- `status` — `success / partial / failed`.

### 3.3. Обязательные секции

```yaml
config:
  config_used_path: string
  config_hash: string | null
  config_overrides: object | null

provenance:
  source_run_ids: []
  source_manifest_paths: []
  source_config_paths: []
  code_version: string | null
  git_commit: string | null

inputs:
  artifacts: []

outputs:
  artifacts: []

resolved_contract:
  spatial: object
  features: object
  valid_policy: object
  normalization: object | null
  aoi_policy: object | null

runtime:
  device_requested: string | null
  device_resolved: string | null
  amp_requested: bool | null
  amp_used: bool | null
  oom_fallbacks_applied: []
  notes: []

diagnostics:
  warnings: []
  errors: []
```

Это согласуется с уже зафиксированной общей политикой run-артефактов и runtime-sensitive metadata.

## 4. Универсальная схема описания артефакта

Каждый элемент `inputs.artifacts[]` и `outputs.artifacts[]` должен использовать общую структуру:

```yaml
path: string
role: string
format: string
is_required: bool
exists: bool
checksum: string | null
size_bytes: int | null
crs: string | null
transform: string | null
width: int | null
height: int | null
count: int | null
dtype: string | null
nodata: number | string | null
channel_semantics: [] | null
notes: []
```

### 4.1. Что важно

- Для raster-артефактов желательно фиксировать `crs`, `transform`, `width`, `height`, `count`, `dtype`, `nodata`.
- Для vector-артефактов обязательно фиксировать `format`, `crs`, `role`.
- Для model/checkpoint artifacts обязательно фиксировать `role`, `checksum`, и ссылку на metadata sidecar, если он хранится отдельно.
- Для manifest/config artifacts `channel_semantics` обычно `null`.

## 5. Общая секция resolved_contract

Это одна из самых важных частей.

### 5.1. resolved_contract.spatial

```yaml
spatial:
  reference_grid_source: string
  reference_crs: string
  reference_resolution: number | null
  alignment_policy: string
  aoi_used: bool
  aoi_buffer_m: number | null
```

### 5.2. resolved_contract.features

```yaml
features:
  dataset_feature_mode: string | null
  assembled_model_input: string | null
  feature_channel_count: int | null
  final_input_channel_count: int | null
  channel_semantics: []
  valid_as_input_channel: bool | null
```

### 5.3. resolved_contract.valid_policy

```yaml
valid_policy:
  valid_source: string
  valid_representation: string
  invalid_handling: string
  nodata_policy: string
```

### 5.4. resolved_contract.normalization

```yaml
normalization:
  dtype_before_model: string | null
  normalization_name: string | null
  stats_source: string | null
  clip_percentiles: [number, number] | null
  scaling_range: [number, number] | null
```

### 5.5. resolved_contract.aoi_policy

```yaml
aoi_policy:
  aoi_present: bool
  aoi_role: string | null
  buffer_m: number | null
  output_extent_mode: string | null
```

Эта секция нужна потому, что проект различает dataset-side feature mode и финальный assembled model input, а `valid` имеет двойную роль: mask + input channel. Это уже жёстко зафиксировано в ТЗ и должно быть восстановимо из manifests без ручной догадки.

## 6. Summary vs Manifest

### 6.1. Manifest

Manifest хранит полный воспроизводимый протокол запуска.

### 6.2. Summary

Summary хранит сжатое техническое резюме результата, пригодное для быстрого чтения человеком и comparison tooling.

### 6.3. Правило

- `Summary` может быть неполным.
- `Manifest` — нет.

## 7. Схемы module_prep_data

`module_prep_data` обязан сохранять manifests, summaries и диагностику, а его pipeline организован как последовательность стадий. В exported baseline явно перечислены:

- `check_inputs_manifest.json`
- `aoi_manifest.json`
- `features_manifest.json`
- `patches_manifest.json`
- `split_manifest.json`
- `summary.json`

### 7.1. check_inputs_manifest.json

#### Назначение

Фиксирует результат входной проверки исходных данных.

#### Обязательные поля сверх общего каркаса

```yaml
schema_name: prep_data.check_inputs_manifest

input_raster:
  path: string
  crs: string | null
  width: int
  height: int
  count: int
  dtype: string
  nodata: number | string | null

input_vectors:
  path: string
  crs: string | null
  feature_count: int | null
  geometry_types: []

input_aoi:
  path: string | null
  crs: string | null
  feature_count: int | null

checks:
  raster_readable: bool
  vector_readable: bool
  aoi_readable: bool | null
  crs_compatible: bool
  band_count_ok: bool
  geometry_validity_ok: bool | null
  nodata_interpretation_resolved: bool
  blocking_issues: []
```

### 7.2. aoi_manifest.json

#### Назначение

Фиксирует, использовался ли `AOI`, как он был приведён и какая `buffer-policy` применялась.

```yaml
schema_name: prep_data.aoi_manifest

aoi_present: bool
aoi_source_path: string | null
aoi_source_crs: string | null
aoi_target_crs: string | null
aoi_reprojected: bool
buffer_m: number | null
effective_extent_bounds: [number, number, number, number] | null
```

Это должно отражать общесистемную `AOI-policy` и baseline buffer 30 м, если именно он был применён.

### 7.3. features_manifest.json

#### Назначение

Фиксирует собранный feature stack и metadata, нужную downstream-модулям.

```yaml
schema_name: prep_data.features_manifest

feature_mode: string
feature_channel_count: int
channel_semantics: []
derived_indices: []
valid_saved_separately: bool
assembled_model_input_variants: []
normalization_plan:
  normalization_name: string
  dtype_before_model: string
  clip_percentiles: [number, number] | null
  scaling_range: [number, number] | null
```

#### Обязательные значения v1

- `feature_mode`: `raw8` или `raw8_idx3`
- `derived_indices` для `raw8_idx3`: `NDVI`, `SAVI`, `NDWI`
- `assembled_model_input_variants`: `raw8_valid` или `raw8_idx3_valid` в зависимости от режима.

### 7.4. patches_manifest.json

#### Назначение

Фиксирует sampling патчей и статистику отбраковок.

```yaml
schema_name: prep_data.patches_manifest

patch_size: int
sampling_policy: string
written_total: int
written_center: int | null
written_boundary: int | null
written_negative: int | null
shortfall_negative: int | null

rejection_stats:
  invalid_ratio_rejects: int | null
  mask_ratio_rejects: int | null
  boundary_quality_rejects: int | null
  duplicate_or_overlap_rejects: int | null

patch_exports:
  img_count: int
  extent_count: int
  boundary_count: int
  distance_count: int
  valid_count: int
  meta_count: int
```

### 7.5. split_manifest.json

#### Назначение

Фиксирует split `train/val/test` и финальный экспорт `train-ready` датасета.

```yaml
schema_name: prep_data.split_manifest

split_policy: string
random_seed: int | null
feature_mode: string
feature_channel_count: int
channel_semantics: []
splits:
  train:
    sample_count: int
  val:
    sample_count: int
  test:
    sample_count: int
export_structure:
  required_dirs:
    - img
    - extent
    - boundary
    - distance
    - valid
    - meta
```

### 7.6. summary.json

#### Назначение

Краткое техническое резюме `module_prep_data` run.

```yaml
schema_name: prep_data.summary

status: string
feature_mode: string
patch_size: int
written_total: int
split_counts: object
warnings: []
key_notes: []
```

## 8. Схемы module_net_train

`module_net_train` обязан сохранять checkpoints, manifests, историю обучения и metadata, достаточную для автоматического использования в `module_target_predict`.

### 8.1. train_manifest.json

```yaml
schema_name: net_train.train_manifest

dataset_source_run_id: string
dataset_source_manifest_path: string
feature_mode: string
assembled_model_input: string
feature_channel_count: int
final_input_channel_count: int
channel_semantics: []
valid_as_input_channel: bool

model:
  architecture_name: string
  encoder_name: string | null
  heads:
    - extent
    - boundary
    - distance

loss:
  extent_loss_name: string
  boundary_loss_name: string
  distance_loss_name: string
  loss_weights:
    extent: number
    boundary: number
    distance: number

optimizer:
  name: string
  lr: number | null

scheduler:
  name: string | null

training:
  batch_size: int | null
  epochs_completed: int
  amp_used: bool
  best_checkpoint_metric: string
```

### 8.2. checkpoint_metadata.json

Можно хранить внутри checkpoint sidecar или как отдельный JSON, но его схема должна быть явной.

```yaml
schema_name: net_train.checkpoint_metadata

checkpoint_path: string
feature_mode: string
assembled_model_input: string
in_channels: int
channel_semantics: []
valid_as_input_channel: bool
normalization:
  normalization_name: string
  stats_source: string
  clip_percentiles: [number, number]
  scaling_range: [number, number]
target_heads:
  extent: object
  boundary: object
  distance: object
model_version: string | null
```

Это критично, потому что `module_target_predict` строится как `checkpoint-driven inference` и не должен требовать ручных догадок о числе каналов, роли `valid` и порядке сборки тензора.

### 8.3. summary.json

```yaml
schema_name: net_train.summary

status: string
feature_mode: string
assembled_model_input: string
best_metric_name: string
best_metric_value: number | null
best_checkpoint_path: string | null
epochs_completed: int
warnings: []
```

## 9. Схемы module_target_predict

`module_target_predict` обязан сохранять полный `predict manifest`, геопривязанные raster outputs и достаточно диагностики для forensic-проверки.

### 9.1. predict_manifest.json

```yaml
schema_name: target_predict.predict_manifest

checkpoint:
  checkpoint_path: string
  checkpoint_metadata_path: string | null
  train_run_id: string | null
  train_manifest_path: string | null

input_raster:
  path: string
  crs: string
  width: int
  height: int
  count: int
  dtype: string
  nodata: number | string | null

resolved_contract:
  features:
    dataset_feature_mode: string
    assembled_model_input: string
    feature_channel_count: int
    final_input_channel_count: int
    channel_semantics: []
    valid_as_input_channel: bool
  normalization:
    normalization_name: string
    stats_source: string
    clip_percentiles: [number, number]
    scaling_range: [number, number]
  valid_policy:
    valid_source: string
    invalid_handling: string
  aoi_policy:
    aoi_present: bool
    buffer_m: number | null
    output_extent_mode: string

tiling:
  tile_size: int
  overlap_fraction: number
  blending: string
  invalid_only_tiles_skipped: int | null
  processed_tiles: int | null

outputs_expected:
  - extent_prob
  - boundary_prob
  - distance_pred
  - valid
```

### 9.2. summary.json

```yaml
schema_name: target_predict.summary

status: string
input_raster_path: string
feature_mode: string
assembled_model_input: string
output_paths:
  extent_prob: string | null
  boundary_prob: string | null
  distance_pred: string | null
  valid: string | null
warnings: []
key_notes: []
```

Это напрямую соответствует зафиксированному baseline: `checkpoint-driven inference`, `raw8/raw8_idx3` adapter, assembled input `raw8_valid / raw8_idx3_valid`, tiled inference, Gaussian blending, output set `extent_prob`, `boundary_prob`, `distance_pred`, `valid`.

## 10. Схемы module_postprocess_vectorize

Хотя отдельный файл ТЗ этого модуля здесь не приложен, принятый baseline уже зафиксирован: входы `extent_prob`, `boundary_prob`, `distance_pred`, `valid`; обязательные outputs `parcel_instance.tif`, `parcels.gpkg`, `postprocess_manifest.json`, `summary.json`, `config_used.yaml`; default format — `GPKG`.

### 10.1. postprocess_manifest.json

```yaml
schema_name: postprocess_vectorize.postprocess_manifest

source_predict_run_id: string | null
source_predict_manifest_path: string | null

inputs:
  extent_prob_path: string
  boundary_prob_path: string
  distance_pred_path: string
  valid_path: string
  aoi_path: string | null

resolved_policy:
  valid_suppression: string
  aoi_policy: object | null
  threshold_policy: object
  boundary_repair_policy: string
  marker_generation_policy: string
  watershed_policy: string
  filtering_policy: string
  topology_cleanup_policy: string
  export_format: string

outputs:
  parcel_instance_path: string | null
  parcels_gpkg_path: string | null
  optional_exports: []
```

### 10.2. summary.json

```yaml
schema_name: postprocess_vectorize.summary

status: string
parcel_count: int | null
export_format: string
warnings: []
key_notes: []
```

## 11. Схемы module_eval

`module_eval` обязан сохранять provenance, source run identifiers, thresholds, scene list, valid/AOI policy и comparison sources. Все параметры eval должны сохраняться в manifest для полной воспроизводимости.

### 11.1. eval_manifest.json

```yaml
schema_name: eval.eval_manifest

eval_mode: string
source_run_ids: []
source_manifest_paths: []
gt_sources: []
scene_selection:
  policy: string
  resolved_scene_list: []
  excluded_scenes: []
valid_aoi_policy:
  valid_policy: string
  aoi_policy: string | null

thresholds:
  raster_binarization: object | null
  vector_matching: object | null
  threshold_provenance: string

metrics_enabled:
  pixel: []
  boundary: []
  object_structure: []

comparison:
  enabled: bool
  comparison_pairs: [] | null

runtime:
  device_requested: string | null
  device_resolved: string | null
  parallel_mode: string | null
  scene_fallback_mode: string | null
```

### 11.2. summary.json

```yaml
schema_name: eval.summary

status: string
eval_mode: string
scene_count: int
metric_summary: object
ranking_summary: object | null
warnings: []
```

## 12. Правила именования schema_name

Рекомендуемый канонический формат:

```text
<module_short_name>.<artifact_name>
```

Примеры:

- `prep_data.check_inputs_manifest`
- `prep_data.features_manifest`
- `net_train.train_manifest`
- `net_train.checkpoint_metadata`
- `target_predict.predict_manifest`
- `postprocess_vectorize.postprocess_manifest`
- `eval.eval_manifest`

## 13. Версионирование схем

### 13.1. Общее правило

У каждой схемы есть собственная версия:

```yaml
schema_version: "v1"
```

### 13.2. Когда повышать версию

Версия схемы должна меняться, если меняется:

- обязательный набор полей;
- смысл поля;
- тип поля;
- `required/optional` статус;
- правила трактовки provenance;
- `resolved contract` structure.

### 13.3. Совместимость

- Изменение только значения поля без изменения схемы — это не новая версия схемы.
- Изменение структуры — новая версия схемы.

## 14. Обязательные поля для forensic-ready run

Для любого важного production-like run обязательно должны быть восстановимы:

- `run_id`
- `data_contract_version`
- `config_used_path`
- `source_run_ids`
- `source_manifest_paths`
- `dataset_feature_mode`
- `assembled_model_input`
- `channel_semantics`
- `valid_as_input_channel`
- `valid_policy`
- `normalization.stats_source`
- `device_resolved`
- `amp_used`
- `oom_fallbacks_applied`
- `warnings`
- `errors`

Именно этот набор нужен, чтобы потом честно разбирать, почему два run-а отличаются не только по метрикам, но и по реальному контракту исполнения.

## 15. Что запрещено

Запрещено выпускать manifests, в которых:

- неясно происхождение входных артефактов;
- потерян `source run id`;
- не зафиксирован `feature_mode`;
- не зафиксирован `assembled model input`;
- не зафиксирована роль `valid`;
- не зафиксирован источник `normalization stats`;
- не зафиксированы thresholds или valid/AOI evaluation policy там, где они влияют на сравнение;
- скрыты runtime fallback-решения, которые могли поменять поведение.

## 16. Минимальный набор файлов по модулям

### module_prep_data

- `check_inputs_manifest.json`
- `aoi_manifest.json`
- `features_manifest.json`
- `patches_manifest.json`
- `split_manifest.json`
- `summary.json`

### module_net_train

- `train_manifest.json`
- `checkpoint_metadata.json`
- `summary.json`

### module_target_predict

- `predict_manifest.json`
- `summary.json`

### module_postprocess_vectorize

- `postprocess_manifest.json`
- `summary.json`

### module_eval

- `eval_manifest.json`
- `summary.json`

## 17. Роль документа в кодинге

Этот файл должен использоваться как опора для:

- `JSON/YAML schema validators`;
- `manifest writers`;
- `summary writers`;
- `run registries`;
- `comparison tooling`;
- `unit/integration tests` на воспроизводимость и provenance.

Новый код не должен писать manifests «как получится».
Он должен писать их по зафиксированной схеме.
