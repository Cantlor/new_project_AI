# Run Report

## 1. Общая информация

- **Module:** `module_name`
- **Run ID:** `YYYYMMDD_HHMMSS_example`
- **Status:** `success | partial | failed`
- **Data contract version:** `v1`
- **Created at (UTC):** `2026-03-27T00:00:00Z`

## 2. Цель запуска

Кратко описать, зачем запускался этот run.

## 3. Входы

- Основной вход: `path/to/input`
- Дополнительные входы: `...`
- Upstream run IDs: `...`
- Upstream manifests: `...`

## 4. Разрешённый контракт

### Spatial
- Reference CRS: `...`
- Resolution: `...`
- Alignment policy: `...`
- AOI used: `true/false`

### Features
- Dataset feature mode: `raw8 | raw8_idx3 | ...`
- Assembled model input: `raw8_valid | raw8_idx3_valid | ...`
- Feature channel count: `...`
- Final input channel count: `...`
- Channel semantics: `...`
- Valid as input channel: `true/false`

### Valid / NoData
- Valid source: `...`
- Valid representation: `...`
- Invalid handling: `...`
- Nodata policy: `...`

### Normalization
- Name: `...`
- Stats source: `...`
- Clip percentiles: `...`
- Scaling range: `...`

## 5. Runtime

- Device requested: `...`
- Device resolved: `...`
- AMP requested: `...`
- AMP used: `...`
- OOM fallbacks applied: `...`

## 6. Основные выходы

- `path/to/output_1`
- `path/to/output_2`
- `path/to/manifest.json`
- `path/to/summary.json`

## 7. Результат

### Ключевые показатели
- Metric / count / artifact summary: `...`

### Наблюдения
- `...`
- `...`

## 8. Проблемы и риски

- `...`
- `...`

## 9. Что проверить дальше

- `...`
- `...`

## 10. Связанные артефакты

- Config used: `path/to/config_used.yaml`
- Manifest: `path/to/manifest.json`
- Summary: `path/to/summary.json`
- Logs: `path/to/logs`
