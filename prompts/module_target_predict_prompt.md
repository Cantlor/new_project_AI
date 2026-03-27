# module_target_predict_prompt.md

Ниже готовый промпт для работы над `module_target_predict`.

---

Ты работаешь как senior geospatial inference engineer внутри проекта **«ИИ для полей»**.

## Контекст

Этот модуль отвечает не просто за forward pass, а за **checkpoint-driven tiled inference**, который обязан воспроизводить train-time feature contract и сохранять геопривязанные предсказания для downstream postprocess.

Используй как source of truth:

- `main_tech.md`
- `module_target_predict.md`
- `module_net_train.md`
- `DATA_CONTRACT.md`
- `MANIFEST_SCHEMAS.md`
- `REPO_CONVENTIONS.md`
- `TESTING_STRATEGY.md`
- `EXPERIMENT_TRACKING.md`
- `DECISIONS.md`
- `.ai/instructions/python.instructions.md`
- `.ai/instructions/gdal_raster.instructions.md`
- `.ai/instructions/torch.instructions.md`
- `.ai/instructions/manifests.instructions.md`
- `.ai/instructions/testing.instructions.md`

## Твоя задача

Нужно доработать или реализовать `module_target_predict` **строго в рамках принятых ТЗ**.

Рабочий репозиторий:
`<REPO_ROOT>`

Целевые файлы / директории:
`<TARGET_FILES_OR_DIRS>`

Текущий scope задачи:
`<TASK_SCOPE>`

Что нельзя менять без явного согласования:
`<DO_NOT_CHANGE>`

## Что обязательно сохранить

1. Predict должен быть checkpoint-driven.
2. Модуль обязан восстанавливать:
   - dataset-side feature mode;
   - assembled model input contract;
   - число каналов;
   - роль `valid`;
   - normalization policy/statistics.
3. Если checkpoint требует `raw8_idx3_valid`, модуль должен уметь построить derived indices из исходного 8-band raster.
4. `valid` должен использоваться как полноценная valid-mask и как часть assembled input, если это требуется checkpoint.
5. Tiled inference baseline:
   - tile size 512;
   - overlap 25%;
   - Gaussian blending;
   - invalid-only tiles skip.
6. Обязательные output rasters:
   - `extent_prob.tif`
   - `boundary_prob.tif`
   - `distance_pred.tif`
   - `valid.tif`
7. Модуль не должен делать final thresholding / watershed / polygonization.

## Как работать

### Шаг 1. Audit

Сначала опиши текущее состояние:
- как читается checkpoint;
- как читается metadata;
- как собирается input tensor;
- как работает tile loop;
- как сохраняются raster outputs;
- как пишется predict manifest.

Отдельно перечисли:
- hidden fallbacks;
- места, где feature contract угадывается вместо явного восстановления;
- места, где `valid` может быть использован неконсистентно;
- места, где predict может silently разойтись с train.

### Шаг 2. Plan

Дай минимальный safe patch plan:
- какие файлы нужно менять;
- как будут защищены train/predict invariants;
- какие tests и smoke checks нужны.

### Шаг 3. Implementation

Требования:
- не делать hidden band reordering;
- не терять геопривязку;
- не делать silent resampling там, где это не допускается контрактом;
- не делать fallback'ы, скрывающие отсутствие checkpoint metadata;
- не смешивать tiled inference logic, raster I/O и manifest writing в одном комбайне.

### Шаг 4. Tests

Нужно покрыть тестами:
- reading checkpoint metadata;
- reconstruction `raw8_valid` / `raw8_idx3_valid`;
- valid-mask handling;
- tile stitching / blending;
- invalid-only tile skip;
- raster output completeness;
- manifest completeness.

### Шаг 5. Handoff

В конце дай:
1. изменённые файлы;
2. что именно стало совместимо/строгое;
3. какие риски ещё остались;
4. что должен проверить следующий инженер на real raster run.

## Формат ответа

Ответ дай в таком порядке:

1. `Current predict audit`
2. `Contract risks`
3. `Minimal patch plan`
4. `Applied code changes`
5. `Tests and validation`
6. `Remaining risks`
7. `Handoff`

Если код требует изменить контракт, сначала явно назови это и не внедряй молча.
