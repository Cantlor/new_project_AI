# module_postprocess_vectorize_prompt.md

Ниже готовый промпт для работы над `module_postprocess_vectorize`.

---

Ты работаешь как senior geospatial postprocess engineer внутри проекта **«ИИ для полей»**.

## Контекст

Этот модуль отвечает за превращение georeferenced raster predictions в parcel-level outputs. Он не должен менять смысл upstream model outputs, но обязан воспроизводимо разделять объекты, сохранять provenance и экспортировать финальный векторный результат.

Используй как source of truth:

- `main_tech.md`
- `DATA_CONTRACT.md`
- `MANIFEST_SCHEMAS.md`
- `REPO_CONVENTIONS.md`
- `TESTING_STRATEGY.md`
- `EXPERIMENT_TRACKING.md`
- `DECISIONS.md`
- `.ai/instructions/python.instructions.md`
- `.ai/instructions/gdal_raster.instructions.md`
- `.ai/instructions/postprocess.instructions.md`
- `.ai/instructions/manifests.instructions.md`
- `.ai/instructions/testing.instructions.md`

Если существует отдельное ТЗ для `module_postprocess_vectorize`, считай его приоритетным.

## Твоя задача

Нужно доработать или реализовать `module_postprocess_vectorize` **без ломки принятых границ ответственности**.

Рабочий репозиторий:
`<REPO_ROOT>`

Целевые файлы / директории:
`<TARGET_FILES_OR_DIRS>`

Текущий scope задачи:
`<TASK_SCOPE>`

Что нельзя менять без явного согласования:
`<DO_NOT_CHANGE>`

## Что обязательно сохранить

1. Базовые входы:
   - `extent_prob`
   - `boundary_prob`
   - `distance_pred`
   - `valid`
2. Базовый pipeline:
   - valid/AOI suppression;
   - reproducible thresholding;
   - boundary-aware repair;
   - marker generation;
   - watershed или эквивалентный instance-separation step;
   - `parcel_instance.tif`;
   - polygonization;
   - conservative topology cleanup;
   - export `GPKG`.
3. `polygon_confidence` должен быть поддержан как rule-based QC score.
4. Модуль обязан писать `postprocess_manifest.json` и `summary.json`.
5. Predict и postprocess нельзя смешивать: postprocess не должен превращаться в ещё один inference module.

## Как работать

### Шаг 1. Audit

Сначала разберись с текущим состоянием:
- где читаются raster predictions;
- как проверяется spatial compatibility;
- где применяются thresholds и masks;
- как делается instance separation;
- где происходит polygonization;
- как пишутся manifests.

Отдельно перечисли:
- hidden fallbacks;
- места, где invalid трактуется как обычный фон;
- места, где thresholds/cleanup зашиты неявно;
- места, где provenance может теряться.

### Шаг 2. Plan

Дай safe patch plan:
- какие файлы меняются;
- какие policies фиксируются явно;
- как будут подтверждены downstream outputs и manifests;
- какие tests нужны.

### Шаг 3. Implementation

Требования:
- не делать silent resampling входных raster outputs;
- не менять смысл `extent_prob`/`boundary_prob`/`distance_pred`;
- не делать hidden topology heuristics без фиксации в config/manifest;
- не выпускать vector outputs без `parcel_instance` и provenance;
- не терять AOI/valid policy.

### Шаг 4. Tests

Обязательно покрыть:
- input alignment checks;
- valid suppression;
- threshold policy;
- marker generation logic;
- parcel instance output;
- polygon export completeness;
- manifest completeness.

### Шаг 5. Handoff

В конце дай:
1. изменённые файлы;
2. какие postprocess policies теперь явные;
3. какие риски остались;
4. что проверить на real scene.

## Формат ответа

Ответ дай в таком порядке:

1. `Current postprocess audit`
2. `Contract and provenance risks`
3. `Minimal patch plan`
4. `Applied code changes`
5. `Tests and validation`
6. `Remaining risks`
7. `Handoff`

Если предлагается менять базовый pipeline или contract boundaries, сначала явно опиши это как архитектурное изменение.
