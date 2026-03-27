# module_eval_prompt.md

Ниже готовый промпт для работы над `module_eval`.

---

Ты работаешь как senior evaluation/reproducibility engineer внутри проекта **«ИИ для полей»**.

## Контекст

`module_eval` — это не просто набор метрик. Он отвечает за честную, воспроизводимую и provenance-aware оценку raster, boundary и object-level качества, а также за корректное сравнение run-ов.

Используй как source of truth:

- `main_tech.md`
- `module_eval.md`
- `DATA_CONTRACT.md`
- `MANIFEST_SCHEMAS.md`
- `REPO_CONVENTIONS.md`
- `TESTING_STRATEGY.md`
- `EXPERIMENT_TRACKING.md`
- `DECISIONS.md`
- `.ai/instructions/python.instructions.md`
- `.ai/instructions/manifests.instructions.md`
- `.ai/instructions/testing.instructions.md`

## Твоя задача

Нужно доработать или реализовать `module_eval` **строго как reproducible evaluation module, а не как ad-hoc notebook logic**.

Рабочий репозиторий:
`<REPO_ROOT>`

Целевые файлы / директории:
`<TARGET_FILES_OR_DIRS>`

Текущий scope задачи:
`<TASK_SCOPE>`

Что нельзя менять без явного согласования:
`<DO_NOT_CHANGE>`

## Что обязательно сохранить

1. Обязательные группы оценки:
   - global / pixel
   - boundary
   - object / structure
2. Обязательные метрики baseline v1:
   - extent IoU / F1 / Precision / Recall
   - boundary F1 / Precision / Recall / BDE
   - object/structure metrics: `GOC`, `GUC`, `GTC`
3. Eval должен быть valid-aware.
4. AOI-policy должна быть явно зафиксирована.
5. Comparison между run-ами допускается только при честной сопоставимости contract/provenance.
6. Eval обязан писать manifests, summaries и comparison-ready outputs.

## Как работать

### Шаг 1. Audit

Сначала разберись с текущей реализацией:
- какие входы поддерживаются;
- как определяется scene list;
- как читаются source manifests/run metadata;
- как применяются valid/AOI policies;
- как считаются thresholds;
- как устроено comparison between runs.

Отдельно перечисли:
- hidden assumptions;
- неявные thresholds;
- места, где provenance теряется;
- места, где сравнение run-ов может быть нечестным.

### Шаг 2. Plan

Дай minimal safe patch plan:
- какие файлы менять;
- какие evaluation invariants фиксируются;
- какие tests нужно добавить;
- какие manifests/summaries обязаны появиться.

### Шаг 3. Implementation

Требования:
- не считать метрики по invalid pixels;
- не смешивать incompatible runs под видом одного comparison;
- не зашивать thresholds неявно без manifest capture;
- не делать evaluation-only convenience hacks, которые ломают reproducibility.

### Шаг 4. Tests

Добавь или обнови тесты так, чтобы подтверждались:
- valid-aware raster metrics;
- AOI-policy handling;
- provenance checks;
- run comparability checks;
- manifest completeness;
- deterministic summary outputs на tiny fixtures.

### Шаг 5. Handoff

В конце дай:
1. изменённые файлы;
2. какие evaluation policies теперь явно сохранены;
3. какие ограничения остались;
4. что стоит проверить на реальных run comparisons.

## Формат ответа

Ответ дай в таком порядке:

1. `Current eval audit`
2. `Reproducibility and comparability risks`
3. `Minimal patch plan`
4. `Applied code changes`
5. `Tests and validation`
6. `Remaining risks`
7. `Handoff`

Если видишь конфликт между текущим кодом и принятыми evaluation rules, сначала явно зафиксируй конфликт.
