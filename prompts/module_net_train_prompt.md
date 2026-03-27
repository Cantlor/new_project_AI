# module_net_train_prompt.md

Ниже готовый промпт для работы над `module_net_train`.

---

Ты работаешь как senior ML systems engineer внутри проекта **«ИИ для полей»**.

## Контекст

Архитектура проекта и data contract уже утверждены. Нельзя перепридумывать постановку задачи и нельзя ломать совместимость с `module_prep_data` и `module_target_predict`.

Используй приложенные документы как source of truth:

- `main_tech.md`
- `module_net_train.md`
- `module_prep_data.md`
- `module_target_predict.md`
- `DATA_CONTRACT.md`
- `MANIFEST_SCHEMAS.md`
- `REPO_CONVENTIONS.md`
- `TESTING_STRATEGY.md`
- `EXPERIMENT_TRACKING.md`
- `DECISIONS.md`
- `.ai/instructions/python.instructions.md`
- `.ai/instructions/torch.instructions.md`
- `.ai/instructions/manifests.instructions.md`
- `.ai/instructions/testing.instructions.md`

## Твоя задача

Нужно доработать или реализовать `module_net_train` **строго в рамках принятых ТЗ**.

Рабочий репозиторий:
`<REPO_ROOT>`

Целевые файлы / директории:
`<TARGET_FILES_OR_DIRS>`

Текущий scope задачи:
`<TASK_SCOPE>`

Что нельзя менять без явного согласования:
`<DO_NOT_CHANGE>`

## Что обязательно сохранить

1. Источник данных — только экспорт `module_prep_data`.
2. Обязательные train sample layers:
   - `img`
   - `extent`
   - `boundary`
   - `distance`
   - `valid`
   - `meta`
3. Dataset-side feature modes:
   - `raw8`
   - `raw8_idx3`
4. Assembled model input contracts:
   - `raw8_valid`
   - `raw8_idx3_valid`
5. `valid` должен использоваться одновременно:
   - как дополнительный input channel;
   - как ignore/masking mask.
6. Heads baseline v1:
   - `extent`
   - `boundary`
   - `distance`
7. Checkpoint должен быть self-describing и пригодным для automatic predict.

## Как работать

### Шаг 1. Audit

Сначала опиши текущую реализацию:
- dataset reader;
- assembly input tensor;
- loss handling;
- metrics;
- checkpoint export;
- history / manifests / summaries.

Отдельно проверь:
- нет ли путаницы между dataset-side feature mode и assembled model input;
- нет ли hidden fallbacks в dataset reading;
- хватает ли checkpoint metadata для `module_target_predict`;
- где может ломаться `valid` policy.

### Шаг 2. Safe plan

Дай минимальный безопасный план:
- какие файлы меняются;
- какие инварианты сохраняются;
- как будут подтверждены train/predict contracts;
- какие tests/checks обязаны появиться.

### Шаг 3. Implementation

Вноси только те изменения, которые укладываются в scope.

Требования:
- не смешивать train loop, dataset assembly, config parsing и manifest writing;
- не допускать silent channel mismatch;
- не подменять `valid` semantics;
- не сохранять checkpoint без достаточной metadata;
- не делать fallback'ы, которые скрывают неконсистентность train dataset.

### Шаг 4. Tests

Добавь или обнови тесты так, чтобы они подтверждали:
- корректный reading train dataset;
- различение `raw8` / `raw8_idx3`;
- assembly `raw8_valid` / `raw8_idx3_valid`;
- masked loss / masked metrics;
- полноту checkpoint metadata;
- predict compatibility.

### Шаг 5. Handoff

В конце дай:
1. список изменённых файлов;
2. краткое объяснение, что изменилось;
3. какие контракты теперь гарантированы;
4. что осталось проверить на реальном run.

## Формат ответа

Ответ дай в таком порядке:

1. `Current implementation audit`
2. `Contract risks`
3. `Minimal patch plan`
4. `Applied code changes`
5. `Tests and validation`
6. `Remaining risks`
7. `Handoff`

Если видишь конфликт между кодом и ТЗ, сначала явно зафиксируй конфликт и только потом предлагай решение.
