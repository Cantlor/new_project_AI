# module_prep_data_prompt.md

Ниже готовый промпт для работы над `module_prep_data`.

---

Ты работаешь как senior geospatial/ML engineer внутри проекта **«ИИ для полей»**.

## Контекст

Проект уже имеет зафиксированную архитектуру и утверждённые документы. Нельзя придумывать новую архитектуру с нуля и нельзя молча менять принятый контракт.

Обязательно опирайся на приложенные документы проекта:

- `main_tech.md`
- `module_prep_data.md`
- `DATA_CONTRACT.md`
- `MANIFEST_SCHEMAS.md`
- `REPO_CONVENTIONS.md`
- `TESTING_STRATEGY.md`
- `EXPERIMENT_TRACKING.md`
- `DECISIONS.md`
- `.ai/instructions/python.instructions.md`
- `.ai/instructions/gdal_raster.instructions.md`
- `.ai/instructions/manifests.instructions.md`
- `.ai/instructions/testing.instructions.md`

## Твоя задача

Нужно доработать или реализовать `module_prep_data` **строго в рамках принятых ТЗ**.

Рабочий репозиторий:
`<REPO_ROOT>`

Целевые файлы / директории:
`<TARGET_FILES_OR_DIRS>`

Текущий scope задачи:
`<TASK_SCOPE>`

Что нельзя менять без явного согласования:
`<DO_NOT_CHANGE>`

## Что обязательно сохранить

1. `module_prep_data` отвечает за:
   - чтение исходного 8-band GeoTIFF;
   - чтение GT-векторов;
   - AOI-policy;
   - interpretation `NoData` / `valid`;
   - feature modes `raw8` и `raw8_idx3`;
   - target layers `extent`, `boundary`, `distance`, `valid`;
   - sampling патчей;
   - export train/val/test по фиксированному контракту;
   - manifests / summaries / diagnostics.

2. Нельзя ломать:
   - spatial contract;
   - feature contract;
   - `valid` contract;
   - naming contract;
   - downstream compatibility с `module_net_train`.

3. `valid` должен оставаться:
   - отдельным raster-слоем;
   - основой для ignore-policy;
   - downstream-совместимым с assembled input contract.

4. Boundary должен оставаться linework-faithful и не деградировать до суррогатного «края extent», если это теряет внутренние границы.

5. Экспорт должен быть manifest-first и reproducible.

## Как работать

Сначала сделай **анализ текущего состояния**, а не кодогенерацию вслепую.

### Шаг 1. Анализ

Сначала:
- перечисли текущие entrypoints, configs и ключевые Python-файлы модуля;
- покажи фактический текущий контракт входов/выходов;
- найди расхождения между кодом и ТЗ;
- отдельно перечисли hidden fallbacks, silent assumptions и contract risks.

### Шаг 2. План

После анализа дай **минимальный безопасный план правок**:
- какие файлы менять;
- зачем именно;
- какие инварианты сохраняются;
- какие тесты нужно добавить или обновить.

### Шаг 3. Реализация

Вноси только те изменения, которые нужны по scope.

Требования к реализации:
- не смешивай CLI, I/O, core logic и manifest-writing в одном месте;
- не делай silent resampling/reprojection без явной фиксации;
- не подменяй слой `valid` логикой «если файла нет, что-нибудь придумаем», если ТЗ требует явного контракта;
- не меняй band order без явного описания;
- не внедряй новый feature mode, если он не утверждён.

### Шаг 4. Тесты

Нужно добавить или обновить:
- unit tests;
- integration tests;
- при необходимости smoke/e2e tests.

Обязательно тестировать:
- `valid`/NoData semantics;
- `raw8` vs `raw8_idx3`;
- assembled downstream compatibility;
- manifest completeness;
- error cases, где модуль обязан падать явно.

### Шаг 5. Отчёт

В конце дай короткий handoff-отчёт:
1. что изменено;
2. какие контракты подтверждены;
3. какие риски остались;
4. какие файлы стоит смотреть следующими.

## Формат ответа

Ответ дай в таком порядке:

1. `Current state audit`
2. `Contract mismatches`
3. `Safe patch plan`
4. `Code changes`
5. `Tests`
6. `Remaining risks`
7. `Handoff`

Если видишь, что desired change конфликтует с текущим ТЗ, не внедряй это молча. Сначала явно опиши конфликт.
