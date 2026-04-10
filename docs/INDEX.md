# DOCS INDEX

## Назначение

Этот файл — быстрая карта документации проекта **«ИИ для полей»**.
Он помогает понять:

- какие документы уже являются source of truth;
- какие документы задают правила разработки;
- какие файлы использовать при запуске новой задачи, ревью, handoff и реализации кода.

---

## 1. Базовые проектные документы

### `main_tech.md`
Главный обзор проекта и архитектурная рамка v1.
Использовать как стартовую точку для понимания pipeline, границ MVP и базовых проектных инвариантов.

### `module_prep_data.md`
ТЗ модуля подготовки данных.
Определяет стадии подготовки, feature modes, target generation, export contract и stage artifacts.

### `module_net_train.md`
ТЗ модуля обучения.
Определяет train contract, assembled model input, baseline architecture, losses, checkpoints и train outputs.

### `module_target_predict.md`
ТЗ модуля инференса.
Определяет checkpoint-driven predict, восстановление feature contract, tiled inference и predict outputs.

### `module_postprocess_vectorize.md`
ТЗ модуля постпроцессинга и векторизации.
Определяет baseline pipeline: valid/AOI suppression, thresholding, boundary repair, marker generation, watershed, polygonization, conservative topology cleanup и экспорт parcels.gpkg.

### `module_eval.md`
ТЗ модуля оценки.
Определяет группы метрик, provenance, multi-run comparison и обязательные eval artifacts,
включая baseline single-pair comparison artifacts.

---

## 2. Документы сквозного контракта

### `GLOSSARY.md`
Единый словарь терминов проекта.
Использовать всегда, когда есть риск неоднозначной терминологии.

### `DATA_CONTRACT.md`
Сквозной контракт данных проекта.
Главный документ для:
- readers/writers;
- validators;
- dataset assembly;
- predict compatibility;
- postprocess compatibility;
- eval consistency.

### `MANIFEST_SCHEMAS.md`
Схемы manifests и summaries.
Главный документ для manifest writers, run provenance, experiment tracking и forensic-ready runs.
Также фиксирует schema contract для comparison artifacts `module_eval`.

---

## 3. Документы инженерной организации

### `REPO_CONVENTIONS.md`
Правила структуры репозитория, именования, конфигов, CLI, ошибок и размещения артефактов.

### `TESTING_STRATEGY.md`
Стратегия тестирования: unit, integration, smoke e2e, golden tests, regression checks.

### `EXPERIMENT_TRACKING.md`
Правила ведения run-ов, сравнения экспериментов, именования и хранения результатов.

### `DECISIONS.md`
Реестр архитектурных решений.
Использовать, когда нужно понять не только **что** принято, но и **почему**.

### `REPO_SKELETON.md`
Рекомендуемый каркас репозитория: структура верхнего уровня, внутренняя декомпозиция модулей, layout для `src/`, `configs/`, `runs/`, `tests/`, `.ai/`.

### `IMPLEMENTATION_PLAN.md`
Пошаговый порядок реализации проекта после фиксации ТЗ и контрактов.

### `DEVELOPMENT_WORKFLOW.md`
Короткий рабочий стандарт: как брать задачу в работу, как резать на change sets, как валидировать и как оформлять handoff.

---

## 4. Документы и шаблоны для практической разработки

### `templates/`
Папка с шаблонами:
- `manifest.template.json`
- `summary.template.json`
- `config_used.template.yaml`
- `run_report.template.md`
- `handoff.template.md`

Использовать как основу для новых модулей и run artifacts.

### `.ai/instructions/`
Папка с инструкциями для ИИ-ассистентов:
- `python.instructions.md`
- `gdal_raster.instructions.md`
- `torch.instructions.md`
- `testing.instructions.md`
- `manifests.instructions.md`
- `postprocess.instructions.md`

Использовать как guardrails при генерации кода и ревью.

### `prompts/`
Папка с шаблонами промптов:
- для модулей;
- для code review;
- для hidden fallback audit.

Использовать для повторяемой постановки задач ассистентам.

---

## 5. В каком порядке читать документы

### Если начинаешь работать над проектом с нуля
1. `main_tech.md`
2. `GLOSSARY.md`
3. `DATA_CONTRACT.md`
4. `MANIFEST_SCHEMAS.md`
5. ТЗ нужного модуля
6. `REPO_CONVENTIONS.md`
7. `TESTING_STRATEGY.md`
8. `IMPLEMENTATION_PLAN.md`
9. `DEVELOPMENT_WORKFLOW.md`

### Если берёшь конкретную задачу в модуле
1. ТЗ нужного модуля
2. `DATA_CONTRACT.md`
3. `MANIFEST_SCHEMAS.md`
4. `REPO_CONVENTIONS.md`
5. `TESTING_STRATEGY.md`
6. `DEVELOPMENT_WORKFLOW.md`
7. `DECISIONS.md` при необходимости

### Если делаешь ревью или forensic-анализ
1. `DATA_CONTRACT.md`
2. `MANIFEST_SCHEMAS.md`
3. `EXPERIMENT_TRACKING.md`
4. `DECISIONS.md`
5. `TESTING_STRATEGY.md`
6. manifests / summaries / handoff текущего run

---

## 6. Практическое правило при конфликте документов

Если документы кажутся противоречивыми, использовать следующий приоритет:

1. зафиксированные модульные ТЗ и `main_tech.md`;
2. `DATA_CONTRACT.md`;
3. `MANIFEST_SCHEMAS.md`;
4. `DECISIONS.md`;
5. `REPO_CONVENTIONS.md` и остальные operational docs;
6. README, handoff, prompts и вспомогательные заметки.

Если конфликт нельзя разрешить по этому приоритету, изменение должно быть явно зафиксировано в `DECISIONS.md`, а затем отражено в остальных документах.

---

## 7. Когда обновлять этот файл

`docs/INDEX.md` нужно обновлять, если:

- появился новый обязательный проектный документ;
- изменилась роль существующего документа;
- добавился новый модульный ТЗ;
- изменилась структура папок `templates/`, `.ai/instructions/`, `prompts/` или `docs/`;
- проект перешёл к новой major версии архитектурного freeze.
