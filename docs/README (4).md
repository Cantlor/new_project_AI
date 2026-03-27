# Документация проекта «ИИ для полей»

## Назначение

Этот файл — карта проектной документации.  
Его задача — быстро показать:

- какие документы являются основными;
- какие документы фиксируют контракт и правила реализации;
- какие документы используются во время кодинга;
- в каком порядке их читать новому разработчику или ИИ-ассистенту.

Документация проекта разделяется на четыре слоя:

1. **архитектурная основа** — что за система строится и как устроен pipeline;
2. **контрактный слой** — какие данные, manifests и соглашения считаются обязательными;
3. **операционный слой** — как вести разработку, тесты, эксперименты и handoff;
4. **LLM/dev support layer** — инструкции, шаблоны и промпты, которые ускоряют разработку, но не заменяют ТЗ.

---

## 1. Основные документы проекта

### `main_tech.md`
Главный архитектурный документ проекта.

Содержит:
- общий pipeline;
- целевой baseline v1;
- ключевые проектные инварианты;
- feature contract;
- hardware-adaptive policy;
- связь между модулями.

Читать первым.

### `module_prep_data.md`
ТЗ модуля подготовки данных.

Содержит:
- входной и выходной контракт модуля;
- стадии подготовки данных;
- feature modes;
- valid / NoData policy;
- patching, split, manifests, export structure.

### `module_net_train.md`
ТЗ модуля обучения.

Содержит:
- train dataset contract;
- assembled model input;
- baseline architecture;
- losses, metrics, checkpoint metadata;
- требования к train-time reproducibility.

### `module_target_predict.md`
ТЗ модуля предикта.

Содержит:
- checkpoint-driven inference;
- восстановление feature contract;
- predict-time normalization;
- tiled inference;
- output artifacts и predict manifest.

### `module_postprocess_vectorize.md`
ТЗ модуля постпроцессинга и векторизации.

Содержит:
- baseline postprocess pipeline;
- valid/AOI suppression;
- validation-calibrated thresholding;
- hybrid boundary repair и marker-controlled watershed;
- polygonization и conservative topology cleanup;
- export в GPKG, polygon_confidence, postprocess manifest.

### `module_eval.md`
ТЗ модуля оценки.

Содержит:
- группы метрик;
- provenance requirements;
- fair comparison rules;
- eval artifacts и comparison policy.

### `GLOSSARY.md`
Единый словарь терминов проекта.

Нужен, чтобы все документы, manifests, README и код использовали одни и те же значения терминов.

---

## 2. Контрактные документы

### `DATA_CONTRACT.md`
Сквозной контракт данных между всеми модулями.

Отвечает на вопросы:
- какие данные существуют в системе;
- как называются canonical layers;
- что такое `raw8`, `raw8_idx3`, `raw8_valid`, `raw8_idx3_valid`;
- как трактуется `valid`;
- какие артефакты должны быть совместимыми между модулями.

Это главный документ для validators, readers, writers и compatibility logic.

### `MANIFEST_SCHEMAS.md`
Схемы manifest- и summary-артефактов.

Отвечает на вопросы:
- какие manifests обязательны;
- какие у них минимальные поля;
- как фиксировать provenance;
- как документировать runtime decisions;
- как описывать inputs, outputs и resolved contract.

Это главный документ для manifest writers, run registries и comparison tooling.

### `REPO_CONVENTIONS.md`
Соглашения по репозиторию.

Содержит:
- naming rules;
- layout conventions;
- правила для конфигов, логов и manifests;
- требования к CLI, ошибкам и runtime behavior;
- правила для code organization.

---

## 3. Операционные документы

### `TESTING_STRATEGY.md`
Стратегия тестирования проекта.

Содержит:
- unit / integration / smoke e2e / golden tests;
- приоритеты покрытия;
- обязательные проверки для каждого модуля;
- правила приёмки изменений.

### `EXPERIMENT_TRACKING.md`
Правила ведения экспериментов и сравнения run-ов.

Содержит:
- run naming;
- обязательные артефакты экспериментов;
- как фиксировать baseline;
- когда два run-а считаются честно сопоставимыми;
- как хранить результаты и сравнения.

### `DECISIONS.md`
Реестр архитектурных решений.

Нужен для фиксации:
- почему были приняты текущие baseline-решения;
- что уже frozen;
- что временно;
- какие вопросы ещё открыты.

### `REPO_SKELETON.md`
Рекомендуемый каркас репозитория.

Содержит:
- top-level структуру папок;
- внутреннюю декомпозицию каждого модуля;
- layout `src/`, `configs/`, `runs/`, `tests/`, `reports/`, `.ai/`;
- рекомендации по gradual-переходу без big-bang рефактора.

### `DEVELOPMENT_WORKFLOW.md`
Рабочий стандарт разработки.

Содержит:
- жизненный цикл задачи (intake → handoff);
- pre-change checklist;
- правила формирования change set;
- когда обновлять manifests, документацию и DECISIONS.md;
- правила AI-assisted разработки.

### `IMPLEMENTATION_PLAN.md`
Пошаговый план реализации проекта.

Содержит:
- порядок разработки модулей;
- что писать сначала;
- какие проверки делать до перехода дальше;
- когда модуль считается готовым к следующему этапу.

### `templates/`
Папка с шаблонами служебных файлов:
- `manifest.template.json`
- `summary.template.json`
- `config_used.template.yaml`
- `run_report.template.md`
- `handoff.template.md`

Нужна для ускорения рутины и унификации артефактов.

---

## 4. Документы для AI-assisted разработки

### `.ai/instructions/`
Набор прикладных инструкций для ассистентов и code agents.

Сейчас включают:
- `python.instructions.md`
- `gdal_raster.instructions.md`
- `torch.instructions.md`
- `testing.instructions.md`
- `manifests.instructions.md`
- `postprocess.instructions.md`

Эти документы **не заменяют** ТЗ.  
Они уточняют, **как писать код**, оставаясь внутри уже принятых контрактов проекта.

### `prompts/`
Папка с готовыми промптами под задачи разработки и ревью.

Содержит:
- промпты по модулям;
- промпт для поиска hidden fallbacks;
- README по использованию.

Используется как слой ускорения, а не как источник истины.

---

## 5. Рекомендуемый порядок чтения

### Для нового человека в проекте
1. `main_tech.md`
2. `GLOSSARY.md`
3. `DATA_CONTRACT.md`
4. `MANIFEST_SCHEMAS.md`
5. `REPO_CONVENTIONS.md`
6. нужное ТЗ модуля
7. `TESTING_STRATEGY.md`
8. `EXPERIMENT_TRACKING.md`
9. `DECISIONS.md`
10. `IMPLEMENTATION_PLAN.md`
11. `DEVELOPMENT_WORKFLOW.md`
12. `REPO_SKELETON.md`

### Для разработчика, который начинает писать код конкретного модуля
1. `DATA_CONTRACT.md`
2. `MANIFEST_SCHEMAS.md`
3. ТЗ нужного модуля
4. `REPO_CONVENTIONS.md`
5. `DEVELOPMENT_WORKFLOW.md`
6. `TESTING_STRATEGY.md`
7. профильные `.ai/instructions/*`
8. нужный шаблон из `prompts/`

### Для человека, который делает ревью
1. ТЗ модуля
2. `DATA_CONTRACT.md`
3. `MANIFEST_SCHEMAS.md`
4. `REPO_CONVENTIONS.md`
5. `TESTING_STRATEGY.md`
6. `DECISIONS.md`
7. `review_hidden_fallbacks_prompt.md`

---

## 6. Что считается источником истины

При конфликте между документами приоритет такой:

1. **Архитектурно принятые ТЗ и main_tech**
2. **DATA_CONTRACT.md**
3. **MANIFEST_SCHEMAS.md**
4. **DECISIONS.md**
5. **REPO_CONVENTIONS.md**
6. **TESTING_STRATEGY.md / EXPERIMENT_TRACKING.md**
7. **.ai/instructions/**
8. **prompts/**
9. **templates/**

Шаблоны, инструкции и промпты не могут отменять ТЗ и контрактные документы.

---

## 7. Как поддерживать документацию в актуальном состоянии

При каждом значимом изменении нужно проверить:

- не изменился ли `data_contract_version`;
- не изменилась ли схема manifest;
- не появился ли новый canonical term;
- не изменилась ли логика `valid`, `feature_mode`, `assembled_model_input`;
- не изменились ли обязательные run artifacts;
- не появилось ли новое архитектурное решение, которое нужно зафиксировать в `DECISIONS.md`.

Если меняется реализация, но не меняется смысл — обычно достаточно обновить:
- `REPO_CONVENTIONS.md`,
- `TESTING_STRATEGY.md`,
- `EXPERIMENT_TRACKING.md`,
- шаблоны и AI-instructions.

Если меняется смысл контракта — сначала обновляются:
- ТЗ / `main_tech.md`,
- `DATA_CONTRACT.md`,
- `MANIFEST_SCHEMAS.md`,
- `DECISIONS.md`,
и только потом код.

---

## 8. Практическое правило

Для этого проекта документация — не “дополнение к коду”, а часть системы.  
Если поведение модуля нельзя восстановить по документации, config_used, manifest и summary, значит проект теряет воспроизводимость.

Поэтому любой новый код должен появляться **не раньше**, чем понятны:
- его место в pipeline;
- его data contract;
- его manifest contract;
- его критерии тестирования;
- его место в общем implementation plan.
