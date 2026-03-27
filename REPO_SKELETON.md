# REPO_SKELETON.md

## 1. Назначение документа

Этот документ фиксирует рекомендуемый **каркас репозитория** для проекта **«ИИ для полей»**.

Его задача:

- разложить код, конфиги, тесты, документы и run artifacts по понятным зонам ответственности;
- поддержать уже принятый pipeline:
  `module_prep_data -> module_net_train -> module_target_predict -> module_postprocess_vectorize -> module_eval`;
- сделать разработку удобной как для человека, так и для ИИ-ассистентов;
- не допустить смешения логики модулей, тестов, конфигов и runtime-артефактов.

Этот документ задаёт **recommended baseline structure**, а не требует слепого мгновенного рефактора существующего репозитория.

---

## 2. Главные принципы структуры

### 2.1. Pipeline-first layout
Структура должна отражать реальные модули проекта, а не случайную историю роста кода.

### 2.2. Contract-first layout
Readers, validators, schemas, manifests и configs должны быть такими же видимыми частями проекта, как и модельный код.

### 2.3. Separate code from artifacts
Исходный код, документация, конфиги, тесты и результаты запусков не должны жить вперемешку.

### 2.4. Keep modules explicit
Каждый модуль должен иметь собственную папку и собственные entrypoints.

### 2.5. Runtime outputs must be disposable
Run artifacts должны лежать в предсказуемой зоне, чтобы их можно было чистить, архивировать, сравнивать и переносить без риска повредить исходный код.

---

## 3. Рекомендуемая верхнеуровневая структура

```text
repo/
  README.md
  pyproject.toml
  .gitignore

  docs/
  configs/
  src/
  tests/
  scripts/
  templates/
  prompts/
  .ai/

  data/
  runs/
  reports/
  notebooks/
```

---

## 4. Роль верхнеуровневых папок

### `docs/`
Вся проектная документация.

Пример:

```text
docs/
  README.md
  INDEX.md
  main_tech.md
  GLOSSARY.md
  DATA_CONTRACT.md
  MANIFEST_SCHEMAS.md
  REPO_CONVENTIONS.md
  TESTING_STRATEGY.md
  EXPERIMENT_TRACKING.md
  DECISIONS.md
  IMPLEMENTATION_PLAN.md
  DEVELOPMENT_WORKFLOW.md
```

### `configs/`
Все versioned конфиги.

Пример:

```text
configs/
  prep_data/
  net_train/
  target_predict/
  postprocess_vectorize/
  eval/
  shared/
```

### `src/`
Основной исходный код проекта.

### `tests/`
Тесты проекта.

### `scripts/`
Тонкие пользовательские или сервисные entrypoints.
Скрипты не должны содержать тяжёлую бизнес-логику.

### `templates/`
Шаблоны manifests, summaries, handoff, run reports и config-used артефактов.

### `prompts/`
Повторно используемые промпты для ассистентов.

### `.ai/`
Вспомогательные инструкции для ИИ-ассистентов.

### `data/`
Данные проекта, если хранение внутри репозитория вообще допускается.
Если данные слишком большие, в репозитории оставлять только layout, README и ignore rules.

### `runs/`
Run artifacts модулей.

### `reports/`
Сводные отчёты, comparison-таблицы, ручные forensic summary и итоговые аналитические материалы.

### `notebooks/`
Только для исследований, визуализаций и временного анализа.
Ни один production-critical pipeline не должен жить только в notebook.

---

## 5. Рекомендуемая структура `src/`

```text
src/
  ai_fields/
    common/
    module_prep_data/
    module_net_train/
    module_target_predict/
    module_postprocess_vectorize/
    module_eval/
```

Название корневого Python-пакета может быть другим, но оно должно быть единым и стабильным.

---

## 6. Что должно быть внутри каждого модуля

Базовый паттерн для модуля:

```text
module_x/
  __init__.py
  cli.py
  config.py
  schemas.py
  validators.py
  manifest.py
  summary.py
  io.py
  constants.py
  errors.py
  stages/
  utils/
```

### Обязательные роли файлов

- `cli.py` — парсинг CLI и запуск orchestration.
- `config.py` — загрузка и валидация конфигов.
- `schemas.py` — структуры данных и schema-like описания.
- `validators.py` — проверки входов, контрактов и совместимости.
- `manifest.py` — запись manifest artifacts.
- `summary.py` — запись summary artifacts.
- `io.py` — чтение/запись файлов и адаптеры к форматам.
- `constants.py` — общие константы модуля.
- `errors.py` — типы ошибок и error codes.
- `stages/` — отдельные шаги pipeline.
- `utils/` — небольшие вспомогательные функции без права становиться свалкой.

---

## 7. Специфика по модулям

### 7.1. `module_prep_data/`

Рекомендуемая структура:

```text
module_prep_data/
  cli.py
  config.py
  schemas.py
  validators.py
  manifest.py
  summary.py
  io.py
  constants.py
  errors.py
  stages/
    check_inputs.py
    prepare_aoi.py
    prepare_features.py
    build_targets.py
    extract_patches.py
    split_dataset.py
    validate_export.py
```

### 7.2. `module_net_train/`

```text
module_net_train/
  cli.py
  config.py
  schemas.py
  validators.py
  manifest.py
  summary.py
  dataset.py
  model_factory.py
  losses.py
  metrics.py
  trainer.py
  checkpointing.py
  runtime.py
  stages/
    prepare_run.py
    train.py
    validate.py
    export_artifacts.py
```

### 7.3. `module_target_predict/`

```text
module_target_predict/
  cli.py
  config.py
  schemas.py
  validators.py
  manifest.py
  summary.py
  checkpoint_adapter.py
  feature_builder.py
  tiling.py
  runtime.py
  writer.py
  stages/
    prepare_predict.py
    run_inference.py
    write_outputs.py
```

### 7.4. `module_postprocess_vectorize/`

```text
module_postprocess_vectorize/
  cli.py
  config.py
  schemas.py
  validators.py
  manifest.py
  summary.py
  thresholds.py
  markers.py
  watershed.py
  filtering.py
  vectorize.py
  writer.py
  stages/
    prepare_inputs.py
    build_instances.py
    polygonize.py
    cleanup.py
    export_outputs.py
```

### 7.5. `module_eval/`

```text
module_eval/
  cli.py
  config.py
  schemas.py
  validators.py
  manifest.py
  summary.py
  loaders.py
  pixel_metrics.py
  boundary_metrics.py
  object_metrics.py
  comparison.py
  reporting.py
  stages/
    resolve_sources.py
    run_metrics.py
    compare_runs.py
    export_reports.py
```

---

## 8. Общий код в `common/`

В `common/` выносится только действительно сквозная логика.

Пример:

```text
common/
  paths.py
  logging.py
  ids.py
  manifests.py
  summaries.py
  config_loading.py
  raster_io.py
  vector_io.py
  spatial.py
  runtime.py
  exceptions.py
  typing.py
```

### Что нельзя делать
Нельзя превращать `common/` в свалку всего неудобного.
Если функция специфична для одного модуля, ей место в этом модуле.

---

## 9. Структура `tests/`

Рекомендуемый baseline:

```text
tests/
  unit/
    common/
    module_prep_data/
    module_net_train/
    module_target_predict/
    module_postprocess_vectorize/
    module_eval/

  integration/
    prep_to_train/
    train_to_predict/
    predict_to_postprocess/
    postprocess_to_eval/

  e2e/
    smoke/
    regression/

  fixtures/
  golden/
  helpers/
```

### Роли
- `unit/` — тесты чистых функций и локальных компонентов.
- `integration/` — тесты стыков модулей.
- `e2e/` — короткие и полные сценарии.
- `fixtures/` — маленькие тестовые входы.
- `golden/` — эталонные ожидаемые outputs.
- `helpers/` — тестовые утилиты.

---

## 10. Структура `configs/`

```text
configs/
  shared/
    runtime.yaml
    logging.yaml
    manifests.yaml

  prep_data/
    prep_data.raw8.yaml
    prep_data.raw8_idx3.yaml

  net_train/
    train.raw8_valid.yaml
    train.raw8_idx3_valid.yaml

  target_predict/
    predict.default.yaml

  postprocess_vectorize/
    postprocess.default.yaml

  eval/
    eval.default.yaml
```

### Правила
- конфиг должен отражать модуль и вариант запуска;
- скрытые overrides из кода не допускаются;
- фактически использованный конфиг должен сериализоваться в `config_used.yaml`.

---

## 11. Структура `runs/`

Рекомендуемая схема:

```text
runs/
  module_prep_data/
    <run_id>/
      config_used.yaml
      manifests/
      summaries/
      outputs/
      logs/

  module_net_train/
    <run_id>/
      config_used.yaml
      manifests/
      summaries/
      checkpoints/
      metrics/
      logs/

  module_target_predict/
    <run_id>/
      config_used.yaml
      manifests/
      summaries/
      outputs/
      previews/
      logs/

  module_postprocess_vectorize/
    <run_id>/
      config_used.yaml
      manifests/
      summaries/
      outputs/
      logs/

  module_eval/
    <run_id>/
      config_used.yaml
      manifests/
      summaries/
      reports/
      tables/
      logs/
```

Это помогает совместить reproducibility, forensic review и чистую навигацию по run artifacts.

---

## 12. Структура `reports/`

```text
reports/
  comparisons/
  audits/
  run_summaries/
  handoffs/
```

Здесь хранятся уже человекоориентированные отчёты, а не первичные runtime outputs.

---

## 13. Структура `.ai/`

```text
.ai/
  instructions/
    README.md
    python.instructions.md
    gdal_raster.instructions.md
    torch.instructions.md
    testing.instructions.md
    manifests.instructions.md
    postprocess.instructions.md
```

Эта папка не должна подменять проектную документацию, а только усиливать её при работе с ассистентами.

---

## 14. Что должно лежать в `scripts/`

В `scripts/` должны жить только:

- тонкие entrypoints;
- локальные сервисные обёртки;
- dev-tools и maintenance helpers;
- reproducible helper-команды.

### Чего там быть не должно
- ядра обучения;
- единственной реализации pipeline stage;
- логики, которую нельзя переиспользовать из Python API.

---

## 15. Что делать с notebook-файлами

Notebook допустимы для:

- exploratory analysis;
- визуализации;
- sanity checks;
- отчётных иллюстраций.

Notebook не должен быть единственным источником:

- preprocessing logic;
- evaluation logic;
- postprocess logic;
- train/inference pipeline.

Если идея родилась в notebook, потом она должна быть перенесена в versioned code.

---

## 16. Минимальный bootstrap-набор для старта кода

Если начинать реализацию с нуля, первым делом в репозитории должны появиться:

```text
README.md
pyproject.toml
docs/
configs/
src/
tests/
templates/
prompts/
.ai/
```

И только после этого:

- модульные пакеты;
- CLI entrypoints;
- validators;
- manifest writers;
- datasets/models/predictors.

---

## 17. Что можно отложить на потом

Не обязательно делать сразу:

- сложную monorepo automation;
- генераторы docs-site;
- CI matrix на все платформы;
- packaging для публичного PyPI;
- полностью идеальную структуру data storage.

Сначала нужна ясная скелетная архитектура, а не максимальная инфраструктурная сложность.

---

## 18. Что запрещено этой структурой

Запрещённые анти-паттерны:

- смешивать manifests, outputs и код в одной папке;
- хранить production logic только в `scripts/`;
- держать критичную pipeline-логику только в notebook;
- складывать модульно-специфичный код в `common/`;
- делать неявные runtime side effects вне run directories;
- писать конфиги, которые реально используются, но не сохраняются в `config_used`;
- делать новые папки и naming conventions без отражения в `REPO_CONVENTIONS.md` и смежных docs.

---

## 19. Практический переход без жёсткого big-bang рефактора

Если репозиторий уже существует и живой, переход к этой структуре лучше делать поэтапно:

1. Зафиксировать документы и contracts.
2. Ввести предсказуемую папку `runs/`.
3. Нормализовать `configs/`.
4. Выделить `src/<package>/module_*`.
5. Перенести manifest/summary logic в явные модули.
6. После этого уже резать legacy scripts и дубли.

Это безопаснее, чем пытаться переписать всё сразу.

---

## 20. Роль документа

`REPO_SKELETON.md` нужен, чтобы:

- строить repo layout без хаоса;
- одинаково понимать, где должен жить новый код;
- не путать продуктовые артефакты с инженерными;
- облегчить handoff, ревью и работу ассистентов;
- соотнести реальную структуру репозитория с уже принятыми ТЗ, контрактами и workflow.
