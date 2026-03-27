# REPO_CONVENTIONS.md

## 1. Назначение документа

Этот документ фиксирует **репозиторные и инженерные соглашения** проекта **«ИИ для полей»**.

Его задача:

- задать единые правила организации кода и артефактов;
- снизить риск дрейфа между модулями pipeline;
- сделать разработку, ревью, тестирование и сопровождение предсказуемыми;
- обеспечить совместимость реализации с `main_tech.md`, модульными ТЗ, `DATA_CONTRACT.md`, `MANIFEST_SCHEMAS.md` и `GLOSSARY.md`.

Этот файл не заменяет модульные ТЗ и не дублирует контракт данных.  
Он отвечает на вопрос: **как должен быть организован репозиторий и как в нём должен писаться код**.

---

## 2. Главные принципы

### 2.1. Contract-first, а не code-first

В проекте первичны:

1. архитектурные документы;
2. сквозной data contract;
3. manifest schemas;
4. модульные ТЗ;
5. и только после этого — реализация.

Код не должен молча переопределять уже зафиксированные правила.

### 2.2. Явность важнее удобной магии

Если поведение модуля влияет на downstream-контракт, оно должно быть:

- явно задано в конфиге;
- явно сохранено в manifest;
- явно проверено валидатором.

Скрытые fallback-механизмы допускаются только как явно задокументированный transitional layer.

### 2.3. Модульность важнее «универсальных» скриптов

Pipeline состоит из отдельных модулей:

- `module_prep_data`
- `module_net_train`
- `module_target_predict`
- `module_postprocess_vectorize`
- `module_eval`

Каждый модуль должен иметь чёткие границы ответственности и не должен незаметно брать на себя задачи соседнего модуля.

### 2.4. Reproducibility-by-default

Любой production-like запуск должен быть воспроизводимым по артефактам run-а:

- `config_used`
- `manifest`
- `summary`
- checkpoints / outputs
- diagnostics / logs

Если поведение run-а нельзя восстановить постфактум, это считается дефектом инженерной дисциплины.

---

## 3. Source of truth hierarchy

При конфликте между источниками действует следующий приоритет:

1. утверждённое архитектурное решение / architectural freeze;
2. `main_tech.md`;
3. модульные ТЗ (`module_*.md`);
4. `DATA_CONTRACT.md`;
5. `MANIFEST_SCHEMAS.md`;
6. `GLOSSARY.md`;
7. `REPO_CONVENTIONS.md`;
8. README, комментарии в коде, локальные заметки;
9. неформальные устные/чатовые договорённости, не перенесённые в документы.

Если код противоречит верхнему уровню и это не оформлено отдельным решением, код считается устаревшим или ошибочным.

---

## 4. Канонические имена модулей

В репозитории должны использоваться именно следующие канонические имена модулей:

- `module_prep_data`
- `module_net_train`
- `module_target_predict`
- `module_postprocess_vectorize`
- `module_eval`

Недопустимо вводить параллельно вторые «почти такие же» названия модулей без явной причины.

Если внутри кода используются короткие алиасы, они не должны подменять канонические имена в документах, manifests и config-артефактах.

---

## 5. Рекомендуемая структура репозитория

Базовая рекомендуемая структура v1:

```text
repo/
  docs/
    main_tech.md
    DATA_CONTRACT.md
    MANIFEST_SCHEMAS.md
    GLOSSARY.md
    REPO_CONVENTIONS.md
    TESTING_STRATEGY.md
    EXPERIMENT_TRACKING.md

  configs/
    module_prep_data/
    module_net_train/
    module_target_predict/
    module_postprocess_vectorize/
    module_eval/

  module_prep_data/
  module_net_train/
  module_target_predict/
  module_postprocess_vectorize/
  module_eval/

  tests/
    unit/
    integration/
    e2e/
    golden/

  scripts/
  tools/
  prompts/
  outputs/
```

### 5.1. Что важно

- Документы верхнего уровня должны лежать в предсказуемом месте.
- Конфиги не должны быть размазаны случайно по всем папкам.
- Тесты должны быть отделены от runtime outputs.
- `outputs/` не должен становиться свалкой без структуры.

### 5.2. Допустимые отклонения

Физическая структура репозитория может отличаться, если:

- сохраняются канонические имена модулей;
- роли папок остаются однозначными;
- это не ломает manifests, configs и tooling;
- отклонение явно задокументировано.

---

## 6. Внутренняя структура модуля

Каждый модуль должен стремиться к одинаковой инженерной декомпозиции.

Рекомендуемая структура:

```text
module_x/
  README.md
  configs/
  cli/
  stages/
  io/
  core/
  validators/
  manifests/
  diagnostics/
  tests/   # если модульные тесты хранятся локально
```

### 6.1. Роли подсекций

- `cli/` — entrypoints и запуск стадий;
- `stages/` — stage runners и orchestration logic;
- `io/` — чтение/запись GeoTIFF, vector, json/yaml;
- `core/` — чистая предметная логика;
- `validators/` — проверки контракта и входных данных;
- `manifests/` — сериализация manifests / summaries;
- `diagnostics/` — визуализации, overlays, debug-artifacts.

### 6.2. Что запрещено

Плохо, когда один файл одновременно:

- парсит CLI;
- читает растр;
- нормализует данные;
- строит targets;
- пишет outputs;
- принимает решение о fallback;
- логирует summary.

Такие файлы должны быть разложены на слои ответственности.

---

## 7. Naming conventions

## 7.1. Общие правила именования

Имена должны быть:

- предсказуемыми;
- семантичными;
- без двусмысленных сокращений;
- едиными между кодом, конфигами, manifests и документацией.

### 7.2. Имена файлов

Предпочтительные форматы:

- документы: `UPPER_SNAKE_CASE.md` для общесистемных файлов;
- модульные ТЗ: `module_name.md`;
- manifests: `snake_case_manifest.json`;
- summaries: `summary.json`;
- конфиги: `*.yaml`;
- checkpoints: без двусмысленных суффиксов, с ясной привязкой к run.

### 7.3. Имена сущностей данных

Следует использовать canonical names:

- `img`
- `extent`
- `boundary`
- `distance`
- `valid`
- `meta`
- `extent_prob`
- `boundary_prob`
- `distance_pred`
- `parcel_instance`

Нельзя в одном месте использовать `extent`, а в другом — почти эквивалентное имя без явного адаптера.

### 7.4. Имена feature-режимов

Допустимы только канонические имена:

- `raw8`
- `raw8_idx3`
- `raw8_valid`
- `raw8_idx3_valid`

Запрещены локальные синонимы вроде `raw11`, `full_input`, `features_plus_mask`, если это не формализованный adapter layer.

---

## 8. Конвенции по конфигам

### 8.1. Конфиг обязателен

Все production-like и experiment-like запуски должны быть конфиг-управляемыми.

CLI override допустим, но он должен быть:

- минимальным;
- явным;
- отражённым в `config_used` или manifest.

### 8.2. Разделение конфигов

Конфиги должны быть разделены по модулям.

Пример:

```text
configs/
  module_prep_data/
  module_net_train/
  module_target_predict/
  module_postprocess_vectorize/
  module_eval/
```

### 8.3. Что должно быть в конфиге, а не в коде

В конфиги должны выноситься:

- feature mode;
- valid / nodata policy;
- AOI policy;
- normalization policy;
- patch / tile size;
- overlap;
- thresholds;
- loss weights;
- scheduler / optimizer settings;
- export policy;
- evaluation policy.

### 8.4. Что не должно жить только в конфиге

Если значение определяет runtime-факт, недостаточно хранить его только в конфиге — оно должно попасть и в manifest как **resolved value**.

---

## 9. Конвенции по manifests и summaries

### 9.1. Manifest обязателен

Каждый значимый run должен выпускать manifest по утверждённой схеме.

### 9.2. Summary обязателен

Summary должен быть кратким и пригодным для быстрого чтения человеком и comparison tooling.

### 9.3. Manifest и summary не взаимозаменяемы

- `manifest` — полный протокол запуска;
- `summary` — сжатое техническое резюме.

### 9.4. Manifest writer не должен быть ad-hoc

Запись manifests должна идти через единый слой сериализации / schema-aware writer, а не через разрозненные ручные `json.dump(...)` в десяти местах проекта.

---

## 10. Конвенции по CLI и entrypoints

### 10.1. Один entrypoint — одна явная задача

Entry-point не должен быть «комбайном на всё сразу» без понятных режимов.

### 10.2. Предпочтительный стиль команд

Для stage-based модулей предпочтителен один из двух стилей:

1. отдельные entrypoints по стадиям;
2. единый orchestrator с явными subcommands.

Пример допустимых имён стадий:

- `01_check_inputs`
- `02_prepare_spatial_context`
- `03_prepare_features`
- `04_prepare_targets`
- `05_make_patches`
- `06_split_dataset`
- `07_validate_outputs`

### 10.3. Правила CLI

Каждый CLI должен:

- принимать путь к конфигу;
- уметь писать `config_used`;
- возвращать явный exit status;
- завершаться с понятной ошибкой при нарушении контракта;
- не скрывать критичные решения внутри логики по умолчанию без записи в manifest.

---

## 11. Конвенции по коду

### 11.1. Разделение слоёв ответственности

Внутри модуля код должен быть разделён на уровни:

1. schema / types;
2. validation;
3. pure domain logic;
4. IO adapters;
5. stage orchestration;
6. CLI.

### 11.2. Чистые функции предпочтительны

Если кусок логики можно реализовать как чистую функцию с явным входом и выходом, так и нужно делать.

### 11.3. Минимум глобального состояния

Не рекомендуется строить модуль на скрытом глобальном runtime state.

### 11.4. Нельзя завязывать критичную логику на print/debug-flow

Logging, diagnostics и logic должны быть разделены.

### 11.5. Ошибки должны быть явными

Контрактные нарушения должны приводить к:

- понятному исключению;
- понятному сообщению в логах;
- отражению ошибки в diagnostics / manifest, если это допустимо по жизненному циклу run-а.

---

## 12. Конвенции по geospatial IO

### 12.1. Raster — пространственный эталон

При raster-vector взаимодействии spatial grid растра считается эталонным, если конкретный модуль явно не документирует иное.

### 12.2. Нельзя терять spatial metadata

При чтении и записи raster/vector артефактов нельзя терять без следа:

- CRS;
- transform / georeferencing;
- width / height / count;
- nodata;
- dtype;
- channel semantics, где они важны.

### 12.3. Нельзя молча исправлять пространственные несовместимости

Если данные spatially incompatible и это нельзя безопасно разрешить по конфигу/контракту, модуль обязан упасть с явной диагностикой.

### 12.4. NoData и valid не смешиваются с обычным фоном

Любая операция чтения/записи, затрагивающая invalid-зоны, обязана соблюдать общий valid-policy проекта.

---

## 13. Конвенции по model-side коду

### 13.1. Dataset-side features и assembled model input нельзя смешивать терминологически

Нужно различать:

- dataset-side feature mode (`raw8`, `raw8_idx3`)
- assembled model input (`raw8_valid`, `raw8_idx3_valid`)

### 13.2. Checkpoint без metadata считается неполным

Checkpoint production-like уровня должен сопровождаться metadata, достаточной для безопасного predict-time восстановления:

- `feature_mode`
- `assembled_model_input`
- `in_channels`
- `channel_semantics`
- `valid_as_input_channel`
- normalization-related facts

### 13.3. Predict не должен делать скрытые догадки

Если predict-time восстановление контракта невозможно, система должна завершаться с ошибкой, а не угадывать режим каналов.

---

## 14. Конвенции по output-артефактам

### 14.1. Каждый run должен жить в собственной директории

Рекомендуемая идея:

```text
outputs/<module_name>/<run_id>/
```

### 14.2. Что обычно хранится внутри run-dir

```text
outputs/<module>/<run_id>/
  config_used.yaml
  manifest.json
  summary.json
  logs/
  diagnostics/
  artifacts/
```

### 14.3. Нельзя писать в «общую одну папку» без run isolation

Это ломает воспроизводимость, provenance и сравнение запусков.

### 14.4. Временные рабочие артефакты должны быть отделены от финальных

Нельзя смешивать:

- work artifacts;
- final exported outputs;
- manually edited files;
- cached intermediates.

---

## 15. Конвенции по тестам

### 15.1. У проекта должны быть несколько уровней тестов

Минимально рекомендуются:

- `unit`
- `integration`
- `golden`
- `e2e/smoke`

### 15.2. Что тестировать обязательно

Как минимум:

- valid / nodata semantics;
- feature assembly;
- channel order;
- manifest writing;
- stage restartability;
- checkpoint metadata compatibility;
- predict-time contract restoration;
- fair-comparison safeguards в eval.

### 15.3. Golden tests особенно важны

Для этого проекта golden tests полезны для ловли тихих поломок:

- band order drift;
- target encoding drift;
- normalization drift;
- manifest schema drift.

---

## 16. Логирование и диагностика

### 16.1. Logging не заменяет manifest

Логи нужны для оперативной диагностики, но не являются заменой structured manifest.

### 16.2. Diagnostics должны быть управляемыми

Санити-картинки, overlays, таблицы и промежуточные проверки полезны, но должны:

- быть опциональными или управляемыми уровнем режима;
- не ломать основной pipeline;
- иметь предсказуемое место хранения.

### 16.3. Debug-артефакты не должны подменять production outputs

То, что полезно для расследования, не должно автоматически считаться основным выходом модуля.

---

## 17. Конвенции по обратной совместимости

### 17.1. Backward compatibility допустима, но только явно

Если нужно поддержать старый формат артефактов или старое имя поля:

- это должно быть оформлено как adapter / compatibility layer;
- должно быть явно задокументировано;
- должно быть покрыто тестом;
- не должно становиться скрытой магией по умолчанию.

### 17.2. Deprecated-логика должна иметь срок жизни

Нельзя бесконечно копить старые форматы и временные костыли.

---

## 18. Конвенции по изменениям

### 18.1. Что считается контрактным изменением

Контрактным изменением считаются изменения, затрагивающие:

- feature modes;
- assembled model input;
- valid semantics;
- target encoding;
- manifest schema;
- canonical names;
- правила normalization;
- evaluation fairness rules.

### 18.2. Порядок изменения

Правильный порядок:

1. обновить документ-источник истины;
2. обновить schemas / validators / tests;
3. обновить код;
4. обновить manifests / summaries / README при необходимости.

Не наоборот.

---

## 19. Конвенции при использовании ИИ-ассистентов

### 19.1. Нельзя давать задачу «реализуй весь модуль целиком» без декомпозиции

Предпочтительный стиль:

- сначала schemas;
- потом validators;
- потом pure logic;
- потом IO;
- потом stage runner;
- потом tests;
- потом review.

### 19.2. Любой сгенерированный код должен проверяться на:

- hidden fallbacks;
- silent dtype/range conversions;
- band-order assumptions;
- потерю valid semantics;
- нарушение manifest/provenance policy.

### 19.3. Документы важнее генеративной самоуверенности

Если ассистент предлагает решение, которое противоречит документам проекта, приоритет у документов, пока не принят новый архитектурный консенсус.

---

## 20. Минимальный checklist для нового кода

Перед merge или принятием новой части кода стоит проверить:

- не нарушен ли data contract;
- не введены ли новые неканонические имена;
- не появилась ли скрытая догадка вместо явного правила;
- пишет ли код manifest и summary корректно;
- воспроизводим ли run;
- можно ли восстановить provenance;
- покрыт ли новый кусок хотя бы минимальным тестом;
- не смешаны ли responsibilities разных модулей.

---

## 21. Краткая формула инженерной дисциплины проекта

Для проекта **«ИИ для полей»** хороший код — это код, который:

- реализует уже зафиксированный контракт;
- не теряет provenance;
- не прячет важные решения;
- не ломает воспроизводимость;
- не смешивает роли модулей;
- остаётся читаемым и проверяемым человеком.

Именно это считается основной инженерной нормой v1.
