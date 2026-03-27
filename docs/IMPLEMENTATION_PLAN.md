# IMPLEMENTATION_PLAN.md

## 1. Назначение документа

Этот документ задаёт **практический порядок реализации** проекта `ИИ для полей`.

Он отвечает на вопросы:

- в каком порядке писать код;
- что нужно сделать до начала реализации каждого модуля;
- какие критерии готовности должны быть выполнены перед переходом к следующему этапу;
- какие ошибки нельзя допускать в процессе разработки.

Документ **не заменяет ТЗ модулей**.  
Его задача — превратить уже принятые документы проекта в рабочий plan-of-execution.

---

## 2. Базовый принцип реализации

Проект должен разрабатываться **не “по вдохновению”, а по контракту**.

Порядок всегда такой:

1. сначала архитектура и ТЗ;
2. потом сквозной data contract;
3. потом manifest/schema conventions;
4. потом repo/testing rules;
5. потом skeleton-код;
6. потом validators и I/O;
7. потом core logic;
8. потом integration;
9. потом experiments и refinement.

Запрещённый путь:
- сначала быстро написать “почти working code”,
- потом пытаться понять, какой у него контракт,
- потом чинить совместимость между модулями.

---

## 3. Общая последовательность реализации

Рекомендуемый порядок:

```text
0. документация и contract layer
1. repo skeleton и infrastructure
2. module_prep_data
3. module_net_train
4. module_target_predict
5. module_postprocess_vectorize
6. module_eval
7. end-to-end hardening
8. baseline comparison and refinement
```

Именно такой порядок соответствует принятому pipeline проекта и минимизирует количество временных костылей между модулями.

---

## 4. Этап 0 — документация и контрактный слой

### Цель
Перед началом кода завершить минимальный набор документов, который делает проект управляемым.

### Должно быть готово
- `main_tech.md`
- ТЗ модулей
- `GLOSSARY.md`
- `DATA_CONTRACT.md`
- `MANIFEST_SCHEMAS.md`
- `REPO_CONVENTIONS.md`
- `TESTING_STRATEGY.md`
- `EXPERIMENT_TRACKING.md`
- `DECISIONS.md`

### Критерий готовности этапа
Можно однозначно ответить:
- какие данные ходят между модулями;
- как называется каждый обязательный слой;
- как трактуется `valid`;
- какие manifests должны выпускаться;
- как сравниваются run-ы.

Переходить дальше без этого нельзя.

---

## 5. Этап 1 — repo skeleton и infrastructure

### Цель
Подготовить каркас репозитория и инженерную основу, не реализуя ещё “умную” логику.

### Что сделать
- оформить базовую структуру каталогов;
- определить места для модулей, configs, tests, docs, artifacts;
- подготовить общий package/layout;
- определить общие utility modules, где это оправдано;
- подготовить заготовки manifest/summary/config writers;
- подготовить общий logging approach;
- подготовить базовые validators / schema helpers;
- добавить минимальные smoke-tests для инфраструктуры.

### Что не делать на этом этапе
- не писать полностью модель;
- не писать полноценный postprocess;
- не городить сложный training loop;
- не делать “временные” contracts ради скорости.

### Критерий готовности этапа
В репозитории есть устойчивый каркас, в который дальше можно встраивать модули без хаоса.

---

## 6. Этап 2 — `module_prep_data`

### Почему первым
Все downstream-модули зависят от того, как именно определены:
- feature modes;
- valid / NoData policy;
- target layers;
- train-ready dataset layout;
- manifests подготовки данных.

Если начать не с `module_prep_data`, остальные модули либо будут строиться на догадках, либо потребуют потом болезненного переписывания.

### Порядок реализации внутри модуля

#### 2.1. Сначала schemas / config / manifests
Нужно определить:
- config structure;
- входные артефакты;
- выходные артефакты;
- схемы manifests;
- required directories;
- sample-level metadata.

#### 2.2. Потом validators
Проверки должны покрывать:
- raster readability;
- vector readability;
- CRS compatibility;
- band count;
- nodata interpretation;
- возможность построить `valid`;
- spatial consistency.

#### 2.3. Потом feature assembly
Реализовать:
- `raw8`;
- `raw8_idx3`;
- корректный расчёт индексов;
- фиксацию `channel_semantics`;
- совместимость с `DATA_CONTRACT.md`.

#### 2.4. Потом target generation
Реализовать:
- `extent`;
- `boundary` как linework-faithful target;
- `distance`;
- `valid`;
- `boundary_raw` как диагностический слой.

#### 2.5. Потом patching и split
Реализовать:
- стратегический sampling;
- экспорт patch artifacts;
- split train/val/test;
- manifests и summaries.

### Критерий готовности этапа
Модуль стабильно:
- принимает source raster/vector/AOI;
- выпускает train-ready dataset;
- пишет обязательные manifests;
- проходит smoke e2e и целевые integration checks.

Без этого нельзя переходить к `module_net_train`.

---

## 7. Этап 3 — `module_net_train`

### Цель
Построить training pipeline, который работает только на экспортированном контракте `module_prep_data`.

### Порядок реализации внутри модуля

#### 3.1. Dataset reader
Сначала реализовать reader, который:
- читает только canonical train-ready layout;
- понимает `feature_mode`;
- собирает `raw8_valid` или `raw8_idx3_valid`;
- строго уважает `valid` и ignore policy.

#### 3.2. Training config и validators
Определить:
- входные каналы;
- target semantics;
- loss weights;
- метрику выбора best checkpoint;
- device / AMP policy;
- error behavior при несовместимом датасете.

#### 3.3. Model baseline
Реализовать baseline architecture без преждевременной усложнённости.

Важно:
- входной контракт должен быть явным;
- головы `extent / boundary / distance` должны быть согласованы с target contract;
- deep supervision и refine-блоки допустимы только если не ломают baseline clarity.

#### 3.4. Training loop
Реализовать:
- optimizer / scheduler;
- mixed precision policy;
- логирование;
- checkpoint saving;
- metrics history;
- summary and manifest writing.

#### 3.5. Checkpoint metadata
Отдельно обеспечить:
- сохранение `feature_mode`;
- `assembled_model_input`;
- `in_channels`;
- `channel_semantics`;
- `valid_as_input_channel`;
- normalization metadata.

### Критерий готовности этапа
Есть reproducible train-run, который:
- берёт только prep-data export;
- сохраняет usable checkpoint;
- выпускает checkpoint metadata, достаточную для автоматического predict.

---

## 8. Этап 4 — `module_target_predict`

### Цель
Сделать checkpoint-driven inference без ручной магии.

### Порядок реализации

#### 4.1. Checkpoint resolver
Реализовать загрузку:
- checkpoint;
- checkpoint metadata;
- train config / manifest при необходимости.

#### 4.2. Predict-time feature reconstruction
Реализовать:
- автоматическое определение `feature_mode`;
- сборку `raw8_valid` или `raw8_idx3_valid`;
- восстановление normalization;
- корректную сборку `valid` как input channel.

#### 4.3. Tiled inference
Реализовать:
- tile size;
- overlap;
- blending;
- invalid-only tile skip;
- stitched output writing.

#### 4.4. Predict outputs
Стабильно выпускать:
- `extent_prob.tif`
- `boundary_prob.tif`
- `distance_pred.tif`
- `valid.tif`
- `predict_manifest.json`
- `summary.json`

### Критерий готовности этапа
Predict запускается только по checkpoint + input raster + config overrides, без ручного знания training setup.

---

## 9. Этап 5 — `module_postprocess_vectorize`

### Цель
Преобразовать predict outputs в parcel instances и финальные полигоны без подмены ответственности predict-модуля.

### Порядок реализации

#### 5.1. Input validators
Проверить совместимость:
- rasters;
- valid layer;
- AOI;
- threshold policy;
- source manifests.

#### 5.2. Baseline pipeline
Реализовать:
- valid/AOI suppression;
- thresholding;
- boundary-aware repair;
- marker generation;
- watershed / region separation;
- filtering and constrained merge;
- polygonization;
- conservative cleanup.

#### 5.3. Outputs и provenance
Сохранять:
- `parcel_instance.tif`
- vector outputs (`.gpkg`)
- `postprocess_manifest.json`
- `summary.json`

### Критерий готовности этапа
Есть стабильный baseline-vectorization pipeline, который можно честно оценивать в `module_eval`.

---

## 10. Этап 6 — `module_eval`

### Цель
Сделать честную оценку, сравнение run-ов и forensic-ready reporting.

### Порядок реализации

#### 6.1. Input/provenance validation
Реализовать проверку:
- GT sources;
- predict/postprocess outputs;
- source run ids;
- threshold provenance;
- valid/AOI policy;
- compatibility rules for comparison.

#### 6.2. Metric groups
Реализовать минимум:
- global/pixel metrics;
- boundary metrics;
- object/structure metrics.

#### 6.3. Reporting
Сохранять:
- `eval_manifest.json`
- `summary.json`
- comparison tables
- scene-level and bucketed reports
- error taxonomy outputs

### Критерий готовности этапа
Модуль умеет делать не просто “оценку числами”, а воспроизводимую и fair comparison evaluation.

---

## 11. Этап 7 — end-to-end hardening

### Цель
После появления всех модулей стабилизировать систему как единый pipeline.

### Что сделать
- прогнать full e2e на baseline dataset;
- проверить manifests и provenance по всей цепочке;
- проверить честную совместимость run-ов;
- проверить отсутствие hidden fallbacks;
- проверить, что `valid` не теряет роль ни в одном модуле;
- проверить, что feature contract одинаково читается в train и predict;
- сделать один baseline handoff и один baseline run report.

### Критерий готовности этапа
Pipeline работает сквозным запуском и даёт артефакты, которые можно разбирать не только вручную, но и формально.

---

## 12. Этап 8 — refinement и сравнение baseline-вариантов

### Цель
После появления стабильного baseline начинать улучшения осмысленно, а не хаотично.

### Что делать
- сравнивать `raw8_valid` vs `raw8_idx3_valid`;
- тестировать чувствительность к patch size;
- настраивать loss weights;
- анализировать postprocess thresholds;
- улучшать boundary quality;
- делать comparison only against compatible runs.

### Что не делать
- не менять сразу много факторов за один run;
- не сравнивать run-ы с разным contract silently;
- не менять data semantics без обновления документации.

---

## 13. Как писать код внутри каждого этапа

Общий безопасный порядок для любой новой части кода:

1. сначала обновить или проверить документ-контракт;
2. потом сделать config/schema;
3. потом validators;
4. потом I/O;
5. потом core logic;
6. потом manifest/summary writing;
7. потом tests;
8. потом integration run;
9. потом handoff / run report.

Если шаги переставлены, риск хаоса резко растёт.

---

## 14. Definition of Done для любого модуля

Модуль считается условно завершённым только если:

- его входной контракт ясен;
- его выходной контракт ясен;
- manifests и summary пишутся стабильно;
- тесты покрывают хотя бы критические сценарии;
- hidden fallbacks не используются там, где нужен явный контракт;
- есть хотя бы один рабочий run;
- есть краткий run report / handoff;
- downstream-модуль может использовать его outputs без ручной догадки.

---

## 15. Основные риски реализации

### Риск 1. Начать с модели, а не с данных
Почти гарантированно приведёт к переизобретению train contract.

### Риск 2. Потерять семантику `valid`
Это сломает и train, и predict, и postprocess, и eval.

### Риск 3. Смешать dataset-side features и assembled model input
Нужно всегда помнить:
- `raw8` и `raw8_idx3` — это dataset feature modes;
- `raw8_valid` и `raw8_idx3_valid` — это финальный model input.

### Риск 4. Недодокументированные runtime changes
OOM fallback, AMP policy, device changes и implicit thresholds должны фиксироваться.

### Риск 5. Слишком ранний “оптимизационный” рефактор
Сначала baseline correctness, потом ускорение и усложнение.

---

## 16. Что делать прямо сейчас

Практический ближайший порядок работ:

1. довести документационный слой до законченного baseline;
2. сверить, что все созданные supporting docs согласованы между собой;
3. подготовить repo skeleton;
4. начать реализацию `module_prep_data` в contract-first режиме;
5. не переходить к `module_net_train`, пока prep-data export не станет стабильным;
6. дальше двигаться строго по pipeline.

---

## 17. Короткий operational rule

Для проекта `ИИ для полей` правильная стратегия такая:

**Сначала сделать понятный и проверяемый контракт.  
Потом сделать минимально достаточную реализацию.  
Потом проверить её через manifests, tests и reproducible runs.  
И только потом улучшать качество модели и постпроцессинга.**
