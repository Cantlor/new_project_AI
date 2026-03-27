# DECISIONS.md

## Назначение документа

Этот документ фиксирует **ключевые архитектурные решения проекта** `ИИ для полей` в коротком, операционном формате.

Его задача:

- хранить ответы на вопрос **«почему мы сделали именно так»**;
- не размазывать важные решения по чатам, README, manifests и случайным заметкам;
- отделять **принятые решения** от открытых вопросов и future branches;
- помогать при ревью кода, сравнении run-ов и продолжении работы в новых чатах.

Этот файл **не заменяет**:

- `main_tech.md`
- `module_prep_data.md`
- `module_net_train.md`
- `module_target_predict.md`
- `module_eval.md`
- `DATA_CONTRACT.md`
- `MANIFEST_SCHEMAS.md`

Эти документы задают полное ТЗ и контракты.

`DECISIONS.md` хранит **сжатый реестр проектных решений**, на которые потом должны опираться код, manifests, тесты и сравнение экспериментов.

---

## Как пользоваться этим файлом

### Когда добавлять новую запись

Новая запись нужна, если принято решение, которое:

- влияет на несколько модулей сразу;
- меняет baseline pipeline;
- меняет data contract или manifest policy;
- влияет на сравнимость run-ов;
- исключает альтернативы и должно быть remembered как project freeze.

### Когда запись не нужна

Отдельная запись обычно не нужна, если изменение:

- чисто локальное и не меняет контракт;
- редакторское;
- относится только к refactor без изменения поведения;
- является временным экспериментом, а не baseline-решением.

### Статусы

Допустимые статусы:

- `accepted`
- `superseded`
- `deprecated`
- `proposed`
- `rejected`

### Рекомендуемый формат записи

```md
## DEC-00X — Краткое название

- Status: accepted
- Date: YYYY-MM-DD
- Scope: global | module_prep_data | module_net_train | module_target_predict | module_postprocess_vectorize | module_eval

### Контекст
...

### Решение
...

### Последствия
...

### Чем сознательно не стали делать
...

### Связанные документы
- ...
```

---

# Принятые решения проекта

## DEC-001 — Модульный pipeline является обязательным baseline

- Status: accepted
- Date: 2026-03-27
- Scope: global

### Контекст

Проект решает задачу выделения и разделения сельскохозяйственных полей не одним монолитным скриптом, а последовательностью разных стадий с разной логикой, артефактами и критериями качества.

### Решение

В проекте фиксируется единый baseline pipeline:

```text
module_prep_data
  -> module_net_train
  -> module_target_predict
  -> module_postprocess_vectorize
  -> module_eval
```

Каждый модуль имеет собственные входы, выходы, manifests, summaries и критерии приемки.

### Последствия

- Нельзя смешивать подготовку данных, обучение, инференс, постпроцессинг и оценку в один неразделимый runtime.
- Становится возможным воспроизводимое сравнение run-ов по модулям.
- Появляется жёсткая необходимость в data contract и manifest schemas.

### Чем сознательно не стали делать

- Не используем single-script pipeline как baseline.
- Не прячем postprocess и eval внутрь predict.

### Связанные документы

- `main_tech.md`
- `DATA_CONTRACT.md`
- `MANIFEST_SCHEMAS.md`

---

## DEC-002 — `valid` имеет двойную роль: mask и входной канал модели

- Status: accepted
- Date: 2026-03-27
- Scope: global

### Контекст

В проекте критично корректно обрабатывать границы растра, вырезанные области, NoData и invalid-пиксели. Простая политика «invalid = обычный фон» признана недостаточной и опасной для качества.

### Решение

Слой `valid` фиксируется как обязательный элемент baseline pipeline сразу в двух ролях:

1. **служебная valid-mask** для ignore-policy в train / predict / postprocess / eval;
2. **дополнительный входной канал модели** в assembled model input contract.

`valid` должен вычисляться **до** любых замен NoData на fill value.

### Последствия

- Train и predict обязаны одинаково трактовать valid.
- Loss, metrics, diagnostics и downstream postprocess не могут игнорировать valid-policy.
- Model input contract нельзя “урезать” ради памяти или удобства.

### Чем сознательно не стали делать

- Не встраиваем valid только неявно внутрь `img`.
- Не делаем valid только служебной маской без входного канала.
- Не допускаем hidden fallback, где отсутствие valid silently tolerated.

### Связанные документы

- `main_tech.md`
- `module_prep_data.md`
- `module_net_train.md`
- `module_target_predict.md`
- `DATA_CONTRACT.md`

---

## DEC-003 — Финальный feature contract ограничен двумя baseline-режимами

- Status: accepted
- Date: 2026-03-27
- Scope: global

### Контекст

Нужно сравнивать как минимум два честных baseline-подхода: обучение на исходных 8 каналах и обучение на 8 каналах плюс три фиксированных индекса.

### Решение

Фиксируются только два dataset-side feature modes:

- `raw8`
- `raw8_idx3`

И два assembled model input contracts:

- `raw8_valid`
- `raw8_idx3_valid`

Для `raw8_idx3` фиксированы derived indices:

- `NDVI`
- `SAVI`
- `NDWI`

### Последствия

- Сравнение `raw8` vs `raw8_idx3` становится официальным baseline comparison.
- Любой код обязан различать dataset-side feature mode и финальный assembled model input.
- `in_channels` должен совпадать не с abstract feature mode, а с реально собранным input tensor.

### Чем сознательно не стали делать

- Не открываем baseline на произвольное число feature modes.
- Не поддерживаем скрытые channel hacks и молчаливые адаптации.

### Связанные документы

- `main_tech.md`
- `module_prep_data.md`
- `module_net_train.md`
- `module_target_predict.md`
- `DATA_CONTRACT.md`

---

## DEC-004 — `module_prep_data` остаётся source of truth для train-ready dataset

- Status: accepted
- Date: 2026-03-27
- Scope: module_prep_data / module_net_train

### Контекст

Обучение становится хрупким, если train pipeline начинает сам “досчитывать” пропущенные слои, silently infer target semantics или работать напрямую по произвольным сырым входам без нормализованного экспортного контракта.

### Решение

`module_net_train` официально принимает train-ready dataset только из экспорта `module_prep_data`.

Обязательные sample-level слои baseline v1:

- `img`
- `extent`
- `boundary`
- `distance`
- `valid`
- `meta`

### Последствия

- `module_net_train` не должен жить на ad-hoc dataset assembly как baseline.
- Ошибки контракта должны ловиться на стыке `prep_data -> train`.
- `module_prep_data` становится местом, где окончательно фиксируются feature stack и target semantics.

### Чем сознательно не стали делать

- Не делаем train baseline, который читает “что дадут” и сам догадается.
- Не переносим target-building в training runtime.

### Связанные документы

- `module_prep_data.md`
- `module_net_train.md`
- `DATA_CONTRACT.md`
- `TESTING_STRATEGY.md`

---

## DEC-005 — Boundary-aware multitask formulation принята как baseline задачи

- Status: accepted
- Date: 2026-03-27
- Scope: module_net_train / module_target_predict / module_postprocess_vectorize / module_eval

### Контекст

Задача проекта не сводится к грубой бинарной сегментации “поле / не поле”. Требуется поддержка разделения соседних участков и сохранения внутренней структуры границ.

### Решение

Baseline постановка фиксируется как multitask:

- `extent`
- `boundary`
- `distance`

Эти три target/output не считаются взаимозаменяемыми.

### Последствия

- Архитектура, losses, manifests и eval обязаны учитывать три головы/выхода.
- Postprocess может использовать не только extent, но и boundary/distance для marker generation и watershed.
- Eval не может ограничиваться только общей IoU.

### Чем сознательно не стали делать

- Не делаем single-head binary segmentation baseline.
- Не подменяем boundary общей точностью extent.

### Связанные документы

- `main_tech.md`
- `module_net_train.md`
- `module_target_predict.md`
- `module_eval.md`
- `GLOSSARY.md`

---

## DEC-006 — Boundary target должен быть linework-faithful

- Status: accepted
- Date: 2026-03-27
- Scope: module_prep_data / module_net_train / module_eval

### Контекст

Ранний опыт показал, что boundary-таргет, построенный как суррогатный градиент от extent, может почти потерять внутренние границы и исказить саму задачу.

### Решение

Boundary target baseline v1 строится **linework-faithful от реальных polygon boundaries**.

Boundary encoding для v1:

- `background`
- `skeleton`
- `buffer`

`boundary_raw` сохраняется как обязательный диагностический промежуточный артефакт подготовки данных.

### Последствия

- Training и eval получают boundary semantics, согласованные с реальными границами.
- Внутренние границы не должны пропадать из target pipeline как артефакт simplification.
- Postprocess и forensic-анализ получают более честную опору для оценки ошибок границ.

### Чем сознательно не стали делать

- Не считаем edge-of-extent достаточной заменой real boundary linework.
- Не выбрасываем `boundary_raw` из prep pipeline.

### Связанные документы

- `module_prep_data.md`
- `module_net_train.md`
- `DATA_CONTRACT.md`

---

## DEC-007 — `module_target_predict` работает в checkpoint-driven режиме

- Status: accepted
- Date: 2026-03-27
- Scope: module_target_predict / module_net_train

### Контекст

Инференс нельзя делать безопасным, если пользователь или код вручную задаёт критичные параметры вроде `feature_mode`, `in_channels`, роли `valid` и normalization rules отдельно от checkpoint.

### Решение

Baseline predict работает как **checkpoint-driven inference**.

Источники истины для predict:

- checkpoint metadata;
- `config_used.yaml` из train run;
- exported train manifest.

Predict обязан автоматически восстановить:

- dataset-side feature mode;
- assembled model input contract;
- число каналов;
- channel semantics;
- роль `valid`;
- normalization policy.

### Последствия

- Predict не должен зависеть от ручных догадок.
- Checkpoint metadata становится обязательной частью training contract.
- Любой несовместимый checkpoint должен приводить к явной ошибке.

### Чем сознательно не стали делать

- Не разрешаем baseline predict, который “примерно знает”, как был обучен чекпойнт.
- Не разрешаем ручное переописание model input contract как нормальный режим.

### Связанные документы

- `module_net_train.md`
- `module_target_predict.md`
- `DATA_CONTRACT.md`
- `MANIFEST_SCHEMAS.md`

---

## DEC-008 — Train-derived normalization stats обязательны и едины для train/predict

- Status: accepted
- Date: 2026-03-27
- Scope: global

### Контекст

Рассинхрон нормализации между training и inference способен сделать даже правильную модель practically incompatible с собственным checkpoint.

### Решение

В baseline v1 используется единая смысловая нормализация:

- `valid` вычисляется до преобразований;
- данные приводятся к `float32`;
- используется robust normalization по train-derived statistics;
- predict обязан использовать те же train-derived stats.

### Последствия

- Статистики нормализации должны сохраняться как часть metadata/manifests.
- Predict не может silently переоценивать stats на новом растре как baseline behavior.
- Сравнение run-ов требует знания normalization provenance.

### Чем сознательно не стали делать

- Не используем раздельные train/predict normalization rules.
- Не допускаем hidden dtype or scaling changes.

### Связанные документы

- `module_prep_data.md`
- `module_target_predict.md`
- `DATA_CONTRACT.md`
- `MANIFEST_SCHEMAS.md`

---

## DEC-009 — Hardware-adaptive runtime допустим только без поломки model/data contract

- Status: accepted
- Date: 2026-03-27
- Scope: global

### Контекст

Проект должен работать на разном железе, но runtime adaptability не должна ломать воспроизводимость или менять смысл данных/входов модели.

### Решение

Фиксируется hardware-adaptive policy:

- приоритет устройств: `CUDA -> MPS -> CPU`;
- AMP включается автоматически, если это безопасно;
- при OOM деградация идёт через runtime-параметры;
- feature contract и valid-policy не ломаются ради памяти.

### Последствия

- Runtime decisions должны попадать в manifests.
- OOM-handling должен менять batch/tile/runtime knobs, а не смысл входа модели.
- Reproducibility требует логировать `device_requested`, `device_resolved`, `amp_requested`, `amp_used`, `oom_fallbacks_applied`.

### Чем сознательно не стали делать

- Не разрешаем memory workaround через скрытую смену числа каналов.
- Не разрешаем отключение valid-aware logic ради экономии памяти.

### Связанные документы

- `main_tech.md`
- `module_net_train.md`
- `module_target_predict.md`
- `module_eval.md`
- `MANIFEST_SCHEMAS.md`

---

## DEC-010 — Predict и postprocess разделяются по ответственности

- Status: accepted
- Date: 2026-03-27
- Scope: module_target_predict / module_postprocess_vectorize

### Контекст

Финальная векторизация требует thresholding, repair, watershed и topology cleanup. Если смешать это с predict, становится трудно сравнивать модели и отдельно разбирать ошибки сети и ошибки постпроцессинга.

### Решение

`module_target_predict` заканчивается на геопривязанных raster outputs:

- `extent_prob.tif`
- `boundary_prob.tif`
- `distance_pred.tif`
- `valid.tif`

Финальные операции:

- thresholding;
- boundary repair;
- marker generation;
- watershed;
- polygonization;
- topology cleanup;

выносятся в `module_postprocess_vectorize`.

### Последствия

- Модель можно честно оценивать отдельно от векторизации.
- Baseline comparison может отдельно сравнивать predict и postprocess variants.
- Predict manifest и postprocess manifest несут разную provenance-нагрузку.

### Чем сознательно не стали делать

- Не делаем “умный predict”, который сразу молча пишет polygons как baseline.
- Не прячем watershed/polygonization за неявными флагами внутри predict.

### Связанные документы

- `module_target_predict.md`
- `module_postprocess_vectorize` freeze notes
- `module_eval.md`
- `DATA_CONTRACT.md`

---

## DEC-011 — Postprocess baseline использует hybrid boundary repair и marker-controlled watershed

- Status: accepted
- Date: 2026-03-27
- Scope: module_postprocess_vectorize

### Контекст

Для разделения соседних полей одной вероятностной карты extent недостаточно. Нужна процедура, которая опирается на boundary/distance и умеет консервативно восстанавливать разделяющую структуру.

### Решение

Baseline postprocess фиксируется так:

1. valid/AOI suppression;
2. validation-calibrated thresholding;
3. hybrid boundary repair;
4. marker generation = `extent-core ∩ low-boundary ∩ high-distance`;
5. marker-controlled watershed;
6. region filtering / constrained merge;
7. `parcel_instance.tif`;
8. polygonization;
9. conservative topology cleanup;
10. `polygon_confidence` computation;
11. export default в `GPKG`.

### Последствия

- Parcel delineation опирается не на одну эвристику, а на связку extent/boundary/distance.
- Итоговый raster instance layer становится обязательным промежуточным output.
- Векторный confidence должен быть rule-based baseline output, а не только визуальным впечатлением.

### Чем сознательно не стали делать

- Не выбираем morphology-only или graph-only repair как baseline.
- Не делаем SHP основным архивным форматом.
- Не используем агрессивный topology cleanup как default.

### Связанные документы

- `module_postprocess_vectorize` freeze notes
- `GLOSSARY.md`
- `DATA_CONTRACT.md`

---

## DEC-012 — Основной архивный векторный формат: GPKG

- Status: accepted
- Date: 2026-03-27
- Scope: module_postprocess_vectorize / module_eval

### Контекст

Итоговые полигоны должны храниться в стабильном формате, который не ломается на ограничениях старых shapefile-соглашений.

### Решение

Основной рабочий и архивный формат векторных результатов baseline v1:

- `GPKG`

`SHP` допускается только как optional export.

### Последствия

- Все canonical output contracts ориентируются на `parcels.gpkg`.
- Eval и comparison tooling должны считать `GPKG` baseline format.

### Чем сознательно не стали делать

- Не делаем Shapefile default format.
- Не строим archive policy вокруг legacy-ограничений SHP.

### Связанные документы

- `module_postprocess_vectorize` freeze notes
- `GLOSSARY.md`
- `DATA_CONTRACT.md`

---

## DEC-013 — Eval обязан быть provenance-aware и поддерживать fair comparison

- Status: accepted
- Date: 2026-03-27
- Scope: module_eval

### Контекст

Сравнение run-ов бессмысленно, если не зафиксированы источники GT, thresholds, scene list, valid/AOI policy и source manifests.

### Решение

`module_eval` baseline v1 обязан:

- быть provenance-aware;
- считать pixel / boundary / object-structure группы метрик;
- поддерживать multi-run comparison;
- фиксировать source run identifiers, manifests, configs, scene list, threshold provenance и valid/AOI policy.

### Последствия

- Нельзя честно сравнивать run-ы без восстановимого eval context.
- Delta tables без manifest-backed provenance не считаются достаточными.
- Bucketed evaluation и error taxonomy становятся baseline, а не optional add-on.

### Чем сознательно не стали делать

- Не считаем plain IoU leaderboard достаточным итогом проекта.
- Не допускаем скрытую смену eval policy между сравниваемыми run-ами.

### Связанные документы

- `module_eval.md`
- `EXPERIMENT_TRACKING.md`
- `MANIFEST_SCHEMAS.md`

---

## DEC-014 — Manifest-first и run-artifact discipline являются обязательной частью baseline

- Status: accepted
- Date: 2026-03-27
- Scope: global

### Контекст

Без стабильных manifests и summaries проект быстро теряет воспроизводимость, provenance и возможность forensic-аудита.

### Решение

Каждый модуль baseline v1 обязан выпускать как минимум:

- `config_used.*`
- `manifest.*`
- `summary.*`
- logs / diagnostics
- versioned outputs

Manifest должен быть достаточно полным, чтобы восстановить:

- входы;
- выходы;
- resolved contract;
- runtime decisions;
- provenance.

### Последствия

- Код, который пишет outputs без сопроводительных manifests, не считается production-like baseline.
- Сравнение run-ов должно опираться на manifests, а не только на имена папок.
- Testing strategy обязана включать проверки manifest completeness и provenance consistency.

### Чем сознательно не стали делать

- Не считаем manifests “дополнительной документацией”.
- Не считаем папку с TIFF-файлами достаточным run record.

### Связанные документы

- `MANIFEST_SCHEMAS.md`
- `EXPERIMENT_TRACKING.md`
- `TESTING_STRATEGY.md`

---

## DEC-015 — Baseline coding workflow должен быть contract-first, а не code-first

- Status: accepted
- Date: 2026-03-27
- Scope: global

### Контекст

При сложном geospatial ML pipeline прямой переход “сразу к реализации модуля целиком” создаёт скрытую магию, случайные fallback-ветки и поломки на стыках.

### Решение

В baseline workflow принята последовательность:

1. ТЗ и freeze ключевых решений;
2. сквозной `DATA_CONTRACT.md`;
3. `MANIFEST_SCHEMAS.md`;
4. `REPO_CONVENTIONS.md`;
5. `TESTING_STRATEGY.md`;
6. `EXPERIMENT_TRACKING.md`;
7. только после этого — поэтапная реализация кода модулями.

### Последствия

- Код должен следовать контрактам, а не придумывать их по месту.
- LLM-assisted development становится контролируемее.
- Review смещается с “нравится ли код” на “соблюдает ли он контракт”.

### Чем сознательно не стали делать

- Не идём в full-code generation без frozen interfaces.
- Не рассматриваем README-only documentation как достаточную подготовку к реализации.

### Связанные документы

- `REPO_CONVENTIONS.md`
- `TESTING_STRATEGY.md`
- `EXPERIMENT_TRACKING.md`
- `Вставленный текст.txt` (как исходный импульс, но не как source of truth)

---

# Открытые вопросы для будущих решений

Ниже перечислены вопросы, которые уже упоминались, но пока не должны считаться зафиксированными baseline-решениями.

## OPEN-001 — Нужен ли default export логитов в `module_target_predict`

Статус: open

Сейчас baseline требует probability outputs и допускает logits как optional.
Нужно отдельно решить, становятся ли логиты официальным default-output или остаются диагностической опцией.

## OPEN-002 — Нужен ли официальный HTML-report baseline в `module_eval`

Статус: open

CSV/JSON уже входят в практический baseline, но HTML-report пока не должен считаться обязательным без отдельного решения.

## OPEN-003 — Нужен ли официальный leaderboard / benchmark package format

Статус: open

Внутреннее comparison уже входит в baseline, но отдельный leaderboard-format пока не зафиксирован.

## OPEN-004 — Нужна ли baseline policy для bootstrap CIs / statistical significance

Статус: open

Полезно для зрелой evaluation framework, но пока не входит в обязательный v1.

---

# Короткое правило обновления документа

Если меняется хоть один из следующих пунктов, это почти наверняка повод добавить новую decision entry:

- pipeline order;
- роль `valid`;
- состав feature modes;
- assembled model input contract;
- boundary encoding;
- predict/postprocess responsibility split;
- normalization policy;
- main vector export format;
- manifest-first policy;
- fair-comparison rules.

Если меняется только реализация без изменения смысла и контракта, новую запись обычно добавлять не нужно.
