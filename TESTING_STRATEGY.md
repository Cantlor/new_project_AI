# TESTING_STRATEGY.md

## 1. Назначение документа

Этот документ фиксирует **стратегию тестирования** проекта **«ИИ для полей»**.

Его задача:

- задать единые правила проверки корректности модулей и их стыков;
- сделать тестирование частью инженерного контракта, а не «опциональной доработкой потом»;
- обеспечить совместимость реализации с `main_tech.md`, модульными ТЗ, `DATA_CONTRACT.md`, `MANIFEST_SCHEMAS.md`, `GLOSSARY.md` и `REPO_CONVENTIONS.md`;
- уменьшить риск скрытых регрессий в геопривязке, feature contract, `valid`/NoData policy, manifests и downstream-интеграции.

Этот документ не заменяет модульные критерии приемки и не дублирует eval-метрики проекта.  
Он отвечает на вопрос: **что, на каком уровне и каким образом должно проверяться в кодовой базе**.

---

## 2. Главные принципы

### 2.1. Contract-first testing

В проекте тестируется не только «работает ли код», но и:

- соблюдается ли data contract;
- сохраняется ли feature contract;
- не потеряна ли двойная роль `valid`;
- воспроизводимы ли manifests и run artifacts;
- совместимы ли соседние модули без ручных правок.

Тесты должны подтверждать архитектурные решения, а не только отсутствие исключений.

### 2.2. Явная проверка важнее smoke-иллюзии

Тест «скрипт не упал» полезен, но недостаточен.

Если модуль:

- молча переставил каналы,
- потерял `valid`,
- изменил трактовку `boundary`,
- записал неполный manifest,
- или стал использовать иной normalization contract,

то такой run должен считаться дефектным даже при внешне успешном завершении.

### 2.3. Тесты должны покрывать и успех, и ожидаемые ошибки

Для проекта критичны не только happy-path сценарии, но и гарантированно правильные отказы.

Система обязана завершаться с явной ошибкой, если нарушены:

- spatial alignment contract;
- `feature_mode` / `in_channels` consistency;
- `valid` interpretation;
- target semantics;
- predict compatibility checkpoint metadata.

Значит, отрицательные тесты обязательны.

### 2.4. Reproducibility is testable

Воспроизводимость должна проверяться тестами, а не считаться «по умолчанию».  
Если manifest, summary, config_used или source provenance нельзя восстановить, это предмет тестового провала.

---

## 3. Цели тестирования

Система тестирования должна подтверждать, что проект:

1. корректно обрабатывает геоданные и их пространственные свойства;
2. соблюдает сквозной контракт данных между модулями;
3. правильно работает с `valid` и NoData;
4. одинаково трактует feature contract в `prep_data`, `train` и `predict`;
5. сохраняет достаточные manifests/summaries для forensic-анализа;
6. поддерживает hardware-adaptive runtime без ломания контрактов;
7. допускает честное сравнение run-ов без скрытой смены политик;
8. остаётся пригодным для production-like запуска на новых сценах.

---

## 4. Уровни тестирования

В проекте используются четыре обязательных уровня тестов:

1. **unit tests**
2. **integration tests**
3. **end-to-end smoke tests**
4. **golden / regression tests**

Дополнительно допускаются:

- property-like tests для отдельных преобразований;
- performance smoke tests;
- forensic consistency tests для manifests и run directories.

---

## 5. Unit tests

## 5.1. Назначение

Unit tests проверяют изолированные функции и маленькие компоненты без полного pipeline.

Они должны покрывать:

- чистые функции;
- validators;
- serializers;
- small adapters;
- локальную бизнес-логику без тяжелого I/O.

## 5.2. Что обязательно тестировать unit-тестами

### Геопространственная логика

- интерпретацию CRS и spatial metadata;
- проверку grid compatibility;
- raster/vector alignment helpers;
- корректность resampling/reprojection guards.

### `valid` / NoData логика

- вычисление `valid` до любых fill/replace операций;
- корректную интерпретацию nodata policy;
- mask semantics `0/1`;
- исключение invalid-пикселей из downstream масок.

### Feature logic

- сборку `raw8`;
- вычисление `NDVI`, `SAVI`, `NDWI`;
- сборку `raw8_valid` и `raw8_idx3_valid`;
- порядок каналов и `channel_semantics`.

### Target logic

- `extent` encoding;
- `boundary` encoding `background / skeleton / buffer`;
- `distance` target semantics;
- linework-faithful boundary construction.

### Normalization logic

- применение train-derived stats;
- clipping / scaling rules;
- одинаковое поведение для train и predict.

### Manifest / summary logic

- сериализацию обязательных полей;
- schema-name / schema-version consistency;
- наличие `run_id`, `data_contract_version`, provenance, resolved contract.

## 5.3. Правило для unit-тестов

Если компонент можно протестировать как чистую функцию — его следует тестировать как чистую функцию, а не через CLI-запуск всего модуля.

---

## 6. Integration tests

## 6.1. Назначение

Integration tests проверяют **стыки модулей и подсистем**, где чаще всего появляется скрытый дрейф контрактов.

## 6.2. Критические интеграционные стыки

### `module_prep_data` -> `module_net_train`

Обязательно проверять:

- наличие обязательных train-ready слоёв: `img`, `extent`, `boundary`, `distance`, `valid`, `meta`;
- корректный dataset-side `feature_mode`;
- согласованность shapes / dtypes / semantic encodings;
- возможность собрать final model input с `valid` как дополнительным каналом;
- отсутствие скрытых алиасов и fallback-подстановок target-слоёв.

### `module_net_train` -> `module_target_predict`

Обязательно проверять:

- экспорт checkpoint metadata;
- наличие `feature_mode`, `assembled_model_input`, `in_channels`, `channel_semantics`, `valid_as_input_channel`;
- восстановление predict-time feature stack без ручных догадок;
- совместимость normalization contract между train и predict.

### `module_target_predict` -> `module_postprocess_vectorize`

Обязательно проверять:

- обязательный набор outputs: `extent_prob`, `boundary_prob`, `distance_pred`, `valid`;
- геопривязку raster outputs;
- совместимость `valid` и AOI-политик;
- пригодность output-артефактов для downstream postprocess.

### `module_postprocess_vectorize` -> `module_eval`

Обязательно проверять:

- наличие `parcel_instance` и основного vector output;
- корректное сохранение provenance;
- согласованность threshold / AOI / valid policy;
- отсутствие скрытых изменений, делающих сравнение run-ов нечестным.

## 6.3. Что особенно важно ловить integration-тестами

- несовпадение dataset-side feature mode и final model input contract;
- потерю `valid` как отдельного слоя или как входного канала;
- несоответствие `in_channels` и реально собираемого input tensor;
- несовместимые manifests;
- неполное checkpoint metadata;
- молчаливую смену naming contract.

---

## 7. End-to-end smoke tests

## 7.1. Назначение

E2E smoke tests проверяют, что минимальный сквозной pipeline реально исполняется и сохраняет обязательные артефакты.

Они не обязаны подтверждать финальное качество модели, но обязаны подтверждать:

- целостность pipeline;
- совместимость конфигов и артефактов;
- отсутствие критических контрактных поломок;
- корректную запись outputs/manifests/summaries.

## 7.2. Минимальный обязательный E2E-сценарий v1

На маленьком фиксированном наборе данных должен проходить следующий путь:

```text
module_prep_data
  -> module_net_train
  -> module_target_predict
  -> module_postprocess_vectorize
  -> module_eval
```

## 7.3. Что считается успехом smoke E2E

Успешный smoke E2E должен подтверждать, что:

- каждый модуль завершился без нарушения контракта;
- каждый run сохранил `config_used`, `manifest`, `summary` и outputs;
- predict использовал checkpoint metadata автоматически;
- postprocess получил все нужные raster inputs;
- eval смог прочитать source provenance;
- нигде не потребовалась ручная правка путей, числа каналов или правил `valid`.

## 7.4. Что smoke E2E не обязан делать

Smoke E2E не обязан:

- достигать production-quality метрик;
- покрывать все режимы аппаратуры;
- прогонять большие AOI целиком;
- заменять модульные unit/integration tests.

---

## 8. Golden / regression tests

## 8.1. Назначение

Golden tests нужны для ловли тихих регрессий там, где «всё отработало», но смысл результатов сдвинулся.

## 8.2. Что имеет смысл фиксировать как golden artifacts

### Для `module_prep_data`

- маленький raster + vector пример;
- эталонные `valid`, `boundary`, `distance`;
- эталонный `split_manifest`;
- эталонная структура train-ready sample.

### Для `module_net_train`

- checkpoint metadata schema;
- структура `train_manifest`;
- dataset reader output contract на одном sample;
- стабильные диагностические summary-поля.

### Для `module_target_predict`

- predict manifest;
- список обязательных outputs;
- shape/CRS/channel-count invariants;
- сборка feature tensor для `raw8_valid` и `raw8_idx3_valid`.

### Для `module_postprocess_vectorize`

- marker-generation intermediate masks на маленьком примере;
- `parcel_instance` invariants;
- structure of exported vector attributes.

### Для `module_eval`

- schema eval manifest;
- metric summary structure;
- comparison report structure;
- provenance fields.

## 8.3. Важное ограничение

Golden tests должны сравнивать **смысловые инварианты и schema-stable outputs**, а не всё подряд byte-to-byte.

Допустимо проверять:

- обязательные поля;
- shapes;
- значения в ключевых местах;
- tolerances для численных данных;
- channel semantics;
- invariant masks.

Нежелательно делать хрупкие golden tests на весь JSON или raster целиком, если изменение несущественно по смыслу.

---

## 9. Negative tests

## 9.1. Почему они обязательны

Проект должен не только «уметь работать», но и **уметь корректно отказываться**.

## 9.2. Обязательные negative-сценарии

### Data contract errors

- отсутствует обязательный слой;
- отсутствует `valid`;
- отсутствует `channel_semantics`;
- неизвестный `feature_mode`;
- shapes таргетов не совпадают.

### Spatial errors

- CRS не согласован;
- raster/vector misalignment;
- transform drift;
- AOI в несовместимом пространственном контракте.

### Train/predict compatibility errors

- checkpoint metadata неполна;
- `in_channels` не совпадает с фактическим input tensor;
- normalization stats отсутствуют или несовместимы;
- predict не может однозначно восстановить assembled model input.

### Eval fairness errors

- сравниваются run-ы с разной valid-policy;
- сравниваются run-ы с разной AOI-policy без явной фиксации;
- threshold provenance потерян;
- source provenance неполна.

## 9.3. Что проверяет negative-тест

Negative-тест считается корректным, если:

- система действительно завершилась ошибкой;
- ошибка произошла в правильном месте;
- ошибка достаточно явная и диагностическая;
- pipeline не «проскочил дальше» с испорченным контрактом.

---

## 10. Тестирование manifests, summaries и provenance

## 10.1. Это обязательный контур

Для этого проекта manifest tests — не второстепенная вещь, а часть базового acceptance.

## 10.2. Что проверять

У каждого важного run-а должно быть тестами подтверждено наличие и корректность:

- `schema_name`
- `schema_version`
- `module_name`
- `run_id`
- `data_contract_version`
- `config_used_path`
- `source_run_ids`
- `source_manifest_paths`
- `resolved_contract`
- `runtime.device_resolved`
- `runtime.amp_used` при runtime-sensitive сценарии
- `warnings`
- `errors`

## 10.3. Forensic-ready minimum

Тест должен уметь подтвердить, что по run-артефактам можно восстановить:

- происхождение входов;
- использованный feature contract;
- роль `valid`;
- normalization rules или ссылку на их источник;
- состав outputs;
- ключевые runtime-решения.

---

## 11. Тестирование `valid` и NoData policy

## 11.1. Это один из самых критичных блоков

Поскольку `valid` имеет двойную роль — служебная mask + входной канал модели — любой дрейф здесь критичен.

## 11.2. Что обязательно покрыть

- `valid` вычисляется до NoData replacement;
- `valid` сохраняется отдельно как слой 0/1;
- invalid-пиксели исключаются из loss;
- invalid-пиксели не участвуют в eval-метриках;
- predict-time `valid` строится по тем же правилам, что и train-time;
- assembled model input обязательно включает `valid`, если того требует checkpoint contract.

## 11.3. Отдельные edge-cases

- полностью invalid tile;
- почти полностью invalid tile;
- boundary рядом с invalid edge;
- AOI, обрезающая часть valid area;
- отсутствие явного nodata в metadata при наличии sidecar/internal mask.

---

## 12. Тестирование feature contract

## 12.1. Обязательные baseline-режимы

Нужно явно тестировать оба dataset-side режима:

- `raw8`
- `raw8_idx3`

и оба финальных assembled input contracts:

- `raw8_valid`
- `raw8_idx3_valid`

## 12.2. Что проверять

- число входных каналов;
- порядок каналов;
- корректность derived indices;
- наличие `valid` как отдельного канала в final input;
- согласованность `feature_mode`, `channel_semantics`, `in_channels`, manifest metadata и runtime tensor assembly.

## 12.3. Что запрещено пропускать

Нельзя ограничиваться тестом «модель приняла tensor».  
Нужно проверять, что tensor собран **правильно по смыслу**.

---

## 13. Тестирование hardware-adaptive runtime

## 13.1. Цель

Подтвердить, что runtime адаптация под устройство и память не ломает data/model contract.

## 13.2. Что проверять

- корректное device resolution (`CUDA -> MPS -> CPU`);
- явную запись device policy в manifest;
- AMP policy;
- OOM fallback policy;
- отсутствие скрытого изменения feature contract, tile size contract или обязательных outputs.

## 13.3. Важное правило

Hardware-adaptive tests не обязаны запускаться на всех устройствах в каждом CI-прогоне.  
Но код должен позволять:

- отдельные smoke-tests на CPU;
- условные device-specific tests, если среда это поддерживает;
- проверку manifest/runtime metadata даже на CPU-only среде.

---

## 14. Тестирование CLI и конфигов

## 14.1. Что обязательно проверить

- каждый основной entrypoint запускается с `--help`;
- конфиги читаются детерминированно;
- config overrides отражаются в `config_used` / manifest;
- отсутствие скрытых fallback-подмен путей, таргетов, feature modes и runtime policies.

## 14.2. Негативные сценарии CLI

- несуществующий путь;
- отсутствующий обязательный input;
- несовместимые config-параметры;
- конфликт `feature_mode` и `in_channels`;
- попытка predict-run без достаточного checkpoint metadata.

---

## 15. Тестовые датасеты и fixtures

## 15.1. Обязательный набор тестовых данных

В репозитории должен существовать компактный test-fixture набор, который позволяет прогонять unit/integration/smoke без больших production-данных.

Минимальный набор:

- маленький 8-band GeoTIFF;
- совместимый vector GT;
- AOI-пример;
- пример с явным NoData;
- пример с выраженной внутренней границей;
- минимальный expected manifest set.

## 15.2. Fixture design rules

Fixtures должны быть:

- маленькими;
- воспроизводимыми;
- юридически и технически безопасными для хранения в репозитории;
- достаточно богатыми, чтобы покрывать ключевые edge-cases.

## 15.3. Отдельный visual benchmark set

Кроме tiny-fixtures должен существовать фиксированный небольшой visual benchmark set для ручной и полуавтоматической проверки.

Он нужен для:

- baseline vs ablation comparison;
- визуального forensic-анализа boundary quality;
- проверки downstream постпроцессинга.

Этот набор не должен незаметно меняться между сравниваемыми run-ами.

---

## 16. Структура тестов в репозитории

Рекомендуемая структура:

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
    mini_pipeline/

  golden/
    manifests/
    sample_contracts/
    raster_invariants/
    vector_invariants/

  fixtures/
    rasters/
    vectors/
    configs/
    manifests/
```

Допустима иная физическая структура, если роли тестов остаются столь же однозначными.

---

## 17. Приоритеты покрытия

Если ресурсы ограничены, приоритет тестирования должен быть таким:

1. `valid` / NoData contract;
2. feature contract;
3. manifest / provenance contract;
4. `prep_data -> train` integration;
5. `train -> predict` integration;
6. minimal E2E smoke;
7. postprocess / eval contract checks;
8. golden regressions;
9. performance / hardware-specific smoke.

Это связано с тем, что именно первые пункты чаще всего создают тихие, но разрушительные дефекты во всём downstream pipeline.

---

## 18. Что считается обязательным минимумом перед merge

Любое изменение production-like кода не должно сливаться, если не пройдены:

1. релевантные unit tests;
2. релевантные integration tests;
3. manifest/schema checks;
4. хотя бы один smoke-test для затронутого модуля или стыка;
5. negative-tests для новых contract-sensitive отказов, если они появились.

Если изменение меняет:

- feature contract;
- valid-policy;
- naming contract;
- manifest schema;
- threshold provenance;
- checkpoint metadata;

то без обновления тестов merge считаться не должен.

---

## 19. Acceptance map по модулям

## 19.1. `module_prep_data`

Модуль считается тестово прикрытым, если подтверждено:

- корректное чтение inputs;
- spatial consistency;
- корректный `valid`;
- корректная сборка `raw8` / `raw8_idx3`;
- корректное построение `extent` / `boundary` / `distance`;
- корректный экспорт `img`, `extent`, `boundary`, `distance`, `valid`, `meta`;
- запись manifests/summaries;
- корректный split export.

## 19.2. `module_net_train`

Подтверждено:

- чтение train-ready датасета;
- правильная сборка final model input;
- правильное использование ignore/valid policy;
- корректные losses/metrics masks;
- сохранение checkpoints и metadata;
- совместимость exported metadata с `module_target_predict`.

## 19.3. `module_target_predict`

Подтверждено:

- чтение checkpoint metadata;
- восстановление feature contract;
- корректная сборка predict input tensor;
- tiled inference с сохранением output contract;
- сохранение georeferenced outputs и manifests.

## 19.4. `module_postprocess_vectorize`

Подтверждено:

- чтение обязательных predict outputs;
- valid/AOI suppression работает по контракту;
- marker generation / watershed / filtering не ломают provenance;
- экспортируются `parcel_instance` и vector output;
- manifest фиксирует реальную threshold/postprocess policy.

## 19.5. `module_eval`

Подтверждено:

- чтение source provenance;
- корректный учёт valid/AOI policy;
- корректный расчёт raster / boundary / object metrics;
- корректная запись eval manifest / summary / comparison artifacts.

---

## 20. Что не считается достаточным тестированием

Недостаточно, если в проекте есть только:

- ручной запуск на одной сцене;
- визуальная проверка «на глаз»;
- один общий e2e без unit/integration coverage;
- тесты только на happy-path;
- тесты только на метрики без проверки manifests и contracts;
- тесты, которые не проверяют отрицательные сценарии;
- тесты, которые не замечают дрейф `valid` или feature contract.

---

## 21. Роль документа в разработке

`TESTING_STRATEGY.md` должен использоваться как опора для:

- построения test plan по каждому модулю;
- написания unit/integration/e2e/golden tests;
- постановки acceptance criteria перед merge;
- ревью изменений, затрагивающих data contract;
- сравнения baseline и ablation run-ов;
- проектирования CI smoke-checks и локальных validation scripts.

Новый код не должен добавляться в pipeline без понимания, **каким уровнем теста он будет подтверждён**.

