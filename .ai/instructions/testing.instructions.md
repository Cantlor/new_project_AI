# testing.instructions.md

## Назначение

Этот файл задаёт **прикладные правила написания тестов** для проекта **«ИИ для полей»**.

Он дополняет `TESTING_STRATEGY.md` и помогает превращать общую стратегию в конкретные тесты в кодовой базе.

---

## 1. Главные принципы

### 1.1. Тестируется контракт, а не только отсутствие падения

Для этого проекта недостаточно проверить, что:

- скрипт завершился;
- файл был записан;
- функция вернула массив нужной формы.

Нужно проверять, что не сломан:

- data contract;
- feature contract;
- spatial contract;
- `valid` policy;
- manifest completeness;
- downstream compatibility.

### 1.2. Негативные сценарии обязательны

Если контракт предусматривает явную ошибку, то нужен тест, который подтверждает именно это поведение.

### 1.3. Маленькие тесты лучше монолитных

Предпочтительны короткие тесты с одной ясной причиной провала, чем гигантские e2e-тесты, в которых непонятно, что именно сломалось.

---

## 2. Рекомендуемые уровни тестов

### 2.1. Unit tests

Проверяют маленькие чистые функции и локальные правила.

Примеры:

- сборка `valid` mask;
- derived indices;
- assembly `raw8_valid` / `raw8_idx3_valid`;
- boundary encoding helpers;
- manifest field validation;
- threshold helpers;
- compatibility checks.

### 2.2. Integration tests

Проверяют стык нескольких частей одного модуля или соседних модулей.

Примеры:

- `prep_data` export -> `net_train` dataset reader;
- checkpoint metadata -> `target_predict` input assembly;
- predict outputs -> postprocess input validator;
- manifests -> eval provenance resolver.

### 2.3. Golden tests

Проверяют маленький эталонный набор артефактов с ожидаемым результатом.

Они особенно полезны для:

- target encoding;
- valid/nodata semantics;
- manifests;
- output file structure.

### 2.4. Smoke e2e

Короткие end-to-end проверки на tiny data.

Их задача — проверить связность pipeline, а не качество модели.

---

## 3. Что обязательно покрывать

### 3.1. Data contract

Нужны тесты на:

- обязательные слои train-ready sample;
- canonical names;
- совместимость входов и выходов между модулями;
- корректный assembled model input.

### 3.2. `valid` / NoData

Нужны тесты на:

- вычисление `valid` до NoData replacement;
- использование `valid` в loss/metrics;
- predict-time valid reconstruction;
- invalid-only tile behavior;
- ошибки при неразрешимой valid interpretation.

### 3.3. Spatial contract

Нужны тесты на:

- CRS mismatch;
- alignment mismatch;
- raster/vector compatibility;
- georeferenced outputs;
- отсутствие тихого spatial drift.

### 3.4. Manifest completeness

Нужны тесты на:

- наличие обязательных полей;
- наличие source provenance;
- наличие resolved feature contract;
- runtime metadata, если модуль runtime-sensitive.

### 3.5. Predict compatibility

Нужны тесты на:

- несовпадение `in_channels`;
- отсутствие checkpoint metadata;
- несовместимый `feature_mode`;
- отсутствие `valid` при expected assembled input.

---

## 4. Формат тестов

### 4.1. Один тест — одна основная причина провала

Тест должен быть устроен так, чтобы по его имени и assert-ам было ясно, что он доказывает.

### 4.2. Имена тестов

Имена тестов должны отвечать на вопрос:

- что проверяется;
- при каком условии;
- какой ожидается результат.

Примеры:

- `test_build_valid_mask_uses_nodata_before_fill()`
- `test_predict_raises_when_checkpoint_in_channels_mismatch()`
- `test_split_manifest_contains_feature_mode_and_channel_semantics()`

### 4.3. Избегать тестов «на всё сразу»

Плохо:

- один тест одновременно проверяет CRS, valid, manifest и target encoding.

---

## 5. Работа с test data

### 5.1. Tiny but representative

Test data должны быть маленькими, но репрезентативными.

### 5.2. Что полезно иметь

Желательно иметь небольшой набор fixtures:

- маленький 8-band raster;
- маленький raster с NoData / invalid border;
- маленький vector GT с несколькими внутренними границами;
- маленький AOI;
- tiny train-ready sample bundle;
- tiny fake checkpoint metadata.

### 5.3. Golden artifacts

Если используется golden data, нужно явно хранить:

- что это за данные;
- почему они считаются эталонными;
- что именно по ним проверяется.

---

## 6. Assert policy

### 6.1. Проверять семантику, а не только форму

Недостаточно проверить только `.shape`.

Нужно по возможности проверять:

- допустимые значения;
- channel order;
- presence/absence of invalid pixels;
- manifest content;
- explicit errors.

### 6.2. Ошибки должны проверяться точно

Если функция обязана падать при контрактной ошибке, тест должен проверять:

- что ошибка действительно возникла;
- что сообщение отражает суть нарушения.

---

## 7. Что желательно мокать, а что нет

### 7.1. Можно мокать

Можно мокать:

- тяжёлые внешние вызовы;
- device selection детали;
- отдельные writer-слои, если цель теста — не I/O.

### 7.2. Не надо чрезмерно мокать контрактные части

Если тест проверяет data/spatial/manifest contract, чрезмерный mocking может скрыть настоящую проблему.

Например, не стоит мокать сам manifest content generator, если задача — проверить полноту manifest.

---

## 8. Negative-path tests

Отдельно нужны тесты, которые подтверждают правильные отказы.

Минимально стоит покрыть:

- отсутствующий обязательный файл;
- неверное число каналов;
- неразрешимый `valid` contract;
- checkpoint без нужной metadata;
- incompatible train/predict normalization metadata;
- spatial mismatch между связанными слоями.

---

## 9. Smoke e2e tests

### 9.1. Назначение

Smoke e2e проверяют, что минимальный pipeline проходит целиком.

### 9.2. Чего от них не ждать

Они не должны доказывать качество модели или production robustness.

### 9.3. Что они должны проверять

- пайплайн запускается;
- нужные артефакты появляются;
- manifests пишутся;
- downstream-модуль может прочитать upstream outputs без ручного вмешательства.

---

## 10. Связь тестов с manifests и experiment tracking

Тесты должны подтверждать, что run artifacts достаточны для forensic/compare workflow.

Хорошая практика — иметь проверки на:

- наличие `run_id`;
- наличие `data_contract_version`;
- наличие source provenance;
- наличие resolved feature contract;
- наличие runtime metadata там, где она обязательна.

---

## 11. Антипаттерны

Избегать:

- тестов без смыслового assert-а;
- тестов, которые проверяют только факт существования файла;
- тестов, завязанных на случайный порядок без фиксации seed;
- очень тяжёлых e2e-тестов как единственного уровня проверки;
- моков, скрывающих контрактные ошибки.

---

## 12. Чеклист перед merge

Перед merge желательно проверить:

- есть unit tests для новой core-логики;
- есть negative tests для новых контрактных ошибок;
- есть integration coverage для новых стыков;
- manifests/summaries покрыты проверками;
- новые tests читаемы и локализуют причину провала;
- тестовые фикстуры маленькие и воспроизводимые.
