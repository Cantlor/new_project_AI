# manifests.instructions.md

## Назначение

Этот файл задаёт правила для разработки кода, который:

- создаёт `manifest.*`, `summary.*`, `config_used.*`;
- валидирует manifest-артефакты;
- читает manifests upstream-модулей;
- сравнивает run-ы по provenance и resolved contract.

Эти инструкции обязательны для всех модулей проекта `ИИ для полей`:

- `module_prep_data`
- `module_net_train`
- `module_target_predict`
- `module_postprocess_vectorize`
- `module_eval`

Этот файл дополняет:

- `DATA_CONTRACT.md`
- `MANIFEST_SCHEMAS.md`
- `EXPERIMENT_TRACKING.md`
- `DECISIONS.md`

Если код противоречит этим документам, приоритет у проектных контрактов, а не у удобства реализации.

---

## Главный принцип

Manifest — это не декоративный JSON.

Manifest — это **воспроизводимый протокол запуска**, по которому можно ответить на вопросы:

- что читалось;
- что было записано;
- какой data contract был фактически использован;
- какие runtime-решения реально применились;
- из каких upstream run-ов произошёл результат;
- можно ли честно сравнивать этот run с другим.

Если по manifest нельзя восстановить эти вещи, значит manifest недостаточный.

---

## Что обязательно сохранять

Каждый модуль должен сохранять минимум:

- `config_used.*`
- `manifest.*`
- `summary.*`
- diagnostics / logs
- versioned outputs

Для runtime-sensitive модулей дополнительно обязательно или настоятельно рекомендуется сохранять:

- `device_requested`
- `device_resolved`
- `amp_requested`
- `amp_used`
- `oom_fallbacks_applied`
- warnings / errors

---

## Общие требования к manifest-writing коду

### 1. Manifest писать централизованно

Нельзя разбрасывать запись manifest по многим местам с ручной сборкой словарей.

Нужен единый слой, например:

- `manifest_schema.py`
- `manifest_models.py`
- `manifest_writer.py`
- `summary_writer.py`
- `provenance.py`

### 2. Не собирать manifest “по пути” из случайных данных

Сначала должен существовать явный runtime object/state object, который хранит:

- resolved config;
- resolved data contract;
- inputs;
- outputs;
- runtime metadata;
- warnings/errors.

И уже из него должен строиться manifest.

### 3. Manifest должен отражать **resolved state**, а не только requested state

Нельзя сохранять только то, что просили сделать.

Нужно сохранять и то, что реально получилось:

- какой `feature_mode` реально использовался;
- какой assembled model input реально собрался;
- какой device реально выбрался;
- включился ли AMP;
- были ли OOM fallback-изменения;
- какой AOI-policy реально применился.

### 4. Summary должен быть derivable из manifest

`summary.*` должен быть кратким human-facing резюме.
Но он не должен содержать критичные вещи, которых нет в manifest.

---

## Требования к provenance

### Всегда сохранять

- `run_id`
- `module_name`
- `schema_name`
- `schema_version`
- `data_contract_version`
- `created_at_utc`
- `source_run_ids`
- `source_manifest_paths`
- `source_config_paths`
- `git_commit` или другой code version marker

### Для downstream run-ов обязательно

Если модуль использует outputs другого модуля, он должен явно фиксировать:

- из какого run-а пришли данные;
- какой manifest был прочитан;
- какой config upstream считался источником истины;
- какой checkpoint / dataset / predict run стал входом.

### Никогда не делать

- не терять upstream `run_id`;
- не подменять provenance “удобными” путями без фиксации;
- не считать, что путь к файлу сам по себе уже достаточный provenance.

---

## Секция resolved_contract обязательна

Каждый важный manifest должен хранить секцию вида:

- `resolved_contract.spatial`
- `resolved_contract.features`
- `resolved_contract.valid_policy`
- `resolved_contract.normalization`
- `resolved_contract.aoi_policy`

Это особенно критично для проекта, где:

- dataset-side feature mode и assembled model input — не одно и то же;
- `valid` имеет двойную роль;
- predict является checkpoint-driven;
- AOI может влиять на reporting / masking / output extent;
- normalization должна быть согласована между train и predict.

---

## Секция artifacts

### Каждый artifact должен описываться как объект, а не строка пути

Минимальные поля:

- `path`
- `role`
- `format`
- `is_required`
- `exists`
- `checksum` при возможности
- `size_bytes` при возможности

Для raster-артефактов дополнительно по возможности:

- `crs`
- `transform`
- `width`
- `height`
- `count`
- `dtype`
- `nodata`
- `channel_semantics`

### Почему это важно

В этом проекте путь к файлу без пространственной и семантической metadata почти бесполезен для forensic-аудита.

---

## Требования к runtime metadata

Для модулей `module_net_train`, `module_target_predict`, `module_eval` и при необходимости `module_postprocess_vectorize` manifest должен хранить:

- `device_requested`
- `device_resolved`
- `amp_requested`
- `amp_used`
- `oom_fallbacks_applied`
- runtime warnings

Нельзя скрывать fallback-решения, если они меняли поведение run-а.
Например:

- уменьшение batch size;
- выключение AMP;
- снижение tile batch size;
- смена overlap / tiling policy из-за ограничений памяти.

---

## Правила ошибок и статусов

### Поле `status`

Допустимые базовые значения:

- `success`
- `partial`
- `failed`

### Когда использовать `partial`

`partial` допустим только когда:

- часть outputs успешно записана;
- run можно диагностировать;
- явно описано, что именно не завершилось.

### Ошибки всегда сохранять

Если run завершился ошибкой, manifest всё равно желательно записать, если это технически возможно, с:

- `status: failed`
- списком `errors`
- частично собранным provenance
- списком outputs, которые успели появиться

---

## Контрактные инварианты, которые manifest обязан отражать

### Для `module_prep_data`

Обязательно фиксировать:

- `feature_mode`
- `feature_channel_count`
- `channel_semantics`
- наличие/роль `valid`
- patch size
- sampling policy
- split policy
- output structure

### Для `module_net_train`

Обязательно фиксировать:

- dataset source run
- `feature_mode`
- assembled model input
- final input channel count
- `channel_semantics`
- `valid_as_input_channel`
- loss names and weights
- best metric / checkpoint
- normalization contract

### Для `module_target_predict`

Обязательно фиксировать:

- train run provenance
- checkpoint metadata provenance
- resolved feature reconstruction
- normalization source
- tiling policy
- invalid tile handling
- output rasters

### Для `module_postprocess_vectorize`

Обязательно фиксировать:

- source predict run
- threshold policy
- marker policy
- watershed policy
- filtering / cleanup policy
- vector export format

### Для `module_eval`

Обязательно фиксировать:

- source run ids
- GT sources
- scene selection policy
- valid / AOI policy
- threshold provenance
- comparison mode
- enabled metric groups

---

## Требования к коду валидаторов manifests

### Валидировать нужно не только форму, но и смысл

Недостаточно проверить, что поле существует.
Нужно проверять и смысловую согласованность:

- `feature_channel_count` согласован с `channel_semantics`;
- `assembled_model_input` согласован с `final_input_channel_count`;
- `valid_as_input_channel` не противоречит feature contract;
- `status=success` не сочетается с отсутствием обязательных outputs;
- `source_run_ids` не пусты там, где без upstream provenance запуск невозможен.

### Разделять schema validation и semantic validation

Рекомендуемый подход:

- `schema validation` — структура, типы, required поля;
- `semantic validation` — проектные инварианты и взаимная согласованность.

---

## Формат файлов

### Предпочтение

- JSON — для машиночитаемых manifests
- YAML — допустим для config-like metadata, если это уже принято в модуле

### Общее правило

В пределах одного artifact type формат должен быть стабилен.
Нельзя в одном месте писать `summary.json`, а в другом `summary.yaml`, если это не зафиксировано проектным решением.

---

## Naming conventions

Использовать канонические имена из `MANIFEST_SCHEMAS.md`.

Примеры:

- `check_inputs_manifest.json`
- `features_manifest.json`
- `split_manifest.json`
- `train_manifest.json`
- `checkpoint_metadata.json`
- `predict_manifest.json`
- `postprocess_manifest.json`
- `eval_manifest.json`
- `summary.json`

Не придумывать новые имена без отдельного решения.

---

## Что запрещено

Запрещено:

- писать manifest как побочный лог без схемы;
- терять `run_id` и upstream provenance;
- не фиксировать resolved feature contract;
- не фиксировать normalization source;
- не фиксировать runtime fallback-решения;
- прятать AOI/valid/threshold policy внутри кода без отражения в manifest;
- делать manifest необязательным для supposedly reproducible run.

---

## Как использовать эти инструкции с ИИ-ассистентами

Когда ассистент пишет код для manifests, требовать от него:

1. отдельные модели данных / dataclasses / pydantic-схемы;
2. отдельный writer layer;
3. отдельный validator layer;
4. тесты на schema + semantic validation;
5. примеры manifest outputs;
6. проверку соответствия `MANIFEST_SCHEMAS.md`.

Нельзя просить: “просто добавь манифест как-нибудь”.

---

## Чеклист перед merge

Перед merge проверять:

- manifest создаётся всегда для успешного run-а;
- `run_id` стабильно присутствует;
- upstream provenance не потерян;
- resolved contract действительно записан;
- artifacts перечислены явно;
- runtime metadata зафиксирована;
- summary не противоречит manifest;
- schema validation и semantic validation проходят;
- golden examples manifest-ов обновлены при изменении схемы.
