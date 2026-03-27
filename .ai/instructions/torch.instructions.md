# torch.instructions.md

## Назначение

Этот файл задаёт **правила реализации train/predict логики на PyTorch** для проекта **«ИИ для полей»**.

Он предназначен прежде всего для:

- `module_net_train`
- `module_target_predict`
- частично `module_eval`, если там используется модельно-зависимая логика.

---

## 1. Главные принципы

### 1.1. Модель подчиняется контракту данных

Архитектура, dataset reader, train loop и predict adapter должны подчиняться уже принятому feature contract, а не создавать свой.

Особенно это касается:

- `raw8_valid`
- `raw8_idx3_valid`
- числа входных каналов;
- роли `valid`;
- target heads `extent / boundary / distance`.

### 1.2. Dataset-side mode и assembled model input — не одно и то же

Нужно различать:

- dataset-side feature mode: `raw8` или `raw8_idx3`;
- assembled model input: `raw8_valid` или `raw8_idx3_valid`.

Код не должен путать эти сущности ни в train, ни в predict.

### 1.3. Checkpoint must be self-describing

Checkpoint считается инженерно полноценным, только если по нему можно восстановить:

- `feature_mode`;
- assembled input contract;
- `in_channels`;
- `channel_semantics`;
- `valid` as input channel;
- normalization policy/statistics;
- target heads semantics.

---

## 2. Dataset assembly for training

### 2.1. Dataset reader должен быть строгим

Dataset reader не должен молча угадывать:

- есть ли `valid`;
- сколько каналов в `img`;
- какие target semantics используются;
- можно ли пропустить отсутствующий обязательный слой.

### 2.2. Что должно читаться явно

Для каждого train sample должно быть понятно:

- откуда читается `img`;
- откуда читается `extent`;
- откуда читается `boundary`;
- откуда читается `distance`;
- откуда читается `valid`;
- как формируется assembled input tensor.

### 2.3. `valid` в train

`valid` должен использоваться двумя способами:

1. как дополнительный входной канал;
2. как mask для ignore / masked loss / masked metrics.

Любая реализация, где один из этих путей потерян, считается несовместимой с baseline contract.

---

## 3. Model interface

### 3.1. Явный контракт входа

У модели должен быть явный ожидаемый `in_channels`.

Нельзя допускать ситуацию, когда:

- checkpoint ждёт 12 каналов;
- predict silently подаёт 9;
- код продолжает работать «как-нибудь».

### 3.2. Multi-head outputs

Модель должна ясно различать головы:

- `extent`
- `boundary`
- `distance`

Нельзя полагаться на «позицию в tuple» без явного структурного слоя, если это снижает читаемость и повышает риск перепутать outputs.

### 3.3. Deep supervision / auxiliary outputs

Если используются auxiliary outputs, их нужно:

- либо явно документировать;
- либо чётко отделять от final outputs.

Checkpoint metadata и train manifest не должны оставлять двусмысленность, какие outputs являются основными.

---

## 4. Losses и ignore policy

### 4.1. Losses должны быть valid-aware

Все train losses должны учитывать ignore/invalid policy там, где это требует контракт.

### 4.2. Extent / boundary / distance

При реализации losses нужно явно разделять:

- `extent` loss;
- `boundary` loss;
- `distance` loss;
- loss weights.

### 4.3. Не смешивать семантики target-слоёв

Нельзя писать loss-код так, будто:

- `boundary` — это просто второй бинарный extent;
- `distance` — optional noise target без значения для контракта;
- invalid pixels можно тихо пропустить «потому что так удобнее».

---

## 5. Metrics и best-checkpoint logic

### 5.1. Метрики должны быть совместимы с контрактом

Метрики должны считаться на валидных пикселях и соответствовать semantics target-слоёв.

### 5.2. Best checkpoint

Логика выбора best checkpoint должна быть:

- явной;
- конфигурируемой;
- отражённой в manifest / summary.

Если используется composite metric, это должно быть явно зафиксировано.

---

## 6. AMP, device и hardware-adaptive policy

### 6.1. Device selection

Train/predict код должен уметь явно разделять:

- requested device;
- resolved device.

### 6.2. AMP

Использование mixed precision должно быть:

- осознанным;
- управляемым;
- отражённым в runtime metadata.

### 6.3. OOM behavior

При нехватке памяти допустимы runtime-деградации, но они не должны ломать контракт данных.

Например, можно менять:

- batch size;
- accumulation;
- tile batch size.

Нельзя менять:

- число входных каналов;
- роль `valid`;
- target semantics.

---

## 7. Checkpoint policy

### 7.1. Что обязан содержать checkpoint-side metadata

Минимально нужно сохранять:

- architecture name / version;
- `feature_mode`;
- assembled model input;
- `in_channels`;
- `channel_semantics`;
- `valid_as_input_channel`;
- normalization policy/statistics source;
- head semantics;
- best metric info.

### 7.2. Predict должен зависеть от metadata, а не от ручных догадок

Если checkpoint metadata неполна, правильное поведение — явная ошибка, а не попытка угадать совместимый runtime.

---

## 8. Predict-time assembly

### 8.1. Predict должен собирать вход по checkpoint contract

Если checkpoint ожидает:

- `raw8_valid` — predict должен построить именно его;
- `raw8_idx3_valid` — predict должен сначала собрать derived indices, затем добавить `valid`.

### 8.2. Tiled inference

При tiled inference важно сохранять:

- tile size;
- overlap;
- blending policy;
- invalid-only tile skipping;
- output georeferencing.

### 8.3. Predict не должен делать postprocess задачи

Predict выдаёт raster outputs и не должен брать на себя final thresholding, watershed, polygonization и vector cleanup.

---

## 9. Reproducibility

### 9.1. Train reproducibility

Нужно по возможности сохранять:

- random seed;
- config_used;
- train manifest;
- summary;
- history/metrics artifacts;
- versions / git commit, если доступны.

### 9.2. Predict reproducibility

Нужно сохранять:

- source checkpoint;
- train run provenance;
- predict config_used;
- resolved feature contract;
- tiling/runtime info;
- output paths.

---

## 10. Кодовая организация

### 10.1. Что разделять

Желательно разделять:

- dataset code;
- model definition;
- losses;
- metrics;
- train loop;
- eval loop;
- checkpoint I/O;
- predict assembly;
- tiling utilities.

### 10.2. Что не смешивать

Не смешивать:

- model forward и postprocess heuristics;
- dataset reading и normalization stats estimation;
- checkpoint loading и silent runtime adaptation.

---

## 11. Антипаттерны

Избегать:

- скрытого изменения `in_channels`;
- потери `valid` как input channel;
- losses без ignore handling при наличии invalid pixels;
- predict-time догадок о channel order;
- checkpoint без достаточной metadata;
- hard-coded assumptions, не отражённых в manifest/config.

---

## 12. Чеклист перед merge

Перед merge желательно проверить:

- dataset reader собирает именно baseline contract;
- `valid` используется и как mask, и как input channel;
- checkpoint metadata достаточна для predict;
- losses и metrics valid-aware;
- runtime metadata пишет device/AMP/fallback info;
- predict не смешан с postprocess;
- тесты покрывают mismatch cases по channels/metadata/valid.
